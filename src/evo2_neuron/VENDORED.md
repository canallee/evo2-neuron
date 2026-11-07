# Vendored: Evo 2 / StripedHyena 2 (Arc Institute)

## Source

| | |
|---|---|
| model definition | [`Zymrael/vortex`](https://github.com/Zymrael/vortex) — `vortex/model/{model,engine,layers,attention,rotary,utils}.py` |
| architecture config | [`ArcInstitute/evo2`](https://github.com/ArcInstitute/evo2) — `evo2/configs/evo2-{1b,7b}-8k.yml` |
| weights | [`arcinstitute/evo2_7b_base`](https://huggingface.co/arcinstitute/evo2_7b_base) — `evo2_7b_base.pt` |
| license | **Apache-2.0** (both repos) |
| read at | vortex `a37a9b4` (the last **0.0.2** commit) cross-checked against HEAD `8b00afe`; evo2 `53f1959` |

**Pin evo2 0.1.0 for the 7B config.** 0.6.0 flips `use_fp8_input_projections`
False → True and adds `interleave: True` that vortex 0.0.2 omits. vortex's config
object is a `dotdict` with `__getattr__ = dict.get`, so an absent key silently
reads as off — a config-version mismatch produces a running model with different
numerics and no error. The values in `config.py` are transcribed by hand from the
0.1.0 YAML for exactly this reason, with every divergence annotated inline.

Weights are Arc's own. Third-party HuggingFace re-uploads are **not** used for
either code or weights: at least one stores a pre-transformed `Wqkv` that
silently conflicts with Arc's loader.

This is a **re-implementation against the source**, not a file copy: vortex's
modules are entangled with Triton kernels, TransformerEngine and multi-GPU
placement, none of which exist on this target. Every module here mirrors a named
vortex class, so an Arc checkpoint loads with no key renaming at all.

## Verification

`blocks.10.mlp.l3` on the CUDA reference's own input — this port vs vortex 0.0.2,
both on CPU, evo2-7b:

```
rel_err = 2.145839e-06   max_abs = 5.215406e-07   cosine = 1.0000000
```

i.e. fp32 reduction-order noise. The device and CUDA axes are in
[`docs/PARITY.md`](../../docs/PARITY.md).

## Patches

### 1. FFT long convolution → flipped depthwise causal `conv1d` (`hyena.py`)

`neuronx-cc` cannot compile complex/FFT ops (`NCC_EVRF004`); in eager the torch
fallback errors outright (`aten::_fft_c2r ... complex dtypes are not supported
natively on Neuron`). Two sites replaced:

* `fftconv_func` — the hcm inner filter (length 128)
* `parallel_iir` — the hcl modal-FFT filter (length L)

A linear convolution with a real finite filter is exactly a depthwise causal
`F.conv1d` with the filter flipped and left padding `K-1`. The FFT normalisation
(`rfft(k)/n` combined with `irfft(..., norm="forward")`) cancels to a **plain
unscaled** convolution, so no `1/n` factor is introduced. Computed in fp32 to
match vortex's own fp32 FFT accumulation.

**Do not flip the short filters.** vortex applies raw `F.conv1d` —
cross-correlation, unflipped — to the outer length-3 featurizer and the hcs
length-7 inner filter. That asymmetry is part of the trained definition. Only the
two FFT sites above are true convolutions.

### 2. TransformerEngine `TELinear` → `nn.Linear` (`layers.py`, `hyena.py`)

`use_fp8_input_projections` requires TE and a Hopper GPU. The parameter **name
and shape are preserved** (`projections.weight`, `[3*hidden, hidden]`) so Arc's
checkpoint loads without a key map. TE's FP8 `_extra_state` blobs are dropped —
they carry only amax/scale history, and the stored weights are unquantised
(verified: `blocks.0.projections.weight` absmax `0.2890625` equals the recorded
amax `0.2891`). 33 blobs are dropped on evo2-7b, 26 on evo2-1b.

Arc's **7B** config sets `use_fp8_input_projections: False`, so on 7B there is
nothing for the port to emulate and the CUDA axis can be gated directly. The 1B
config sets it `True`; disabling it there costs 47 accuracy points, so evo2-1b is
**not** gateable against a CUDA reference and is shipped here for shape work only.

### 3. flash-attn → eager causal attention (`attention.py`)

vortex selects `FlashSelfAttention` or its own `SelfAttention`; the latter is
plain causal `scaled_dot_product_attention` and is what runs when flash-attn is
absent — including in the H200 environment that minted the CUDA reference. An
explicit `softmax(QKᵀ/√d + mask)V` variant is also provided (`use_sdpa=False`)
because the fused SDPA *backward* is a known Neuron failure at longer sequences.

### 4. flash-attn RMSNorm and Triton RoPE → plain torch (`layers.py`, `rotary.py`)

Both are vortex's own fallbacks, not new math. `RMSNorm` keeps vortex's exact eps
placement (added to the norm, **outside** the reciprocal). Its L2 norm runs
through a small `autograd.Function` (`_L2Norm`): same forward call, and PyTorch's
own backward formula with the zero mask compared against `0.0` rather than `0`.
On torch-neuronx 2.14 the built-in backward's integer comparison is converted on
the host — one `aten::_to_copy` fallback per RMSNorm per backward. The gradient
is bit-identical to autograd's on CPU in fp32 and bf16. RoPE recomputes
`inv_freq` in fp32, matching vortex, which ignores the checkpoint's bf16
`inv_freq` buffer whenever it is not already fp32. The recompute also avoids
`aten::pow.Scalar`, which falls back to the host on Neuron in bf16.

### 5. KV cache / decode → removed

`use_cache=False`, static prefill only. Generation is out of scope: a read-out
head never decodes, and the next-token objective is teacher-forced.

### 6. Multi-GPU placement, FlashFFTConv, activation logging → removed

`block_idx_to_device`, `cross_device_transfer`, `flashfftconv`, and vortex's
`print_activations` instrumentation. None are reachable on this target.

## Load-time transform that is NOT optional

`loading.convert_wqkv_column_split` ports vortex's savanna→vortex QKV relayout
(`vortex/model/model.py:880-904`, gated on `column_split: True`, which every Evo2
config sets — the only place that key is read).

Arc stores `Wqkv` head-major `(heads, 3, head_dim)`; `MHA.forward` unpacks
three-major. Skipping the transform changes the read-out by rel_err ~1.37 and
silently scrambles q/k/v across heads in every attention block. There is no
error. If a port is "almost right", check this first.

## Deliberate structural difference

`StripedHyena(config, max_blocks=N)` builds only the first `N` blocks. vortex
loads all of them and then replaces the tail with raising sentinels to reclaim
VRAM; building fewer is the same optimisation applied earlier, and it means a
mid-stack read-out needs no exception raised from a forward hook to short-circuit
the stack — which would be hostile to graph capture on device.
