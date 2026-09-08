"""
Hyena cascade + the two block types, ported from `vortex/model/{model,engine}.py`.

This file holds the port's only real math rewrite: **FFT -> depthwise causal
conv1d**. `neuronx-cc` cannot compile complex/FFT ops (`NCC_EVRF004`), and in
eager the torch fallback errors outright (`aten::_fft_c2r ... complex dtypes are
not supported natively on Neuron`). Both are marked `NEURON PATCH` below.

A trap that is easy to get backwards, so it is spelled out:

  * The **short** filters -- the outer `short_filter_weight` (length 3) and the
    hcs inner filter (length 7) -- go through vortex's raw `F.conv1d`, which is
    cross-correlation, NOT convolution. vortex does not flip them. Neither do we.
    That asymmetry is part of the trained definition, not a bug to fix.
  * The **hcm** (length 128) and **hcl** (length L) filters go through FFT, which
    IS true convolution. Those are the two that must be flipped when rewritten as
    `conv1d`.

Flipping the wrong set silently produces a plausible-looking but wrong model.

Normalization invariant for the rewrite: vortex computes
`irfft(rfft(u) * rfft(k)/n, n, norm="forward")`. `norm="forward"` puts the 1/n on
the forward transform, so the inverse applies none; combined with the explicit
`/n` on the filter, the two cancel and the result is a **plain, unscaled** linear
convolution. The conv1d replacement therefore needs no 1/n factor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from evo2_neuron.attention import MHA
from evo2_neuron.layers import ParallelGatedMLP, RMSNorm

# ----------------------------------------------------------------------------
# NEURON PATCH -- FFT long convolution -> flipped depthwise causal conv1d
# ----------------------------------------------------------------------------


def causal_depthwise_conv(
    u: Tensor, k: Tensor, seqlen: int, compute_dtype: torch.dtype = torch.float32
) -> Tensor:
    """Linear causal convolution of `u` (B, C, L) with per-channel filter `k`.

    `k` is (C, 1, K). Replaces
        `irfft(rfft(u, 2L) * rfft(k, 2L)/2L, 2L, norm="forward")[..., :L]`
    exactly (see module docstring on why there is no scale factor).

    `F.conv1d` cross-correlates, so the filter is flipped to turn it into a true
    convolution: with `w[j] = k[K-1-j]` and left padding `K-1`,
        `out[t] = sum_j w[j] u[t+j-(K-1)] = sum_s k[s] u[t-s]`.

    Computed in fp32 to match vortex's own fp32 FFT accumulation, then cast back.
    """
    channels, _, ksize = k.shape
    weight = k.flip(-1).to(compute_dtype)

    y = F.conv1d(
        u.to(compute_dtype), weight, bias=None, stride=1, padding=ksize - 1, groups=channels
    )

    return y[..., :seqlen]


def fftconv_func(u: Tensor, k: Tensor, D: Tensor | None, seqlen: int) -> Tensor:
    """NEURON PATCH for `engine.fftconv_func` (hcm inner filter, length >= 128).

    Original: FFT conv, then `out = y + u * D.unsqueeze(-1)`. Only the FFT is
    replaced; the `D` skip term is unchanged.
    """
    if k.dim() == 2:  # (C, K) -> (C, 1, K)
        k = k.unsqueeze(1)

    y = causal_depthwise_conv(u, k, seqlen)

    if D is not None:
        y = y + u.to(torch.float32) * D.to(torch.float32).unsqueeze(-1)

    return y.to(u.dtype)


# ----------------------------------------------------------------------------
# Engine (the parts of HyenaInferenceEngine that prefill actually reaches)
# ----------------------------------------------------------------------------


def interleave(z_pre: Tensor) -> Tensor:
    """vortex `utils.interleave`, 3-D branch.

    The projections come out channel-interleaved (x1/x2/v strided by 3); this
    de-interleaves them into three contiguous blocks so the later
    `split([H, H, H])` lines up.
    """
    return torch.cat([z_pre[:, 0::3, :], z_pre[:, 1::3, :], z_pre[:, 2::3, :]], dim=1)


def parallel_fir(
    u: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    seqlen: int,
    *,
    hidden_size: int,
    fir_length: int,
    gate: bool,
    gated_bias: bool = False,
    dim_last: bool = True,
) -> Tensor:
    """vortex `engine.HyenaInferenceEngine.parallel_fir`, prefill branches only.

    Two call sites:
      * outer featurizer -- `gate=False`, `dim_last=True`, `fir_length=3`
      * hcs/hcm inner    -- `gate=True`,  `dim_last=False`
    """
    if gate:
        # column_split_hyena is False for every Evo2 config, so this is a plain
        # three-way split along the channel axis.
        x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
        u = x1 * v

    if fir_length >= 128:
        # hcm: the FFT branch. `bias` is D and is folded in by fftconv_func.
        z = fftconv_func(u, weight[:, :, :seqlen], bias, seqlen)
    else:
        if dim_last:
            u = u.permute(0, 2, 1)  # (B, L, D) -> (B, D, L)

        # NOT flipped -- see module docstring. `groups` is always the full channel
        # count regardless of the filter's grouping.
        z = F.conv1d(
            u.to(torch.float32),
            weight.to(torch.float32),
            bias=None,
            stride=1,
            padding=fir_length - 1,
            groups=u.shape[1],
        )[..., :seqlen]
        z = z.to(u.dtype)

        if bias is not None:
            z = z + (bias[None, :, None] * u if gated_bias else bias[None, :, None])

    if gate:
        z = x2 * z

    return z


def modal_causal_conv_chunked(
    x: Tensor, log_poles: Tensor, residues: Tensor, seqlen: int, chunk: int,
    compute_dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Exact linear-time evaluation of the hcl (long) Hyena convolution.

    `causal_depthwise_conv` is O(L^2) here, because the hcl filter's length IS the
    sequence length. Measured: seq 200 -> 512 costs 61x on Neuron, so the direct
    form does not reach the production 8192 context. This is the same convolution
    computed in O(L * chunk + L * state_size) instead.

    It works because the filter is not arbitrary -- it is modal:

        h[s] = sum_j r_j * a_j^s          with  a_j = exp(log_poles_j)

    so a convolution against it satisfies a recurrence. Writing `t = n*chunk + i`
    and splitting the sum at the chunk boundary:

        y[t] = sum_{s<=i} h[s] x[t-s]           <- intra-chunk, a short conv
             + sum_j r_j a_j^i * S_j[n]         <- everything before this chunk

        S_j[n+1] = a_j^chunk * S_j[n] + sum_{i<chunk} a_j^(chunk-i) x[n*chunk+i]

    `S` is the running modal state, `(B, D, state_size)`. Only `chunk` taps of the
    filter are ever materialised, and the loop runs `L/chunk` times rather than
    `L`, which keeps the traced graph small -- a per-timestep recurrence would be
    exact too but would unroll to 8192 iterations and is hopeless to compile.

    Algebraically identical to the direct form; verified against it in
    `tests/test_trn_ops.py`. All poles are negative (max -0.0012 on evo2-1b), so
    `a_j^chunk` is a contraction and the state cannot blow up.
    """
    batch, channels, _ = x.shape
    # fp32 by default, matching `causal_depthwise_conv`. Overridable so the
    # equivalence proof in tests can run in fp64, where the two forms agree to
    # ~1e-15; in fp32 they differ by ~1e-7, which is reduction-order round-off
    # over a length-L accumulation, not disagreement.
    dtype, device = compute_dtype, x.device

    x = x.to(dtype)
    log_poles = log_poles.to(dtype).reshape(channels, -1)  # (D, S)
    residues = residues.to(dtype)                          # (D, S)

    offsets = torch.arange(chunk, device=device, dtype=dtype)
    # a^i for i in [0, chunk) -- the within-chunk decay, and its reverse for the
    # state update. Built in log space so nothing is raised to a tensor power.
    decay = torch.exp(log_poles[:, :, None] * offsets)                    # (D, S, chunk)
    decay_rev = torch.exp(log_poles[:, :, None] * (chunk - offsets))      # (D, S, chunk)
    decay_full = torch.exp(log_poles * chunk)                             # (D, S)

    # Short filter for the intra-chunk part: h[s] for s < chunk.
    h_short = (residues[..., None] * decay).sum(1)                        # (D, chunk)

    state = torch.zeros(batch, channels, log_poles.shape[1], device=device, dtype=dtype)
    pieces = []

    for start in range(0, seqlen, chunk):
        block = x[..., start : start + chunk]
        width = block.shape[-1]

        # Intra-chunk: sum_{s<=i} h[s] x[t-s]. Convolved PER CHUNK, not globally --
        # a global conv with a chunk-length kernel would let taps reach back across
        # the boundary, which the state term below already accounts for, and the
        # overlap would be silently double-counted.
        intra = causal_depthwise_conv(
            block, h_short[:, :width].unsqueeze(1), width, compute_dtype=compute_dtype
        )

        # Everything before this chunk: sum_j r_j a_j^i S_j.
        inter = torch.einsum("bds,ds,dsi->bdi", state, residues, decay[..., :width])
        pieces.append(intra + inter)

        # Roll the state forward across this chunk:
        #   S[n+1] = a^chunk * S[n] + sum_i a^(width-i) x[start+i]
        state = state * decay_full + torch.einsum(
            "bdi,dsi->bds", block, decay_rev[..., chunk - width :]
        )

    return torch.cat(pieces, dim=-1)


_NKI_SCAN = None

#: Sequence tile for `_modal_scan`. Bounds the kernel's redundant `decay` buffer
#: at O(C*S*tile) instead of O(C*S*L). At C=4096 S=16 fp32 a 1024-tile costs
#: 256 MiB against 2 GiB for a whole 8192 sequence -- and 2 GiB is exactly the
#: allocation that failed on the first 8192 gate. The kernel tiles internally at
#: F_TILE_SIZE=2048, so going below that trades throughput for memory; 1024 is
#: one notch under, chosen to leave headroom on a core already holding ~9 GB of
#: weights. Exact at any value: the scan resumes from `final_state`.
_NKI_SCAN_TILE = 1024


def _nki_linear_scan():
    """Lazily resolve nki-library's `linear_scan`, wrapped for torch dispatch.

    Imported lazily and cached because `wrap_nki` registers a custom op with
    torch the first time it runs -- doing that at module import would make
    `evo2_neuron` depend on nki and on a Neuron device, which the CPU parity
    path must not.
    """
    global _NKI_SCAN

    if _NKI_SCAN is None:
        from nkilib.experimental.scan.linear_scan import linear_scan
        from torch_neuronx import wrap_nki

        _NKI_SCAN = wrap_nki(linear_scan)

    return _NKI_SCAN


class _ModalCausalConvNKI(torch.autograd.Function):
    """Autograd wrapper for the NKI modal scan.

    NEEDED BECAUSE THE KERNEL IS NOT DIFFERENTIABLE. `wrap_nki` produces a custom
    op with no autograd formula, so its output carries no `grad_fn`. Calling it
    bare inside `parallel_iir` does NOT raise -- the surrounding `D` skip term and
    `x2` gate re-attach a `grad_fn`, so the block still produces *a* gradient,
    just one missing the convolution's entire contribution to dL/dx. That is a
    silently wrong gradient for every adapter below an hcl block.

    Measured before this class existed: `g1` vs the direct conv had median
    1.6e-06 but max **6.5e-01**, on `blocks.1.mlp.l2.lora_B` -- block 1 sits below
    hcl block 2, exactly where the missing path shows. The magnitude-aware gate
    passed it because `lora_B` is zero-initialised and its gradient is tiny in
    absolute terms. Do not trust that gate alone for a structurally missing path.

    The backward is cheap and exact. For a causal convolution
    `y[t] = sum_s h[s] x[t-s]`,

        dL/dx[u] = sum_{t>=u} gy[t] h[t-u]

    which is the same convolution run backwards in time -- so it is another modal
    scan on the reversed gradient, reversed again on the way out.

    Only dL/dx is produced. `log_poles` and `residues` are backbone parameters,
    and a LoRA fine-tune freezes the backbone, so their gradients are never
    requested. `backward` asserts that rather than silently returning None for a
    tensor that needed one.
    """

    @staticmethod
    def forward(ctx, x, log_poles, residues, seqlen):
        ctx.save_for_backward(log_poles, residues)
        ctx.seqlen = seqlen
        ctx.needs_pole_grad = log_poles.requires_grad or residues.requires_grad

        return _modal_scan(x, log_poles, residues, seqlen)

    @staticmethod
    def backward(ctx, grad_out):
        log_poles, residues = ctx.saved_tensors

        if ctx.needs_pole_grad:
            raise NotImplementedError(
                "modal_causal_conv_nki has no gradient for log_poles/residues. "
                "They are frozen backbone parameters in the LoRA fine-tune; if "
                "you are training them, use hcl_kernel='direct' or 'chunked'."
            )

        grad_x = _modal_scan(
            grad_out.flip(-1).contiguous(), log_poles, residues, ctx.seqlen
        ).flip(-1)

        return grad_x, None, None, None


def _modal_scan(
    x: Tensor, log_poles: Tensor, residues: Tensor, seqlen: int
) -> Tensor:
    """The hcl convolution as a first-order recurrence, via nki-library.

    Same identity `modal_causal_conv_chunked` uses -- the hcl filter is modal, so
    the convolution satisfies a recurrence:

        h[s] = sum_j r_j a_j^s,  a_j = exp(log_poles_j)
        y[t] = sum_s h[s] x[t-s] = sum_j r_j s_j[t]
        s_j[t] = a_j s_j[t-1] + x[t]

    which is exactly `linear_scan`'s `result[t] = decay[t]*result[t-1]+data[t]`.
    Verified fp64-exact against the direct convolution (rel_err 5.2e-10), and the
    kernel reproduces a sequential reference bit-exactly.

    Why it exists when `modal_causal_conv_chunked` already computes this: at the
    production 8192 context the chunked form works but is 3.11x slower and takes
    10.5x longer to compile (0.308 s / 115 s vs 0.099 s / 11 s). The direct
    convolution does not compile at all past ~seq 512.

    COST: `decay` is constant along the sequence but must be materialised at
    (B*C*S, L) anyway -- 2.15 GB at L=8192, C=4096, S=16. `hcl_chunk` bounds live
    memory at O(C*chunk) instead and is the right fallback if that becomes the
    binding constraint. See `docs/OPTIMIZATIONS.md`.
    """
    batch, channels, _ = x.shape
    states = residues.shape[1]
    rows = batch * channels * states

    compute_dtype = torch.float32
    poles = torch.exp(log_poles.squeeze(-1).to(compute_dtype))            # (C, S)
    x = x.to(compute_dtype)
    weights = residues.to(compute_dtype).unsqueeze(0).unsqueeze(-1)

    # Tile along the sequence, carrying the modal state across tiles.
    #
    # `decay` is CONSTANT along the sequence but the kernel takes it at (rows, L)
    # anyway, so a whole-sequence call materialises C*S*L floats of pure
    # redundancy -- 4096*16*8192*4 = exactly 2 GiB at the production context.
    # That is not hypothetical: the first 8192 gate died on
    # `Failed to allocate DEVICE memory (2147483648 bytes)`, having worked
    # standalone, because in the full model it lands on top of 8.9 GB of weights
    # plus activations.
    #
    # `linear_scan` returns its `final_state` and accepts an `initial`, so the
    # recurrence can be resumed exactly across tile boundaries -- no
    # approximation, and live memory becomes O(C*S*tile) instead of O(C*S*L).
    tile = _NKI_SCAN_TILE
    scan = _nki_linear_scan()
    state = None
    pieces = []

    for start in range(0, seqlen, tile):
        width = min(tile, seqlen - start)

        decay_tile = poles.reshape(1, channels, states, 1).expand(
            batch, channels, states, width
        ).reshape(rows, width).contiguous()
        data_tile = x[..., start : start + width].reshape(
            batch, channels, 1, width
        ).expand(batch, channels, states, width).reshape(rows, width).contiguous()

        chunk, state = scan(decay_tile, data_tile, state) if state is not None \
            else scan(decay_tile, data_tile)

        pieces.append((chunk.reshape(batch, channels, states, width) * weights).sum(2))

    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)


def modal_causal_conv_nki(
    x: Tensor, log_poles: Tensor, residues: Tensor, seqlen: int
) -> Tensor:
    """Differentiable entry point -- always call this, never `_modal_scan`."""
    return _ModalCausalConvNKI.apply(x, log_poles, residues, seqlen)


def parallel_iir(
    z_pre: Tensor,
    h: Tensor,
    D: Tensor,
    seqlen: int,
    *,
    hidden_size: int,
    log_poles: Tensor | None = None,
    residues: Tensor | None = None,
    chunk: int | None = None,
    kernel: str = "direct",
) -> Tensor:
    """vortex `engine.HyenaInferenceEngine.parallel_iir`, prefill FFT branch.

    NEURON PATCH: the modal-FFT convolution becomes a flipped depthwise causal
    conv1d. Everything around it -- the split, the `x1 * v` pregate, the `D` skip
    term, the `x2` postgate and the final permute -- is unchanged.

    `chunk` selects the linear-time modal form instead (`hcl_chunk` in the
    config). Off by default: the direct conv is the validated path at seq 200,
    and the chunked form only earns its keep at long context.
    """
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    x1v = x1 * v

    if kernel == "nki-scan":
        # Same modal recurrence as `chunked`, evaluated by nki-library's
        # `linear_scan` instead of a python chunk loop.
        y = modal_causal_conv_nki(x1v, log_poles, residues, seqlen)
    elif chunk:
        # Linear-time modal form. Needs the poles/residues rather than the
        # materialised filter, which is why they are threaded through here.
        y = modal_causal_conv_chunked(x1v, log_poles, residues, seqlen, chunk)
    else:
        # h is (1, D, L) from compute_filter -> (D, 1, L) for depthwise conv1d.
        y = causal_depthwise_conv(x1v, h[0].unsqueeze(1), seqlen)

    y = y.to(x1v.dtype)
    y = (y + x1v * D.unsqueeze(-1)) * x2

    return y.permute(0, 2, 1)


# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------


class HyenaCascade(nn.Module):
    """vortex `model.HyenaCascade`.

    Three variants, selected by `get_block` from the layer index:
      * hcs -- `fir_inner_filter_length=7`,   grouped, no `D`
      * hcm -- `fir_inner_filter_length=128`, grouped, has `D`
      * hcl -- no inner FIR; a modal (pole/residue) filter of length L, has `D`

    Group counts are config-driven and differ per backbone (evo2-1b: 128 for
    hcs/hcm; evo2-7b: 256). Whether a cascade has a `D` skip term at all is NOT
    -- it is driven by `fir_inner_filter_length >= 128`, and 128/7 are the same
    on both backbones, so the state-dict key set is unchanged. Do not confuse
    that threshold with the group count; they collide at 128 on evo2-1b only.
    """

    def __init__(self, config, layer_idx: int, hyena_filter_groups=None, fir_inner_filter_length=None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.state_size = config.state_size
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = self.hidden_size // self.num_attention_heads
        self.short_filter_length = config.short_filter_length
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = (
            hyena_filter_groups if hyena_filter_groups is not None else config.hidden_size
        )

        self.short_filter_weight = nn.Parameter(
            torch.randn(3 * config.hidden_size, 1, config.short_filter_length)
        )
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * config.hidden_size)) if config.short_filter_bias else None
        )

        if fir_inner_filter_length:
            self.h = nn.Parameter(torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length))
            # Only the long (>=128) inner filter carries a skip term.
            self.D = nn.Parameter(torch.zeros(self.hidden_size)) if fir_inner_filter_length >= 128 else None
            self.log_poles = None
            self.residues = None
        else:
            self.h = None
            self.log_poles = nn.Parameter(
                torch.randn(self.hyena_filter_groups, self.state_size, 1, dtype=torch.float32)
            )
            self.residues = nn.Parameter(
                torch.randn(self.hyena_filter_groups, self.state_size, dtype=torch.float32)
            )
            self.D = nn.Parameter(torch.zeros(self.hidden_size))

    def compute_filter(self, seqlen: int, device) -> Tensor:
        """vortex `model.HyenaCascade.compute_filter` -- a real modal filter.

        `h[c, t] = sum_s residues[c, s] * exp(log_poles[c, s] * t)`, built in fp32.
        Both parameters are real, so nothing here is complex -- only the
        convolution that consumes `h` ever was.
        """
        t = torch.arange(seqlen, device=device, dtype=torch.float32)[None, None]
        residues = self.residues.to(torch.float32)
        log_poles = self.log_poles.to(torch.float32)

        return (residues[..., None] * (log_poles * t).exp()).sum(1)[None]  # (1, D, L)

    def forward(self, u: Tensor) -> Tensor:
        seqlen = u.shape[1]

        z_pre = parallel_fir(
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            seqlen,
            hidden_size=self.hidden_size,
            fir_length=self.short_filter_length,
            gate=False,
            dim_last=True,
        )

        if self.config.interleave:
            z_pre = interleave(z_pre)

        h = self.h if self.h is not None else self.compute_filter(seqlen, u.device)

        if self.hyena_filter_groups > 1:
            # Integer division with no remainder check used to be enough by
            # accident: 1920/128 = 15 exactly. 7B is 4096/256 = 16, also exact,
            # but a typo'd group count would silently truncate the ratio and
            # expand `h` to the wrong channel count, surfacing as an opaque
            # `F.conv1d` shape error several frames away. Say what is wrong.
            if self.hidden_size % self.hyena_filter_groups:
                raise ValueError(
                    f"hidden_size {self.hidden_size} is not divisible by "
                    f"hyena_filter_groups {self.hyena_filter_groups}"
                )

            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        if self.fir_inner_filter_length is not None:
            y = parallel_fir(
                z_pre,
                h,
                self.D,
                seqlen,
                hidden_size=self.hidden_size,
                fir_length=self.fir_inner_filter_length,
                gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                dim_last=False,
            )
            return y.permute(0, 2, 1)

        return parallel_iir(
            z_pre, h, self.D, seqlen,
            hidden_size=self.hidden_size,
            log_poles=self.log_poles,
            residues=self.residues,
            chunk=self.config.get("hcl_chunk", None),
            kernel=self.config.get("hcl_kernel", "direct"),
        )


class ParallelGatedConvBlock(nn.Module):
    """vortex `model.ParallelGatedConvBlock` -- the Hyena block (hcs/hcm/hcl).

    `projections` is a TransformerEngine `Linear` upstream; here it is a plain
    `nn.Linear` under the same name and shape. FP8 is off, so vortex's
    `pad_to_multiple(16)` is a no-op and is dropped.
    """

    def __init__(self, config, layer_idx: int, hyena_filter_groups=None, fir_inner_filter_length=None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.pre_norm = RMSNorm(config)
        self.post_norm = RMSNorm(config)
        self.filter = HyenaCascade(
            config,
            layer_idx,
            hyena_filter_groups=hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
        )
        # Direct attribute reads, not `.get(...)`: this key used to be read here
        # with default False and in `AttentionBlock` with default True, so an
        # absent key silently gave the two blocks opposite bias geometry. Every
        # Evo2 YAML sets it explicitly (False); a missing key is a config bug and
        # should raise.
        self.projections = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.qkv_proj_bias
        )
        self.out_filter_dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.get("hyena_out_proj_bias", True)
        )
        self.mlp = ParallelGatedMLP(config, layer_idx)

    def forward(self, u: Tensor, return_readout: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """`return_readout` additionally returns the MLP branch BEFORE the
        residual add -- i.e. the output of `mlp.l3`, which is the fine-tune's
        read-out.

        Returning it is what lets `StripedHyena.forward` extract the read-out
        without a forward hook. Registering a hook per forward is untraceable by
        dynamo (`RemovableHandle.next_id += 1` is a `setattr` on a class), and
        cost 2 graph breaks -- the backbone, head and loss compiled as three
        separate graphs. See `docs/OPTIMIZATIONS.md`.

        The full block output is still returned first, so a forward hook on this
        module still sees the post-residual tensor as `output[0]` and the
        `blocks0_10.pt` stream dump is unchanged.
        """
        z = self.projections(self.pre_norm(u))
        z = self.filter(z)
        z_in = self.out_filter_dense(z) + u
        mlp_out = self.mlp(self.post_norm(z_in))
        out = mlp_out + z_in

        return (out, mlp_out) if return_readout else out


class AttentionBlock(nn.Module):
    """vortex `model.AttentionBlock`."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.pre_norm = RMSNorm(config)
        self.post_norm = RMSNorm(config)
        self.inner_mha_cls = MHA(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            rotary_emb_dim=config.hidden_size // config.num_attention_heads,
            # Both direct reads. `rotary_emb_base` defaulted to 1_000_000 here
            # while every 8k Evo2 YAML says 10_000 -- a 100x-wrong RoPE base
            # corrupts only the attention blocks and presents as "attention is
            # slightly off", which is exactly the signature a flash-attn-vs-SDPA
            # investigation would waste its time on. Fail loud instead.
            rotary_emb_base=config.rotary_emb_base,
            qkv_proj_bias=config.qkv_proj_bias,
            out_proj_bias=config.get("mha_out_proj_bias", True),
            layer_idx=layer_idx,
            use_sdpa=config.get("use_sdpa", True),
        )
        self.mlp = ParallelGatedMLP(config, layer_idx)

    def forward(self, u: Tensor, return_readout: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """See `ParallelGatedConvBlock.forward` for what `return_readout` is for."""
        u = self.inner_mha_cls(self.pre_norm(u)) + u
        mlp_out = self.mlp(self.post_norm(u))
        out = mlp_out + u

        return (out, mlp_out) if return_readout else out


def get_block(config, layer_idx: int) -> nn.Module:
    """vortex `model.get_block`."""
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)

    if layer_idx in config.hcl_layer_idxs:
        return ParallelGatedConvBlock(config, layer_idx)

    if layer_idx in config.hcm_layer_idxs:
        return ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcm_filter_groups,
            fir_inner_filter_length=config.hcm_filter_length,
        )

    if layer_idx in config.hcs_layer_idxs:
        return ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcs_filter_groups,
            fir_inner_filter_length=config.hcs_filter_length,
        )

    raise NotImplementedError(f"no block type for layer {layer_idx}")
