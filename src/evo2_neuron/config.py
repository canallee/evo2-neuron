"""
Evo 2 architecture config, read from Arc's own YAML.

Arc's HuggingFace repos (`arcinstitute/evo2_*`) ship a four-line `config.json`
that carries nothing but a display name -- the real architecture description is
the YAML in the `evo2` repo (`evo2/configs/evo2-{1b,7b}-8k.yml`). Those YAML
values are vendored below so the port needs no network and no `evo2` install.

Deliberately NOT sourced from `Taykhoom/Evo2-1B-8K`'s `config.json`: that repo's
modeling code is known-defective (see `VENDORED.md`), and mixing its config with
Arc's weights is exactly the kind of silent mismatch this file exists to prevent.
The value sets happen to agree, which is why the defect is in the code, not here.

Vendored by hand rather than loaded through `from_yaml`, for the same reason.
The 7B YAML exists on this box in two mutually inconsistent copies (evo2 0.6.0's
and vortex 0.0.2's) that disagree on `use_fp8_input_projections` AND on whether
`interleave` is present at all, and the copy that actually minted the reference
-- evo2 0.1.0's -- is not on this box. `from_yaml` would pick a side silently,
from an unversioned checkout outside the repo. Each dict below is instead
eyeball-diffable against Arc's YAML, with every divergence annotated. Do not
factor the shared keys into a base dict; that would destroy the property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Verbatim from ArcInstitute/evo2 `evo2/configs/evo2-1b-8k.yml` @ 53f1959.
EVO2_1B_8K: dict[str, Any] = {
    "model_name": "shc-evo2-1b-8k-2T-v2",
    "huggingface_id": "evo2_1b_base",
    "readout_block": 10,
    "vocab_size": 512,
    "hidden_size": 1920,
    "num_filters": 1920,
    "attn_layer_idxs": [3, 10, 17, 24],
    "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23],
    "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22],
    "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21],
    "hcm_filter_length": 128,
    "hcl_filter_groups": 1920,
    "hcm_filter_groups": 128,
    "hcs_filter_groups": 128,
    "hcs_filter_length": 7,
    "num_layers": 25,
    "short_filter_length": 3,
    "num_attention_heads": 15,
    "short_filter_bias": False,
    "eps": 1e-6,
    "state_size": 16,
    "rotary_emb_base": 10000,
    "make_vocab_size_divisible_by": 8,
    "inner_size_multiple_of": 16,
    "inner_mlp_size": 5120,
    "proj_groups": 1,
    "hyena_filter_groups": 1,
    "column_split_hyena": False,
    "column_split": True,
    "interleave": True,
    "evo2_style_activations": True,
    "model_parallel_size": 1,
    "tie_embeddings": True,
    "mha_out_proj_bias": True,
    "hyena_out_proj_bias": True,
    "hyena_flip_x1x2": False,
    "qkv_proj_bias": False,
    "max_seqlen": 8192,
    "final_norm": True,
    "mlp_activation": "gelu",
    "prefill_style": "fft",
    # --- forced off by the port (see VENDORED.md) -----------------------------
    # FP8 needs TransformerEngine; flash-attn/flashfft/flash-rmsnorm need CUDA.
    #
    # For evo2-1b these are REAL divergences from the reference, not agreements.
    # Arc's YAML asks for `use_fp8_input_projections: True`, and disabling it
    # costs 47 accuracy points on 1B (79.568% -> 32.546%, cosine 0.131 at the
    # read-out). That is why evo2-1b cannot be gated against its CUDA reference
    # and why evo2-7b, below, is the port's actual target.
    "use_fp8_input_projections": False,
    "use_flash_attn": False,
    "use_flash_rmsnorm": False,
    "use_flash_depthwise": False,
    "use_flashfft": False,
}

# Verbatim from ArcInstitute/evo2 `evo2/configs/evo2-7b-8k.yml` @ **evo2 0.1.0**.
#
# The version matters more than a commit sha here. evo2 0.6.0 (current upstream
# HEAD, and what `requirements-evo2.txt` pins) flipped this model's
# `use_fp8_input_projections` False -> True. The CUDA reference in
# `data/_align/evo2_7b_ref/` was minted under 0.1.0 (see its `manifest.json`
# `provenance`), i.e. FP8 OFF -- which is the entire reason 7B, not 1B, is the
# port's target: there is no FP8 to emulate. Re-minting from 0.6.0 would produce
# an FP8 reference and this port would fail exactly the way 1B did, with no
# obvious cause. Pin `evo2==0.1.0` for any re-mint; do NOT rebuild from
# `requirements-evo2.txt`, which is stale on torch/TE versions too.
#
# Nothing here establishes that 7B *would* tolerate FP8 being on. It does not
# matter, because the reference does not use it. Do not claim more than that.
EVO2_7B_8K: dict[str, Any] = {
    "model_name": "shc-evo2-7b-8k-2T-v2",
    # Arc's *Evo2 model name*, not an HF repo id: the repo is
    # `arcinstitute/evo2_7b_base` and it holds a single `evo2_7b_base.pt`.
    "huggingface_id": "evo2_7b_base",
    # Default mid-stack read-out. Directly measured, not extrapolated: on 7B the
    # Delta-L2 between real and random-ACGT input first crosses 0.10 at block 10,
    # and the raw-L2 growth plateau starts at block 10-11. Block 13 is the
    # conservative alternative (the *sustained* crossing); 10 prunes more.
    "readout_block": 10,
    "vocab_size": 512,
    "hidden_size": 4096,
    "num_filters": 4096,
    "attn_layer_idxs": [3, 10, 17, 24, 31],   # FIVE attention blocks, not 1B's four
    "hcl_layer_idxs": [2, 6, 9, 13, 16, 20, 23, 27, 30],
    "hcm_layer_idxs": [1, 5, 8, 12, 15, 19, 22, 26, 29],
    "hcs_layer_idxs": [0, 4, 7, 11, 14, 18, 21, 25, 28],
    "hcm_filter_length": 128,
    # Never read: `get_block` passes no `hyena_filter_groups` for hcl, so
    # `HyenaCascade` falls back to `config.hidden_size` (hyena.py). That fallback
    # happens to equal this value for both 1B (1920) and 7B (4096). Decorative,
    # but keep it -- it documents Arc's intent and would catch a future config
    # where the two genuinely differ.
    "hcl_filter_groups": 4096,
    # 256 groups against hidden 4096 -> `repeat_interleave` ratio 16 (1B: 15).
    # Exact divisor; hyena.py asserts it rather than truncating silently.
    "hcm_filter_groups": 256,
    "hcs_filter_groups": 256,
    "hcs_filter_length": 7,
    "num_layers": 32,
    "short_filter_length": 3,
    "num_attention_heads": 32,
    "short_filter_bias": False,
    "eps": 1e-6,
    "state_size": 16,
    "rotary_emb_base": 10000,
    "make_vocab_size_divisible_by": 8,
    "inner_size_multiple_of": 16,
    # Must be supplied. layers.py derives 2*4096*4/3 rounded up to 10928, which
    # is NOT 11008 -- omitting this key silently builds an MLP that mismatches
    # the checkpoint.
    "inner_mlp_size": 11008,
    "proj_groups": 1,
    "hyena_filter_groups": 1,
    "column_split_hyena": False,
    "column_split": True,
    # `interleave` is PRESENT in evo2 0.6.0's evo2-7b-8k.yml and ABSENT from
    # vortex 0.0.2's copy of the same file (the 1B ymls are byte-identical
    # between the two sources and both carry it). vortex's config object is a
    # `dotdict` with `__getattr__ = dict.get` (vortex/model/utils.py:207-212),
    # so that absence reads as None -> falsy -> interleave OFF, silently. It is a
    # real channel permutation applied in EVERY hyena block including block 0, so
    # getting it wrong is not subtle -- it just has no error path.
    #
    # The evo2 0.1.0 yml that minted data/_align/evo2_7b_ref is not on this box,
    # so the value was settled empirically against blocks0_10.pt["block_out.0"]
    # -- a 4.07x cancellation point (||u|| 8.4236, ||block_out.0|| 2.186057,
    # cos(u, block_out.0) = -0.092) and therefore the most sensitive probe in the
    # bundle. Measured on CPU, bf16, block 0 only, on the reference's own inputs
    # (ref ||block_out.0|| = 2.186057):
    #
    #     interleave True  -> ||block_out.0|| =   2.186104, rel_err 3.85e-03, cos 0.999993
    #     interleave False -> ||block_out.0|| = 102.402553, rel_err 4.68e+01, cos 0.009589
    #
    # Four orders of magnitude, and the True arm's 3.85e-03 is just the bf16
    # floor (1.24e-03, read off the undiluted `mlp_l3.0`) amplified by that same
    # 4.07x cancellation. This is not a close call; the losing number is recorded
    # so it never has to be re-run.
    #
    # Note the port reads `self.config.interleave` as a DIRECT attribute, so
    # omitting this key raises AttributeError instead of defaulting. That is the
    # one place the port is strictly safer than vortex; keep it that way.
    "interleave": True,
    "evo2_style_activations": True,
    "model_parallel_size": 1,
    "tie_embeddings": True,
    "mha_out_proj_bias": True,
    "hyena_out_proj_bias": True,
    "hyena_flip_x1x2": False,
    "qkv_proj_bias": False,
    # Never read -- seqlen is taken from the input at every site. A buffer
    # ceiling in Arc's code, not a run length.
    "max_seqlen": 32768,
    "final_norm": True,
    "mlp_activation": "gelu",
    "prefill_style": "fft",
    # --- the port's divergences, for 7B -------------------------------------
    # `use_fp8_input_projections` is already False in Arc's 0.1.0 YAML, so this
    # line is a NO-OP that matches the reference exactly. That is the whole
    # point of targeting 7B.
    #
    # `use_flash_attn` is NOT a no-op. Arc's YAML says True and the reference
    # genuinely ran vortex's `FlashSelfAttention` -- `import flash_attn_2_cuda`
    # succeeds in the H200 env once torch is loaded. The port substitutes
    # `F.scaled_dot_product_attention`. FlashAttention-2 is exact in exact
    # arithmetic, so the expected cost is one bf16 rounding (~1e-3 relative) on
    # the attention output, at blocks 3 and 10 only. Do not read this line as a
    # config agreement.
    "use_fp8_input_projections": False,
    "use_flash_attn": False,
    "use_flash_rmsnorm": False,
    "use_flash_depthwise": False,
    "use_flashfft": False,
}

_REGISTRY = {
    "evo2-1b": EVO2_1B_8K,
    "evo2_1b_base": EVO2_1B_8K,
    "evo2-7b": EVO2_7B_8K,
    "evo2_7b_base": EVO2_7B_8K,
}


@dataclass
class Evo2ArcConfig:
    """Attribute + `.get()` access, matching how vortex's code reads its config."""

    values: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.values.items():
            setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @classmethod
    def for_alias(cls, alias: str, **overrides: Any) -> Evo2ArcConfig:
        """`overrides` sets config keys that are not part of Arc's YAML, and is
        also the A/B knob for one-off experiments against a vendored value.

        `hcl_chunk` selects the linear-time modal form of the hcl convolution
        (see `hyena.modal_causal_conv_chunked`). It is an override rather than a
        default because it shifts fp32 numerics by ~1e-7, so it must be re-gated
        before it can become the default.

        `hcl_kernel` picks how the hcl convolution is evaluated:
        `"direct"` (default, the form the CUDA artifacts were gated against),
        `"chunked"` (equivalently, set `hcl_chunk`), or `"nki-scan"`
        (nki-library `linear_scan`). At the production 8192 context only the
        latter two run at all -- see `docs/OPTIMIZATIONS.md`.

        Note `None` overrides are dropped (that is how unset CLI flags arrive),
        but `False` is passed through -- so `for_alias("evo2-7b",
        interleave=False)` is a valid experiment. Run experiments this way rather
        than by registering a second alias, which would leave a known-wrong
        config permanently reachable by name.
        """
        try:
            values = dict(_REGISTRY[alias])
        except KeyError:
            raise KeyError(
                f"no vendored Arc config for {alias!r}; known: {sorted(_REGISTRY)}. "
                "Add the values from the matching evo2/configs/*.yml."
            ) from None

        values.update({k: v for k, v in overrides.items() if v is not None})

        return cls(values)

    @classmethod
    def from_yaml(cls, path: str) -> Evo2ArcConfig:
        """Load one of Arc's own `evo2/configs/*.yml` directly."""
        import yaml

        with open(path) as handle:
            values = yaml.safe_load(handle)

        # The port cannot honour these regardless of what the YAML asks for.
        values.update(
            use_fp8_input_projections=False,
            use_flash_attn=False,
            use_flash_rmsnorm=False,
            use_flash_depthwise=False,
            use_flashfft=False,
        )

        return cls(values)
