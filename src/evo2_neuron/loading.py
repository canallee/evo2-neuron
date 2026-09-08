"""
Load Arc's `evo2_*_base.pt` checkpoint into the ported `StripedHyena`.

The port's module names were chosen to match vortex's exactly, so no key renaming
is required. Four things do need handling.

1. **The `Wqkv` relayout.** Arc stores QKV head-major; the model unpacks it
   three-major. vortex converts at load time and so must we -- see
   `convert_wqkv_column_split` below. This is NOT optional and is the single
   easiest thing to get wrong.
2. **`_extra_state` entries.** TransformerEngine writes an `_io.BytesIO` blob per
   FP8 layer (26 of them on evo2-1b). They are not tensors and the port has no
   TE, so they are dropped -- by an `isinstance(value, torch.Tensor)` filter, so
   the count does not matter and neither does the backbone. Note the blobs carry
   only `scale_fwd`/`amax_history_fwd`; the stored weights are unquantised, so
   dropping them loses nothing. (Some TE versions store `_extra_state` as a uint8
   *tensor* instead. That would survive the filter and land in `unexpected` ->
   the RuntimeError below. Loud, not silent.)
3. **`weights_only=False`.** Those same blobs make `torch.load`'s default
   `weights_only=True` refuse the file. We allowlist `io.BytesIO` and load
   explicitly rather than turning the safety check off wholesale.
4. **Truncated stacks.** With `max_blocks` set, keys for blocks that were never
   built are dropped instead of tripping `strict=True`.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Keys present in Arc's checkpoint that the port has no module for.
#   unembed.weight -- tied to the embedding (`tie_embeddings: True`), so vortex
#   installs a Lambda rather than a parameter. The stored copy is redundant.
_IGNORED_KEYS = {"unembed.weight"}


def default_checkpoint_path(model_name: str = "evo2_1b_base") -> Path:
    """Resolve Arc's single-file checkpoint from the local HF cache.

    Downloads it if absent. `model_name` (e.g. `evo2_7b_base`) is an *Evo2 model
    name*, not an HF repo id: the matching repo is `arcinstitute/<name>` and it
    holds a single `<name>.pt`.

    `EVO2_CHECKPOINT` short-circuits the download. Point it at a local file --
    typically a pruned checkpoint from `scripts/prune_checkpoint.py`, which is
    how a 64-rank job avoids 64 concurrent reads of the full 13 GB file.
    """
    override = os.environ.get("EVO2_CHECKPOINT")

    if override:
        path = Path(override).expanduser()

        if not path.is_file():
            raise FileNotFoundError(f"EVO2_CHECKPOINT={override} is not a file")

        logger.info("checkpoint: EVO2_CHECKPOINT -> %s", path)

        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(f"arcinstitute/{model_name}", f"{model_name}.pt"))


def convert_wqkv_column_split(
    weight: torch.Tensor, hidden_size: int, num_heads: int
) -> torch.Tensor:
    """savanna -> vortex QKV relayout. Arc's checkpoint needs it; do not skip it.

    Arc stores `Wqkv` head-major -- `(heads, 3, head_dim)` -- but `MHA.forward`
    unpacks three-major via `rearrange(qkv, "... (three h d) -> ... three h d")`.
    vortex fixes this at load time, gated on `column_split: True`
    (`vortex/model/model.py:880-904`), which is set for every Evo2 config. Ported
    here verbatim.

    Without it the four attention blocks silently compute with q/k/v scrambled
    across heads. Measured: this changes `blocks.10.mlp.l3` by rel_err ~1.37.
    """
    size_att_head = hidden_size // num_heads

    weight = weight.permute(1, 0)
    weight = weight.reshape(hidden_size, num_heads, 3, size_att_head)
    wq, wk, wv = weight.unbind(dim=-2)
    weight = torch.cat(
        [wq.reshape(hidden_size, -1), wk.reshape(hidden_size, -1), wv.reshape(hidden_size, -1)],
        dim=-1,
    )

    return weight.permute(1, 0).contiguous()


def load_arc_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read `evo2_*_base.pt` and return only its real tensors."""
    torch.serialization.add_safe_globals([io.BytesIO])
    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    for key in ("model_state_dict", "state_dict", "module"):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]

    state_dict = {
        key: value.detach()
        for key, value in raw.items()
        if isinstance(value, torch.Tensor) and key not in _IGNORED_KEYS
    }

    logger.info(
        "loaded %d tensors from %s (dropped %d non-tensor/ignored keys)",
        len(state_dict), path, len(raw) - len(state_dict),
    )

    return state_dict


def load_into(
    module: nn.Module,
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    max_blocks: int | None = None,
    column_split: bool = True,
) -> None:
    """Load Arc weights into a ported `StripedHyena`, strictly.

    `strict=True` on the surviving keys is the point: a silent `missing_keys` is
    how a port ends up running half-random weights and blaming the numerics.

    `hidden_size` / `num_heads` are read off `module.config` rather than passed
    in. They used to be parameters defaulting to evo2-1b's 1920 / 15, which made
    it possible to relayout a 7B `Wqkv` with 1B geometry. A mismatched pair that
    still divides evenly scrambles q/k/v across heads with no error at all (see
    `convert_wqkv_column_split`, measured rel_err ~1.37). Sourcing both from the
    module being loaded into makes that class of bug unrepresentable.
    """
    hidden_size = module.config.hidden_size
    num_heads = module.config.num_attention_heads

    state_dict = load_arc_state_dict(path)

    if max_blocks is not None:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("blocks.") or int(key.split(".")[1]) < max_blocks
        }

    # fp32 by default. Arc stores bf16 for everything except the hcl poles and
    # residues, which are already fp32; on Neuron, bf16 collapses some Evo2
    # activations to zero, so fp32 is the port's source of truth.
    state_dict = {key: value.to(dtype) for key, value in state_dict.items()}

    if column_split:
        for key in [k for k in state_dict if k.endswith("inner_mha_cls.Wqkv.weight")]:
            state_dict[key] = convert_wqkv_column_split(state_dict[key], hidden_size, num_heads)

    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    # `rotary_emb.inv_freq` is recomputed in fp32 at forward time (matching
    # vortex, which ignores the checkpoint's bf16 copy), so it is a legitimate
    # extra. Everything else must line up.
    real_missing = [key for key in missing if not key.endswith("rotary_emb.inv_freq")]

    if real_missing or unexpected:
        raise RuntimeError(
            f"Arc checkpoint did not match the ported module.\n"
            f"  missing ({len(real_missing)}): {real_missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}"
        )

    logger.info("loaded Arc weights into %d blocks (dtype=%s)", len(module.blocks), dtype)


def stabilize_random_init(module: nn.Module) -> int:
    """Force every modal pole inside the unit circle. **Random-init path only.**

    `HyenaCascade.__init__` draws `log_poles` from `randn`, so about half the
    poles land at `a = exp(log_poles) > 1`. The modal filter is `h[s] = sum_j r_j
    a_j^s`, so any such pole diverges over the sequence: on a 256 bp input the
    third block already produces `nan`. Upstream never notices, because vortex
    constructs the module and immediately overwrites every parameter from a
    checkpoint — the init values are dead on arrival.

    Arc's trained poles are all negative (max -0.0012 on evo2-1b), i.e. strictly
    decaying. This reflects `log_poles` to `-|log_poles|` and scales it so the
    slowest mode decays by a useful amount across a typical context, which is the
    regime the trained model actually occupies.

    Returns the number of tensors touched. Do **not** call this on a checkpoint
    load: it would corrupt trained filters. `load_into` never does.
    """
    touched = 0

    with torch.no_grad():
        for name, param in module.named_parameters():
            if not name.endswith("log_poles"):
                continue

            param.copy_(-param.abs().clamp_min(1e-3) * 0.05)
            touched += 1

    logger.info("stabilize_random_init: constrained %d log_poles tensor(s) to |a| < 1", touched)

    return touched
