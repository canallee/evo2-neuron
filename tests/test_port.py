"""The port itself: config integrity, structure, and the read-out contract.

The `slow` tests load the real Arc checkpoint and are skipped unless `--runslow`
is passed. Point `EVO2_CHECKPOINT` at a local (optionally pruned) file first.
"""

from __future__ import annotations

import os

import pytest
import torch

from evo2_neuron import ByteTokenizer, Evo2ArcConfig, load
from evo2_neuron.loading import convert_wqkv_column_split, stabilize_random_init
from evo2_neuron.model import StripedHyena

HAS_CHECKPOINT = bool(os.environ.get("EVO2_CHECKPOINT"))
slow = pytest.mark.slow


class TestConfig:
    @pytest.mark.parametrize("alias,hidden,layers", [("evo2-1b", 1920, 25), ("evo2-7b", 4096, 32)])
    def test_dimensions(self, alias, hidden, layers):
        config = Evo2ArcConfig.for_alias(alias)

        assert config.hidden_size == hidden
        assert config.num_layers == layers

    def test_fp8_is_off_on_both(self):
        """The port has no TransformerEngine, so there is nothing to emulate. The
        7B config sets this False upstream too, which is what makes 7B gateable
        against a CUDA reference at all."""
        for alias in ("evo2-1b", "evo2-7b"):
            assert Evo2ArcConfig.for_alias(alias).get("use_fp8_input_projections") is False

    def test_block_type_indices_partition_the_stack(self):
        """Every block is exactly one of attn / hcl / hcm / hcs. A gap or an
        overlap builds the wrong architecture and still loads."""
        for alias in ("evo2-1b", "evo2-7b"):
            config = Evo2ArcConfig.for_alias(alias)
            groups = [config.get(f"{k}_layer_idxs", []) for k in ("attn", "hcl", "hcm", "hcs")]
            flat = [i for group in groups for i in group]

            assert len(flat) == len(set(flat)), f"{alias}: a block has two types"
            assert sorted(flat) == list(range(config.num_layers)), f"{alias}: incomplete"

    def test_readout_block_is_reachable(self):
        for alias in ("evo2-1b", "evo2-7b"):
            config = Evo2ArcConfig.for_alias(alias)

            assert config.get("readout_block") < config.num_layers

    def test_unknown_alias_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="known:"):
            Evo2ArcConfig.for_alias("evo2-999b")

    def test_overrides_pass_through_false_but_drop_none(self):
        """Unset CLI flags arrive as None and must not clobber a vendored value;
        `False` is a legitimate experiment."""
        assert Evo2ArcConfig.for_alias("evo2-7b", hcl_kernel=None).get("hcl_kernel") is None
        assert Evo2ArcConfig.for_alias("evo2-7b", interleave=False).get("interleave") is False


class TestTokenizer:
    def test_byte_values(self):
        """A=65 C=67 G=71 T=84 — fixed by the checkpoint's 512-row embedding."""
        assert ByteTokenizer().encode("ACGT") == [65, 67, 71, 84]

    def test_case_is_significant(self):
        """Soft-masked reference genome is a different input, not the same one."""
        assert ByteTokenizer().encode("acgt") != ByteTokenizer().encode("ACGT")

    def test_padding_and_mask(self):
        ids, mask = ByteTokenizer().encode_batch(["ACGT", "AC"], max_length=6)

        assert ids.shape == (2, 6)
        assert ids[1, 2:].tolist() == [ByteTokenizer.pad_id] * 4
        assert mask[1].tolist() == [1, 1, 0, 0, 0, 0]

    def test_truncates_to_max_length(self):
        ids, mask = ByteTokenizer().encode_batch(["ACGT" * 10], max_length=8)

        assert ids.shape == (1, 8)
        assert mask.sum() == 8

    def test_round_trip(self):
        tokenizer = ByteTokenizer()

        assert tokenizer.decode(tokenizer.encode("ACGTTTGCA")) == "ACGTTTGCA"


class TestWqkvRelayout:
    def test_is_not_the_identity(self):
        """Skipping it scrambles q/k/v across heads with no error. If this ever
        becomes a no-op, the loader has stopped doing the thing it exists for."""
        heads, hidden = 4, 32
        weight = torch.arange(3 * hidden * hidden, dtype=torch.float32).reshape(3 * hidden, hidden)
        converted = convert_wqkv_column_split(weight, hidden, heads)

        assert converted.shape == weight.shape
        assert not torch.equal(converted, weight)

    def test_is_a_permutation(self):
        heads, hidden = 4, 32
        weight = torch.randn(3 * hidden, hidden)
        converted = convert_wqkv_column_split(weight, hidden, heads)

        torch.testing.assert_close(converted.sort(dim=0).values, weight.sort(dim=0).values)


class TestRandomInitStability:
    def test_poles_are_forced_inside_the_unit_circle(self):
        """`randn` puts half the modal poles outside it, and `a^s` then diverges
        to nan within three blocks on a 256 bp input."""
        config = Evo2ArcConfig.for_alias("evo2-7b")
        model = StripedHyena(config, max_blocks=3)

        touched = stabilize_random_init(model)

        assert touched > 0

        for name, param in model.named_parameters():
            if name.endswith("log_poles"):
                assert float(param.max()) < 0.0, f"{name} has a non-decaying pole"

    def test_forward_is_finite_after_stabilising(self):
        model = load("evo2-7b", max_blocks=3, random_init=True)

        with torch.no_grad():
            out = model(torch.randint(65, 85, (1, 256)))

        assert torch.isfinite(out).all()


class TestStructure:
    def test_prunes_to_max_blocks(self):
        model = load("evo2-7b", max_blocks=11, random_init=True)

        assert model.num_built_blocks == 11
        assert len(model.blocks) == 11

    def test_readout_shape(self):
        model = load("evo2-7b", max_blocks=3, random_init=True)

        with torch.no_grad():
            out = model(torch.randint(65, 85, (2, 128)))

        assert out.shape == (2, 128, 4096)

    def test_readout_idx_returns_the_l3_output_not_the_block_output(self):
        """`hidden0` is the MLP down-projection taken BEFORE the residual add. A
        port that reads out after the add will not match the saved activations."""
        model = load("evo2-7b", max_blocks=3, random_init=True)
        ids = torch.randint(65, 85, (1, 64))

        captured = {}
        handle = model.blocks[2].mlp.l3.register_forward_hook(
            lambda m, i, o: captured.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))

        try:
            with torch.no_grad():
                returned = model(ids, readout_idx=2)
        finally:
            handle.remove()

        assert torch.equal(returned, captured["h"])


@slow
@pytest.mark.skipif(not HAS_CHECKPOINT, reason="set EVO2_CHECKPOINT")
class TestAgainstTheRealCheckpoint:
    def test_readout_matches_the_shipped_cpu_bundle(self):
        from pathlib import Path

        from evo2_finetune.model import build_model

        root = Path(__file__).resolve().parents[1]
        bundle = root / "artifacts" / "cpu_seq200_fwd"
        reference = torch.load(bundle / "hidden0.pt", weights_only=False)["h"]
        ids = torch.load(bundle / "inputs.pt", weights_only=False)["batches"][0]["input_ids"]

        model = build_model(objective="window-regression", alias="evo2-7b", out_dim=2,
                            dtype=torch.bfloat16, head_kwargs={"window": 200})

        with torch.no_grad():
            hidden = model.backbone(ids).float()

        assert torch.equal(hidden, reference), (
            "the shipped CPU bundle is bit-reproducible; a mismatch here means the "
            "port changed"
        )

    def test_hook_and_return_readout_are_bit_identical(self):
        from evo2_finetune.backbone import Evo2Backbone

        backbone = Evo2Backbone("evo2-7b", dtype=torch.float32)
        ids = torch.randint(65, 85, (1, 128))

        with torch.no_grad():
            backbone.readout_mode = "hook"
            by_hook = backbone(ids).clone()
            backbone.readout_mode = "return"
            by_return = backbone(ids).clone()

        assert torch.equal(by_hook, by_return)
