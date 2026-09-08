"""LoRA: shapes, init, scoping, save/load, merge/unmerge."""

from __future__ import annotations

import math
import re

import pytest
import torch
from torch import nn

from evo2_finetune.lora import (
    DEFAULT_TARGET_REGEX,
    LoRALinear,
    apply_lora,
    freeze_base,
    get_lora_state_dict,
    load_lora_state_dict,
    merge_lora_weights,
    scope_regex_to_blocks,
    unmerge_lora_weights,
)


class TinyMLP(nn.Module):
    def __init__(self, dim=16, inner=32):
        super().__init__()
        self.l1 = nn.Linear(dim, inner, bias=False)
        self.l2 = nn.Linear(dim, inner, bias=False)
        self.l3 = nn.Linear(inner, dim, bias=False)


class TinyBlock(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.mlp = TinyMLP(dim)
        self.norm = nn.LayerNorm(dim)


class TinyStack(nn.Module):
    def __init__(self, blocks=4, dim=16):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(dim) for _ in range(blocks))


@pytest.fixture
def stack():
    torch.manual_seed(0)

    return TinyStack()


class TestLoRALinear:
    def test_shapes_and_scaling(self):
        adapter = LoRALinear(nn.Linear(8, 12), r=4, alpha=8.0)

        assert adapter.lora_A.shape == (4, 8)
        assert adapter.lora_B.shape == (12, 4)
        assert adapter.scaling == 2.0

    def test_b_is_zero_and_a_is_kaiming(self):
        adapter = LoRALinear(nn.Linear(64, 64), r=8, alpha=8.0)

        assert float(adapter.lora_B.abs().max()) == 0.0
        # Kaiming-uniform with a = sqrt(5) bounds A by sqrt(6 / ((1+5) * fan_in)).
        bound = math.sqrt(6.0 / (6.0 * 64))
        assert float(adapter.lora_A.abs().max()) <= bound + 1e-6
        assert float(adapter.lora_A.abs().max()) > 0.0

    def test_identity_at_init(self):
        base = nn.Linear(16, 16)
        adapter = LoRALinear(base, r=4, alpha=4.0)
        x = torch.randn(3, 16)

        torch.testing.assert_close(adapter(x), base(x))

    def test_marker_present(self):
        adapter = LoRALinear(nn.Linear(8, 8), r=2, alpha=2.0)

        assert adapter.lora_A._is_lora_param
        assert adapter.lora_B._is_lora_param

    def test_delta_is_the_documented_formula(self):
        adapter = LoRALinear(nn.Linear(8, 8, bias=False), r=4, alpha=8.0)

        with torch.no_grad():
            adapter.lora_B.copy_(torch.randn_like(adapter.lora_B))

        x = torch.randn(2, 8)
        expected = adapter.base(x) + (x @ adapter.lora_A.T @ adapter.lora_B.T) * 2.0

        torch.testing.assert_close(adapter(x), expected)

    @pytest.mark.parametrize("r,alpha,dropout", [(0, 8, 0.0), (-1, 8, 0.0), (4, 0, 0.0),
                                                 (4, -1, 0.0), (4, 8, 1.0), (4, 8, -0.1)])
    def test_rejects_invalid_hyperparameters(self, stack, r, alpha, dropout):
        with pytest.raises(ValueError):
            apply_lora(stack, r=r, alpha=alpha, dropout=dropout,
                       target_regex=DEFAULT_TARGET_REGEX)

    def test_rejects_bool_rank(self, stack):
        with pytest.raises(ValueError):
            apply_lora(stack, r=True, alpha=8.0, dropout=0.0,
                       target_regex=DEFAULT_TARGET_REGEX)


class TestApplyLoRA:
    def test_wraps_every_mlp_linear(self, stack):
        wrapped = apply_lora(stack, r=4, alpha=4.0, dropout=0.0,
                             target_regex=DEFAULT_TARGET_REGEX)

        assert len(wrapped) == 12                       # 4 blocks x l1/l2/l3
        assert isinstance(stack.blocks[0].mlp.l1, LoRALinear)
        assert not isinstance(stack.blocks[0].norm, LoRALinear)

    def test_skips_non_linear_matches(self, stack, caplog):
        wrapped = apply_lora(stack, r=4, alpha=4.0, dropout=0.0,
                             target_regex=r".*blocks\.\d+\.norm$")

        assert wrapped == []
        assert "not nn.Linear" in caplog.text

    def test_output_unchanged_at_init(self, stack):
        x = torch.randn(2, 16)
        before = stack.blocks[0].mlp.l3(stack.blocks[0].mlp.l1(x))
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        after = stack.blocks[0].mlp.l3(stack.blocks[0].mlp.l1(x))

        torch.testing.assert_close(before, after)


class TestScoping:
    def test_narrows_the_block_alternation(self):
        scoped = scope_regex_to_blocks(DEFAULT_TARGET_REGEX, 2)

        assert scoped == r".*blocks\.(0|1|2)\.mlp\.(l1|l2|l3)$"
        assert re.fullmatch(scoped, "blocks.2.mlp.l1")
        assert not re.fullmatch(scoped, "blocks.3.mlp.l1")

    def test_scoped_apply_leaves_deeper_blocks_alone(self, stack):
        wrapped = apply_lora(stack, r=4, alpha=4.0, dropout=0.0,
                             target_regex=scope_regex_to_blocks(DEFAULT_TARGET_REGEX, 1))

        assert len(wrapped) == 6
        assert isinstance(stack.blocks[1].mlp.l1, LoRALinear)
        assert not isinstance(stack.blocks[2].mlp.l1, LoRALinear)


class TestFreeze:
    def test_only_lora_stays_trainable(self, stack):
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        tensors, params = freeze_base(stack)

        assert tensors == 24                            # 12 modules x A/B
        assert params == sum(
            p.numel() for p in stack.parameters() if getattr(p, "_is_lora_param", False))
        assert all(not p.requires_grad for n, p in stack.named_parameters()
                   if not n.endswith((".lora_A", ".lora_B")))

    def test_gradients_reach_every_adapter(self, stack):
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        freeze_base(stack)

        x = torch.randn(2, 16)
        out = x

        for block in stack.blocks:
            out = block.mlp.l3(block.mlp.l1(out) * block.mlp.l2(out))

        out.sum().backward()

        # `lora_B` is zero at init, so `lora_A.grad` is legitimately zero here.
        # `lora_B.grad` is not, and an ABSENT gradient must not be mistaken for a
        # small one — check for None explicitly.
        missing = [n for n, p in stack.named_parameters()
                   if p.requires_grad and p.grad is None]

        assert missing == []
        assert float(stack.blocks[0].mlp.l1.lora_B.grad.abs().max()) > 0.0


class TestStateDict:
    def test_round_trip(self, stack):
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)

        with torch.no_grad():
            for module in stack.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.copy_(torch.randn_like(module.lora_B))

        saved = get_lora_state_dict(stack)

        assert len(saved) == 24
        assert all(k.endswith((".lora_A", ".lora_B")) for k in saved)

        torch.manual_seed(1)
        fresh = TinyStack()
        apply_lora(fresh, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        load_lora_state_dict(fresh, saved)

        for key, value in saved.items():
            torch.testing.assert_close(fresh.state_dict()[key], value)

    def test_missing_lora_keys_are_warned(self, stack, caplog):
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        partial = dict(list(get_lora_state_dict(stack).items())[:4])
        load_lora_state_dict(stack, partial)

        assert "LoRA keys missing" in caplog.text

    def test_unexpected_keys_are_warned(self, stack, caplog):
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        state = get_lora_state_dict(stack)
        state["blocks.9.mlp.l1.lora_A"] = torch.zeros(4, 16)
        load_lora_state_dict(stack, state)

        assert "unexpected keys" in caplog.text


class TestMerge:
    def _randomise(self, stack):
        with torch.no_grad():
            for module in stack.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.copy_(torch.randn_like(module.lora_B) * 0.1)

    def test_merge_folds_the_delta(self, stack):
        apply_lora(stack, r=4, alpha=8.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        self._randomise(stack)

        adapter = stack.blocks[0].mlp.l1
        before = adapter.base.weight.detach().clone()
        expected = before + (adapter.lora_B @ adapter.lora_A) * adapter.scaling

        merge_lora_weights(stack)

        torch.testing.assert_close(adapter.base.weight, expected)

    def test_unmerge_is_exact_in_bf16(self, stack):
        """Restores a saved copy rather than subtracting the delta back off —
        subtraction in bf16 does not round-trip."""
        stack = stack.to(torch.bfloat16)
        apply_lora(stack, r=4, alpha=8.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        self._randomise(stack)

        original = stack.blocks[0].mlp.l1.base.weight.detach().clone()
        merge_lora_weights(stack)
        unmerge_lora_weights(stack)

        assert torch.equal(stack.blocks[0].mlp.l1.base.weight, original)

    def test_merge_is_idempotent(self, stack):
        apply_lora(stack, r=4, alpha=8.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        self._randomise(stack)

        merge_lora_weights(stack)
        once = stack.blocks[0].mlp.l1.base.weight.detach().clone()
        merge_lora_weights(stack)

        torch.testing.assert_close(stack.blocks[0].mlp.l1.base.weight, once)

    def test_unmerge_drops_the_buffer(self, stack):
        apply_lora(stack, r=4, alpha=8.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        merge_lora_weights(stack)

        assert "_original_weight" in dict(stack.blocks[0].mlp.l1.named_buffers())

        unmerge_lora_weights(stack)

        assert "_original_weight" not in dict(stack.blocks[0].mlp.l1.named_buffers())


class TestFreezeChangesNumericsNotFunction:
    """`requires_grad` changes the bf16 CPU GEMM path, and therefore the answer.

    Not a bug — a reduction-order difference. It is here because it is a trap:
    on the real backbone it moves the read-out by 1.24e-02, as large as the
    entire CUDA-to-CPU disagreement, and it looks exactly like a porting
    regression. Every artifact bundle is produced from a frozen model.
    """

    def test_frozen_and_unfrozen_agree_in_fp32(self, stack):
        x = torch.randn(4, 16)
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)

        with torch.no_grad():
            before = stack.blocks[0].mlp.l1(x)

        freeze_base(stack)

        with torch.no_grad():
            after = stack.blocks[0].mlp.l1(x)

        torch.testing.assert_close(before, after)

    def test_adapters_stay_trainable_after_freeze(self, stack):
        apply_lora(stack, r=4, alpha=4.0, dropout=0.0, target_regex=DEFAULT_TARGET_REGEX)
        freeze_base(stack)

        assert stack.blocks[0].mlp.l1.lora_A.requires_grad
        assert stack.blocks[0].mlp.l1.lora_B.requires_grad
        assert not stack.blocks[0].mlp.l1.base.weight.requires_grad
