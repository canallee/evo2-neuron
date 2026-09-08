"""Read-out heads: shapes, masking, and the conv rewrite's equivalence."""

from __future__ import annotations

import pytest
import torch

from evo2_finetune.heads import (
    HEAD_REGISTRY,
    ConvReadoutHead,
    MLPReadoutHead,
    PerPositionHead,
    WindowTrackHead,
    build_head,
)

DIM = 32


def window_mask(batch: int, length: int, width: int, start: int) -> torch.Tensor:
    mask = torch.zeros(batch, length, dtype=torch.long)
    mask[:, start:start + width] = 1

    return mask


class TestMLPReadoutHead:
    def test_shape(self):
        head = MLPReadoutHead(DIM, 5, hidden_dim=16)
        out = head(torch.randn(3, 64, DIM), torch.ones(3, 64, dtype=torch.long))

        assert out.shape == (3, 5)

    def test_masked_positions_are_excluded(self):
        """Padding must not drag the pooled vector toward zero."""
        torch.manual_seed(0)
        head = MLPReadoutHead(DIM, 2, hidden_dim=16)
        hidden = torch.randn(1, 64, DIM)
        mask = torch.zeros(1, 64, dtype=torch.long)
        mask[:, :10] = 1

        padded = hidden.clone()
        padded[:, 10:] = 999.0                        # garbage in the masked region

        torch.testing.assert_close(head(hidden, mask), head(padded, mask))

    def test_all_zero_mask_does_not_divide_by_zero(self):
        head = MLPReadoutHead(DIM, 2, hidden_dim=16)
        out = head(torch.randn(1, 8, DIM), torch.zeros(1, 8, dtype=torch.long))

        assert torch.isfinite(out).all()


class TestConvReadoutHead:
    def test_shape(self):
        head = ConvReadoutHead(DIM, 3, width=16, hidden_dim=8)
        out = head(torch.randn(2, 64, DIM), window_mask(2, 64, 16, 20))

        assert out.shape == (2, 3)

    def test_matmul_form_matches_the_real_conv(self):
        """The rewrite exists because `convolution_backward` does not compile on
        Neuron. It has to be the same function."""
        torch.manual_seed(0)
        head = ConvReadoutHead(DIM, 3, width=16, hidden_dim=8)
        hidden = torch.randn(2, 64, DIM)
        mask = window_mask(2, 64, 16, 20)

        torch.testing.assert_close(
            head(hidden, mask), head.forward_reference(hidden, mask),
            rtol=1e-5, atol=1e-5,
        )

    def test_gradients_match_the_real_conv(self):
        torch.manual_seed(0)
        head = ConvReadoutHead(DIM, 3, width=16, hidden_dim=8)
        hidden = torch.randn(2, 64, DIM, requires_grad=True)
        reference = hidden.detach().clone().requires_grad_(True)
        mask = window_mask(2, 64, 16, 20)

        head(hidden, mask).sum().backward()
        conv_grad = head.conv.weight.grad.clone()
        head.zero_grad()

        head.forward_reference(reference, mask).sum().backward()

        torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(conv_grad, head.conv.weight.grad, rtol=1e-4, atol=1e-5)

    def test_reads_the_region_the_mask_marks(self):
        torch.manual_seed(0)
        head = ConvReadoutHead(DIM, 2, width=8, hidden_dim=4)
        hidden = torch.randn(1, 64, DIM)

        shifted = hidden.clone()
        shifted[:, 40:48] = hidden[:, 8:16]

        torch.testing.assert_close(
            head(hidden, window_mask(1, 64, 8, 8)),
            head(shifted, window_mask(1, 64, 8, 40)),
        )


class TestWindowTrackHead:
    def test_shape(self):
        head = WindowTrackHead(DIM, 2, window=16, hidden_dim=8)
        out = head(torch.randn(3, 128, DIM), torch.ones(3, 128, dtype=torch.long))

        assert out.shape == (3, 8, 2)

    def test_rejects_indivisible_length(self):
        """A partial trailing bin is a silently downweighted label."""
        head = WindowTrackHead(DIM, 1, window=16)

        with pytest.raises(ValueError, match="divisible"):
            head(torch.randn(1, 100, DIM), torch.ones(1, 100, dtype=torch.long))

    def test_windows_are_independent(self):
        torch.manual_seed(0)
        head = WindowTrackHead(DIM, 1, window=16, hidden_dim=8)
        hidden = torch.randn(1, 64, DIM)
        mask = torch.ones(1, 64, dtype=torch.long)

        baseline = head(hidden, mask)
        perturbed = hidden.clone()
        perturbed[:, 32:48] += 5.0

        after = head(perturbed, mask)

        torch.testing.assert_close(after[:, :2], baseline[:, :2])
        torch.testing.assert_close(after[:, 3:], baseline[:, 3:])
        assert not torch.allclose(after[:, 2], baseline[:, 2])


class TestPerPositionHead:
    def test_shape(self):
        head = PerPositionHead(DIM, 4, hidden_dim=8)
        out = head(torch.randn(2, 50, DIM), torch.ones(2, 50, dtype=torch.long))

        assert out.shape == (2, 50, 4)

    def test_position_independent(self):
        """No pooling: perturbing one position must move only that position."""
        torch.manual_seed(0)
        head = PerPositionHead(DIM, 3, hidden_dim=8)
        hidden = torch.randn(1, 20, DIM)

        baseline = head(hidden, None)
        perturbed = hidden.clone()
        perturbed[:, 7] += 3.0

        after = head(perturbed, None)

        torch.testing.assert_close(after[:, :7], baseline[:, :7])
        torch.testing.assert_close(after[:, 8:], baseline[:, 8:])


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(HEAD_REGISTRY))
    def test_every_registered_head_builds(self, name):
        kwargs = {"width": 8} if name == "conv" else {"window": 8} if name == "window" else {}

        assert isinstance(build_head(name, DIM, 2, **kwargs), torch.nn.Module)

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(KeyError, match="known:"):
            build_head("nope", DIM, 2)
