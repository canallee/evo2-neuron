"""Every device rewrite, validated against the naive form it replaces.

These are the tests that matter most in this repo: the rewrites exist because the
naive op falls back to the host or fails to compile, and a rewrite that is
*almost* the same function produces a model that runs, converges, and is wrong.
"""

from __future__ import annotations

import pytest
import torch

from evo2_finetune.ops import (
    first_set_index,
    gather_window,
    logaddexp,
    naive_first_set_index,
    naive_logaddexp,
    windowed_conv_as_matmul,
)


def contiguous_mask(batch: int, length: int, width: int, starts) -> torch.Tensor:
    mask = torch.zeros(batch, length, dtype=torch.long)

    for row, start in enumerate(starts):
        mask[row, start:start + width] = 1

    return mask


class TestFirstSetIndex:
    @pytest.mark.parametrize("starts", [[0], [7], [0, 3, 199], [199]])
    def test_matches_argmax(self, starts):
        mask = contiguous_mask(len(starts), 200, 1, starts)

        assert torch.equal(first_set_index(mask), naive_first_set_index(mask))

    def test_matches_argmax_wide_windows(self):
        starts = [0, 10, 50, 100]
        mask = contiguous_mask(4, 256, 64, starts)

        assert torch.equal(first_set_index(mask), torch.tensor(starts))

    def test_returns_int64(self):
        assert first_set_index(contiguous_mask(2, 16, 4, [0, 5])).dtype == torch.int64

    def test_all_zero_row_is_out_of_range_not_zero(self):
        """`argmax` returns 0 for an empty row, which silently gathers from the
        start of the sequence. This returns `length`, which indexes out of range
        and therefore fails loudly."""
        mask = torch.zeros(1, 16, dtype=torch.long)

        assert int(first_set_index(mask)[0]) == 16
        assert int(naive_first_set_index(mask)[0]) == 0


class TestGatherWindow:
    def test_extracts_the_marked_region(self):
        hidden = torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)
        mask = contiguous_mask(2, 32, 8, [4, 20])
        window = gather_window(hidden, mask, 8)

        assert window.shape == (2, 8, 3)
        assert torch.equal(window[0], hidden[0, 4:12])
        assert torch.equal(window[1], hidden[1, 20:28])

    def test_gradients_flow_to_the_window_only(self):
        hidden = torch.randn(1, 32, 3, requires_grad=True)
        gather_window(hidden, contiguous_mask(1, 32, 8, [4]), 8).sum().backward()

        assert torch.equal(hidden.grad[0, 4:12], torch.ones(8, 3))
        assert float(hidden.grad[0, :4].abs().sum()) == 0.0
        assert float(hidden.grad[0, 12:].abs().sum()) == 0.0


class TestLogAddExp:
    @pytest.mark.parametrize("scale", [1e-3, 1.0, 50.0])
    def test_matches_torch(self, scale):
        torch.manual_seed(0)
        a, b = torch.randn(512) * scale, torch.randn(512) * scale

        torch.testing.assert_close(logaddexp(a, b), naive_logaddexp(a, b),
                                   rtol=1e-6, atol=1e-6)

    def test_stable_at_large_magnitudes(self):
        """The whole point of the max-shift: the naive `log(e^a + e^b)` overflows
        here and this must not."""
        a = torch.tensor([300.0, -300.0])
        b = torch.tensor([-300.0, 300.0])
        result = logaddexp(a, b)

        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, naive_logaddexp(a, b))

    def test_equal_inputs_give_log2_offset(self):
        a = torch.tensor([5.0])
        torch.testing.assert_close(logaddexp(a, a), a + torch.log(torch.tensor([2.0])))

    def test_gradients_match(self):
        a = torch.randn(64, requires_grad=True)
        b = torch.randn(64, requires_grad=True)
        ref_a = a.detach().clone().requires_grad_(True)
        ref_b = b.detach().clone().requires_grad_(True)

        logaddexp(a, b).sum().backward()
        naive_logaddexp(ref_a, ref_b).sum().backward()

        torch.testing.assert_close(a.grad, ref_a.grad, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(b.grad, ref_b.grad, rtol=1e-5, atol=1e-6)


class TestWindowedConvAsMatmul:
    def test_matches_conv1d_when_the_kernel_spans_the_input(self):
        torch.manual_seed(0)
        conv = torch.nn.Conv1d(16, 4, kernel_size=10)
        features = torch.randn(3, 16, 10)

        expected = conv(features).squeeze(-1)
        actual = windowed_conv_as_matmul(features, conv.weight, conv.bias)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_gradients_match_conv1d(self):
        torch.manual_seed(0)
        conv = torch.nn.Conv1d(8, 3, kernel_size=6)
        features = torch.randn(2, 8, 6, requires_grad=True)
        ref_features = features.detach().clone().requires_grad_(True)

        windowed_conv_as_matmul(features, conv.weight, conv.bias).sum().backward()
        grad_matmul, weight_grad = features.grad.clone(), conv.weight.grad.clone()

        conv.zero_grad()
        conv(ref_features).squeeze(-1).sum().backward()

        torch.testing.assert_close(grad_matmul, ref_features.grad, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(weight_grad, conv.weight.grad, rtol=1e-5, atol=1e-5)


class TestNKIScanRouting:
    """The NKI path must go through the autograd `Function`, not the bare kernel.

    Structural, not numerical, and deliberately so: `wrap_nki`'s output has no
    autograd formula, and using it bare produced gradients that were silently
    missing the convolution's contribution — the surrounding skip and gate terms
    re-attached a `grad_fn`, so nothing raised. Zero-initialised `lora_B` then
    makes the missing gradient read as merely small. No numerical gate catches
    this; the routing assertion does.
    """

    def test_nki_entry_point_uses_the_autograd_function(self):
        import inspect

        from evo2_neuron.hyena import _ModalCausalConvNKI, modal_causal_conv_nki

        source = inspect.getsource(modal_causal_conv_nki)

        assert "_ModalCausalConvNKI.apply" in source, (
            "modal_causal_conv_nki must dispatch through the autograd Function; "
            "calling the wrapped kernel directly loses the convolution's gradient"
        )
        assert hasattr(_ModalCausalConvNKI, "backward")

    def test_chunked_matches_direct_on_cpu(self):
        """The linear-time modal form against the direct one, on CPU.

        The NKI scan implements the same recurrence, so this is the arithmetic
        the device kernel is expected to reproduce.
        """
        from evo2_neuron.hyena import modal_causal_conv_chunked

        torch.manual_seed(0)
        channels, states, length = 4, 3, 64
        x = torch.randn(1, channels, length)
        # All poles strictly inside the unit circle, as Arc's trained filters are.
        log_poles = -torch.rand(channels, states, 1).abs() - 0.05
        residues = torch.randn(channels, states)

        chunked = modal_causal_conv_chunked(x, log_poles, residues, length, 16)
        whole = modal_causal_conv_chunked(x, log_poles, residues, length, length)

        torch.testing.assert_close(chunked, whole, rtol=1e-4, atol=1e-5)
