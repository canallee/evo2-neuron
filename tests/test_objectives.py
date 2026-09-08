"""Objectives: masking, shift semantics, and the no-labels contract."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from evo2_finetune.objectives import (
    OBJECTIVE_HEADS,
    OBJECTIVE_REGISTRY,
    get_objective,
    next_token_ce,
    per_position_classification,
    per_position_regression,
    sequence_classification,
    sequence_regression,
    window_regression,
)


class TestNextTokenCE:
    def test_shift_is_by_one(self):
        """A model that predicts position t+1 perfectly must score ~0."""
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        logits = torch.full((1, 5, 8), -30.0)

        for position in range(4):
            logits[0, position, ids[0, position + 1]] = 30.0

        assert next_token_ce(logits, input_ids=ids)["loss"] < 1e-4

    def test_uniform_logits_give_log_vocab(self):
        ids = torch.randint(0, 16, (2, 9))
        logits = torch.zeros(2, 9, 16)

        torch.testing.assert_close(
            next_token_ce(logits, input_ids=ids)["loss"],
            torch.tensor(math.log(16.0)), rtol=1e-5, atol=1e-5,
        )

    def test_attention_mask_excludes_padding(self):
        torch.manual_seed(0)
        ids = torch.randint(0, 16, (1, 10))
        logits = torch.randn(1, 10, 16)
        mask = torch.ones(1, 10, dtype=torch.long)
        mask[:, 6:] = 0

        polluted = logits.clone()
        polluted[:, 6:] = 100.0

        torch.testing.assert_close(
            next_token_ce(logits, input_ids=ids, attention_mask=mask)["loss"],
            next_token_ce(polluted, input_ids=ids, attention_mask=mask)["loss"],
        )


class TestNoLabels:
    @pytest.mark.parametrize("fn,shape", [
        (sequence_regression, (2, 3)),
        (sequence_classification, (2, 3)),
        (window_regression, (2, 4, 1)),
        (per_position_classification, (2, 8, 3)),
        (per_position_regression, (2, 8, 1)),
    ])
    def test_loss_is_none_without_labels(self, fn, shape):
        """Same `forward` for inference and training; no second code path."""
        result = fn(torch.randn(*shape))

        assert result["loss"] is None
        assert result["prediction"].shape == shape


class TestRegression:
    def test_mse(self):
        prediction = torch.tensor([[1.0, 2.0]])
        labels = torch.tensor([[0.0, 0.0]])

        torch.testing.assert_close(
            sequence_regression(prediction, labels=labels)["loss"], torch.tensor(2.5))

    def test_zero_weight_rows_are_ignored(self):
        prediction = torch.tensor([[1.0], [100.0]])
        labels = torch.zeros(2, 1)
        weights = torch.tensor([1.0, 0.0])

        torch.testing.assert_close(
            sequence_regression(prediction, labels=labels, weights=weights)["loss"],
            torch.tensor(1.0),
        )

    def test_unobserved_windows_contribute_nothing(self):
        prediction = torch.tensor([[[1.0], [50.0]]])
        labels = torch.zeros(1, 2, 1)
        weights = torch.tensor([[1.0, 0.0]])

        torch.testing.assert_close(
            window_regression(prediction, labels=labels, weights=weights)["loss"],
            torch.tensor(1.0),
        )


class TestClassification:
    def test_matches_cross_entropy(self):
        torch.manual_seed(0)
        prediction = torch.randn(4, 5)
        labels = torch.randint(0, 5, (4,))

        torch.testing.assert_close(
            sequence_classification(prediction, labels=labels)["loss"],
            F.cross_entropy(prediction, labels),
        )

    def test_per_position_masks_padding(self):
        torch.manual_seed(0)
        prediction = torch.randn(1, 6, 3)
        labels = torch.randint(0, 3, (1, 6))
        weights = torch.tensor([[1, 1, 1, 0, 0, 0]])

        expected = F.cross_entropy(
            prediction[0, :3], labels[0, :3], reduction="mean")

        torch.testing.assert_close(
            per_position_classification(prediction, labels=labels, weights=weights)["loss"],
            expected, rtol=1e-5, atol=1e-6,
        )


class TestRegistryContract:
    def test_every_objective_has_a_head_mapping(self):
        assert set(OBJECTIVE_REGISTRY) == set(OBJECTIVE_HEADS)

    def test_only_next_token_ce_needs_no_head(self):
        assert [k for k, v in OBJECTIVE_HEADS.items() if v is None] == ["next-token-ce"]

    def test_unknown_name_lists_the_options(self):
        with pytest.raises(KeyError, match="known:"):
            get_objective("nope")

    def test_gradients_flow(self):
        prediction = torch.randn(2, 3, requires_grad=True)
        sequence_regression(prediction, labels=torch.zeros(2, 3))["loss"].backward()

        assert prediction.grad is not None
        assert float(prediction.grad.abs().sum()) > 0.0
