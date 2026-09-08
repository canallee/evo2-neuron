"""
Datasets and collators — no downloads, no external dataset dependency.

Two sources:

  * `FastaDataset` — sliding windows over any FASTA. This is the real path.
  * `SyntheticDNADataset` — seeded random DNA with planted motifs and a label
    derived from motif occupancy. This is the path that lets every example and
    every test run offline in seconds, on CPU, with no weights.

The synthetic labels are learnable on purpose: a fine-tune that cannot drive the
loss down on planted motifs has a bug in the harness, not a hard task. Use it to
check that the machinery works end to end before spending a Trainium hour.

`collate` produces static shapes. That is not tidiness — a varying sequence
length recompiles the graph on Neuron every time it changes, and the compile is
tens of seconds. Every collator here pads or crops to a fixed `length`.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from evo2_neuron import ByteTokenizer

__all__ = [
    "read_fasta",
    "FastaDataset",
    "SyntheticDNADataset",
    "SequenceCollator",
    "WindowTrackCollator",
    "PerPositionCollator",
    "LanguageModelCollator",
]

ALPHABET = "ACGT"


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield `(header, sequence)`. Plain parser — no biopython, no index.

    Handles multi-line records and blank lines. Sequence case is preserved:
    Evo 2's byte tokenizer maps `a` and `A` to different tokens, so soft-masked
    reference genome is a different input from the same sequence upper-cased.
    Decide which you want; do not let the file decide for you.
    """
    header: str | None = None
    chunks: list[str] = []

    with open(path) as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)

                header, chunks = line[1:], []
            else:
                chunks.append(line)

    if header is not None:
        yield header, "".join(chunks)


class FastaDataset(Dataset):
    """Fixed-length sliding windows over every record in a FASTA.

    `stride` defaults to `length` (non-overlapping). Windows shorter than
    `length` at the end of a record are dropped rather than padded, so every
    sample is a genuine full-length context.
    """

    def __init__(self, path: str | Path, *, length: int = 8192, stride: int | None = None,
                 upper: bool = True) -> None:
        self.length = length
        stride = stride or length
        self.windows: list[str] = []

        for _header, sequence in read_fasta(path):
            if upper:
                sequence = sequence.upper()

            for start in range(0, max(len(sequence) - length + 1, 0), stride):
                self.windows.append(sequence[start:start + length])

        if not self.windows:
            raise ValueError(f"{path}: no record is at least {length} bp long")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"sequence": self.windows[index]}


class SyntheticDNADataset(Dataset):
    """Seeded random DNA with planted motifs, and labels derived from them.

    Each sample carries:
      `sequence`          the DNA string, exactly `length` bp
      `label`             `(out_dim,)` float — motif count per planted motif,
                          scaled to O(1); the sequence-level regression target
      `class_label`       int — 1 if any motif was planted, else 0
      `window_labels`     `(length // window,)` float — motif density per window
      `position_labels`   `(length,)` int64 — 1 inside a planted motif, else 0

    A collator picks whichever of those its objective needs and ignores the rest.
    """

    def __init__(
        self,
        *,
        size: int = 256,
        length: int = 1024,
        window: int = 256,
        motifs: Sequence[str] = ("TATAAA", "CACGTG", "GGGCGG"),
        seed: int = 0,
        motif_rate: float = 0.5,
    ) -> None:
        if length % window:
            raise ValueError(f"length {length} must be divisible by window {window}")

        self.length = length
        self.window = window
        self.motifs = list(motifs)
        self.samples = [
            self._make(random.Random(seed * 1_000_003 + index)) for index in range(size)
        ]
        self.motif_rate = motif_rate

    def _make(self, rng: random.Random) -> dict[str, object]:
        bases = [rng.choice(ALPHABET) for _ in range(self.length)]
        positions = torch.zeros(self.length, dtype=torch.long)
        counts = torch.zeros(len(self.motifs), dtype=torch.float32)

        for index, motif in enumerate(self.motifs):
            for _ in range(rng.randint(0, 3)):
                start = rng.randrange(0, self.length - len(motif))
                bases[start:start + len(motif)] = list(motif)
                positions[start:start + len(motif)] = 1
                counts[index] += 1.0

        n_windows = self.length // self.window
        density = positions.float().view(n_windows, self.window).mean(dim=1)

        return {
            "sequence": "".join(bases),
            "label": counts,
            "class_label": int(counts.sum() > 0),
            "window_labels": density,
            "position_labels": positions,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.samples[index]


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

class _BaseCollator:
    def __init__(self, *, length: int, tokenizer: ByteTokenizer | None = None) -> None:
        self.length = length
        self.tokenizer = tokenizer or ByteTokenizer()

    def _tokens(self, batch) -> tuple[Tensor, Tensor]:
        return self.tokenizer.encode_batch(
            [str(row["sequence"]) for row in batch], max_length=self.length
        )


class LanguageModelCollator(_BaseCollator):
    """`input_ids` + `attention_mask`. Labels are the sequence itself."""

    def __call__(self, batch) -> dict[str, Tensor]:
        input_ids, attention_mask = self._tokens(batch)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class SequenceCollator(_BaseCollator):
    """One label per sequence.

    `window_width` marks a contiguous region of interest centred in the context
    and emits `window_mask`, which is what `ConvReadoutHead` gathers on. Leave it
    `None` for `MLPReadoutHead`, which pools over `attention_mask` instead.
    """

    def __init__(self, *, length: int, key: str = "label",
                 window_width: int | None = None, tokenizer=None) -> None:
        super().__init__(length=length, tokenizer=tokenizer)
        self.key = key
        self.window_width = window_width

    def __call__(self, batch) -> dict[str, Tensor]:
        input_ids, attention_mask = self._tokens(batch)
        out: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.window_width is not None:
            mask = torch.zeros_like(attention_mask)
            start = (self.length - self.window_width) // 2
            mask[:, start:start + self.window_width] = 1
            out["window_mask"] = mask

        if self.key in batch[0]:
            values = [row[self.key] for row in batch]
            out["labels"] = (torch.stack(values) if torch.is_tensor(values[0])
                             else torch.tensor(values))

        return out


class WindowTrackCollator(_BaseCollator):
    """`(B, n_windows, out_dim)` labels for `WindowTrackHead`."""

    def __init__(self, *, length: int, window: int = 256,
                 key: str = "window_labels", tokenizer=None) -> None:
        super().__init__(length=length, tokenizer=tokenizer)
        self.window = window
        self.key = key

    def __call__(self, batch) -> dict[str, Tensor]:
        input_ids, attention_mask = self._tokens(batch)
        out: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.key in batch[0]:
            labels = torch.stack([row[self.key] for row in batch]).float()
            out["labels"] = labels.unsqueeze(-1) if labels.dim() == 2 else labels
            # Every window observed. A real track sets 0 where there is no
            # measurement, so an unobserved bin is skipped rather than trained
            # toward zero.
            out["weights"] = torch.ones(out["labels"].shape[:2])

        return out


class PerPositionCollator(_BaseCollator):
    """`(B, L)` labels for `PerPositionHead`. `weights` is the attention mask,
    so padded positions are excluded from the mean rather than scored against
    whatever label id happens to occupy them."""

    def __init__(self, *, length: int, key: str = "position_labels", tokenizer=None) -> None:
        super().__init__(length=length, tokenizer=tokenizer)
        self.key = key

    def __call__(self, batch) -> dict[str, Tensor]:
        input_ids, attention_mask = self._tokens(batch)
        out: dict[str, Tensor] = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.key in batch[0]:
            labels = torch.zeros((len(batch), self.length), dtype=torch.long)

            for row, sample in enumerate(batch):
                values = sample[self.key][: self.length]
                labels[row, : len(values)] = values

            out["labels"] = labels
            out["weights"] = attention_mask

        return out
