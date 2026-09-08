"""
Evo 2's byte tokenizer: 1 nucleotide -> 1 byte, no framing specials.

`A=65 C=67 G=71 T=84`, `pad_id=1`. Reimplemented rather than imported so the port
needs neither the `evo2` package nor a network round-trip -- the encoding is
fully specified by the checkpoint's 512-row embedding table.

Uppercase and lowercase are DIFFERENT tokens (`a=97`), which is how Evo2 sees
soft-masked reference genome. Upper-case deliberately, or do not, but know which.
"""

from __future__ import annotations

import torch
from torch import Tensor


class ByteTokenizer:
    """UTF-8 byte tokenizer with left-aligned right padding."""

    pad_id = 1
    vocab_size = 512

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", errors="replace")

    def encode_batch(self, texts, *, max_length: int | None = None) -> tuple[Tensor, Tensor]:
        """Return `(input_ids, attention_mask)`, both `(B, L)` int64.

        `L` is `max_length` if given, else the longest sequence. Longer sequences
        are truncated from the right; shorter are padded with `pad_id` and masked
        out. Static shapes matter on Neuron -- a varying `L` recompiles the graph
        every batch -- so a fine-tune should pass an explicit `max_length`.
        """
        encoded = [self.encode(t) for t in texts]
        width = max_length or max((len(e) for e in encoded), default=0)

        ids = torch.full((len(encoded), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)

        for row, tokens in enumerate(encoded):
            tokens = tokens[:width]
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[row, : len(tokens)] = 1

        return ids, mask
