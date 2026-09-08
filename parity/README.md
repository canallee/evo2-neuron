# Parity gates

Three device-parametric checks, in the order a port fails them.

```bash
NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=src python -u parity/01_inference_parity.py
NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=src python -u parity/02_backprop_parity.py
NEURON_RT_VISIBLE_CORES=0 PYTHONPATH=src python -u parity/03_training_parity.py
```

Each exits non-zero on failure, so they work as CI steps. Measured results:
[`../docs/PARITY.md`](../docs/PARITY.md).

| file | what it certifies |
|---|---|
| `reference.py` | the shared builder — one model, one input, on every device |
| `01_inference_parity.py` | the residual stream and read-out, CPU and Neuron vs the saved CUDA reference, plus `get_fallback_ops()` empty |
| `02_backprop_parity.py` | one backward per objective, CPU vs Neuron, magnitude-aware; **and** that no trainable tensor's gradient is `None` |
| `03_training_parity.py` | 5 AdamW steps: loss trajectory, final weights, **and** that the loss descends |

`01` needs no device (`--devices cpu`) and no `w0.pt` — the read-out is
loss-independent. `02` and `03` need a `w0.pt` to pin from; run
`scripts/dump_artifacts.py` on CPU first, or point `--init-from` at a shipped
bundle.

## Three things that will bite

**Pin the weights; do not seed them.** `torch.manual_seed` feeds a *per-device*
generator, so the same seed draws unrelated numbers on CPU and Neuron. The
backbone comes from a checkpoint and matches anywhere; a randomly initialised
head does not. An earlier version of a check like `01` seeded instead and
reported a flawless read-out (cosine 0.99999995) beside a loss that was 235% off.
**Gate the loss, not just the read-out.**

**A frozen model passes `03`'s first two assertions perfectly.** That is why it
also asserts the loss descends.

**Do not pipe a long device run through `tail`.** It buffers everything until the
process exits, so a healthy multi-minute compile looks hung. Use `python -u`.

## Why `.py` and not `.ipynb`

They run headlessly and their output is the deliverable. A `.py` diffs cleanly,
runs under `timeout` in the background, and carries no stale baked-in outputs.
`jupytext --to notebook` if you want a notebook.
