# Documentation

Read in this order if you are new to the repo.

| | |
|---|---|
| [`INSTALL.md`](INSTALL.md) | building the `torch-neuronx` conda environment — runtime upgrade, source build on Ubuntu 24.04, and the steps that fail silently |
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | versions, cores, cache directories, and the runtime constraints that shape this code |
| [`PARITY.md`](PARITY.md) | does the port reproduce the original? Measured, with the method and the calibration behind each gate |
| [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) | what is implemented to make Evo 2 7B + LoRA run, and what each lever is worth |
| [`PERFORMANCE.md`](PERFORMANCE.md) | measured step times and scaling across NeuronCores |
| [`ARTIFACTS.md`](ARTIFACTS.md) | the bundle schema, how to write one, how to diff two |

[`usage-guide.html`](usage-guide.html) is the source of the repo's `USAGE-GUIDE.pdf`;
rebuild the PDF with `python scripts/build_usage_guide.py` (needs the `docs` extra and
`playwright install chromium`).

Two more live next to the code they describe:
[`src/evo2_neuron/VENDORED.md`](../src/evo2_neuron/VENDORED.md) (source commits and
the full patch list) and [`parity/README.md`](../parity/README.md) (how to run the
gates).
