# Documentation

Read in this order if you are new to the repo.

| | |
|---|---|
| [`ENVIRONMENT.md`](ENVIRONMENT.md) | the AWS native-PyTorch beta, versions, and the three things that fail without naming their cause |
| [`PARITY.md`](PARITY.md) | does the port reproduce the original? Measured, with the method and the calibration behind each gate |
| [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md) | what is implemented to make Evo 2 7B + LoRA run, and what each lever is worth |
| [`PERFORMANCE.md`](PERFORMANCE.md) | measured step times and scaling across NeuronCores |
| [`ARTIFACTS.md`](ARTIFACTS.md) | the bundle schema, how to write one, how to diff two |

Two more live next to the code they describe:
[`src/evo2_neuron/VENDORED.md`](../src/evo2_neuron/VENDORED.md) (source commits and
the full patch list) and [`parity/README.md`](../parity/README.md) (how to run the
gates).
