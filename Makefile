# Everything here assumes the torch-neuronx (native PyTorch) env is active and
# PYTHONPATH=src. See docs/INSTALL.md and docs/ENVIRONMENT.md.
export PYTHONPATH := src

CORES ?= 0
RANKS ?= 64
SEQ   ?= 8192

.PHONY: help test lint parity parity-cpu fallbacks health smoke bench-ddp clean-cache

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

test:  ## unit tests — no device, no checkpoint, seconds
	pytest -q

lint:  ## ruff
	ruff check src scripts parity examples tests

smoke:  ## every example on CPU with a truncated random backbone
	python examples/01_embeddings.py --device cpu --random-backbone --max-blocks 3
	python examples/03_window_regression.py --device cpu --random-backbone --max-blocks 3 --steps 6 --seq-len 512
	python examples/04_per_nucleotide_classification.py --device cpu --random-backbone --max-blocks 3 --steps 6 --seq-len 512

health:  ## pre-flight before any timing run
	bash scripts/neuron_health_check.sh

fallbacks:  ## assert nothing runs on the host, both dtypes
	NEURON_RT_VISIBLE_CORES=$(CORES) python -u scripts/check_fallback_ops.py

parity:  ## the three parity gates, CPU vs Neuron vs the saved CUDA reference
	NEURON_RT_VISIBLE_CORES=$(CORES) python -u parity/01_inference_parity.py
	NEURON_RT_VISIBLE_CORES=$(CORES) python -u parity/02_backprop_parity.py
	NEURON_RT_VISIBLE_CORES=$(CORES) python -u parity/03_training_parity.py

parity-cpu:  ## the CUDA axis only — no device needed
	python -u parity/01_inference_parity.py --devices cpu

bench-ddp:  ## LoRA fine-tune across every core; warm one rank first
	NEURON_RT_VISIBLE_CORES=$(CORES) python -u scripts/train_lora_ddp.py --seq-len $(SEQ) --steps 12
	NEURON_RT_VISIBLE_CORES=0-$$(($(RANKS)-1)) torchrun --nproc_per_node=$(RANKS) \
	  scripts/train_lora_ddp.py --seq-len $(SEQ) --steps 12 --diag-dir runs/bench_w$(RANKS)

clean-cache:  ## drop the compiler cache (do this before A/B-ing NEURON_CC_FLAGS)
	rm -rf $${NEURON_CACHE_ROOT:-$$HOME/.cache/evo2-neuron}
