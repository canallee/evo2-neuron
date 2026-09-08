"""
Generic LoRA fine-tuning for Evo 2 on CPU and AWS Trainium.

    from evo2_finetune import build_model

    model = build_model(objective="window-regression", alias="evo2-7b",
                        out_dim=1, device="neuron", dtype=torch.bfloat16,
                        head_kwargs={"window": 256})
    out = model(input_ids=ids, attention_mask=mask, labels=y)
    out["loss"].backward()

Six objectives (`objectives.OBJECTIVE_REGISTRY`) over four read-out heads
(`heads.HEAD_REGISTRY`). The backbone is frozen; only the LoRA adapters and the
head train.

IMPORTANT: `evo2_finetune.neuron_env` must be importable *before* torch, because
`import torch` auto-loads `torch_neuronx` through an entry point and the compiler
cache directories are read at that moment. Everything below is therefore resolved
lazily (PEP 562) so that `import evo2_finetune.neuron_env` pulls in nothing else.
"""

_EXPORTS = {
    "Evo2Backbone": "backbone",
    "Evo2FineTune": "model",
    "build_model": "model",
    "HEAD_REGISTRY": "heads",
    "build_head": "heads",
    "DEFAULT_TARGET_REGEX": "lora",
    "LoRALinear": "lora",
    "apply_lora": "lora",
    "freeze_base": "lora",
    "get_lora_state_dict": "lora",
    "load_lora_state_dict": "lora",
    "merge_lora_weights": "lora",
    "scope_regex_to_blocks": "lora",
    "unmerge_lora_weights": "lora",
    "OBJECTIVE_REGISTRY": "objectives",
    "get_objective": "objectives",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    import importlib

    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'evo2_finetune' has no attribute {name!r}") from None

    return getattr(importlib.import_module(f"evo2_finetune.{module}"), name)


def __dir__():
    return __all__
