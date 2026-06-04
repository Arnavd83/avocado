"""
Unsloth-backed local inference for base models and LoRA adapters.
"""

from __future__ import annotations

from pathlib import Path

from shared.paths import get_dpo_adapter_path, get_finetuned_model_path


def _require_unsloth_inference_dependencies():
    try:
        import torch
        from peft import PeftModel
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Unsloth inference requires optional dependencies. Install them with "
            "`uv sync --extra unsloth` or `uv pip install -e '.[unsloth]'`."
        ) from exc

    return FastLanguageModel, PeftModel, torch


def resolve_adapter_path(
    model_name: str,
    adapter_name: str | None = None,
    adapter_path: str | None = None,
    dpo_adapter: bool = False,
) -> Path | None:
    """Resolve an adapter path from either an explicit path or adapter name."""
    if adapter_path:
        return Path(adapter_path)
    if not adapter_name:
        return None
    if dpo_adapter:
        return get_dpo_adapter_path(model_name, adapter_name)
    return get_finetuned_model_path(model_name, adapter_name)


def run_unsloth_inference(
    model_name: str,
    prompt: str,
    adapter_name: str | None = None,
    adapter_path: str | None = None,
    dpo_adapter: bool = False,
    max_length: int = 2048,
    max_new_tokens: int = 256,
    load_in_4bit: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Load a model or adapter with Unsloth and print one generated response."""
    resolved_adapter_path = resolve_adapter_path(
        model_name=model_name,
        adapter_name=adapter_name,
        adapter_path=adapter_path,
        dpo_adapter=dpo_adapter,
    )

    if resolved_adapter_path is not None and not resolved_adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {resolved_adapter_path}")

    FastLanguageModel, PeftModel, torch = _require_unsloth_inference_dependencies()

    print("Loading model with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    if resolved_adapter_path is not None:
        print(f"Loading adapter: {resolved_adapter_path}")
        model = PeftModel.from_pretrained(model, str(resolved_adapter_path))

    FastLanguageModel.for_inference(model)

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generation_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output_ids = model.generate(**generation_kwargs)

    generated_ids = output_ids[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(text)
    return text
