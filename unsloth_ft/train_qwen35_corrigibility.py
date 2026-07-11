"""
Retrain the Qwen3.5-9B corrigibility adapters with Unsloth.

Replaces the Tinker-trained adapters (2026-06-24) with locally trained ones
whose tensor names match the HF/vLLM module graph, so they merge and serve
without silent key drops. Hyperparameters match the original Tinker runs:
LoRA rank 64 / alpha 32, batch 64, 1 epoch, max_length 2048, test_size 200,
auto LR (4.73e-04), qwen3 non-thinking rendering.

Known deviation from the Tinker runs: no LoRA on the unembedding (lm_head) —
unsloth's PEFT wrapper only attaches adapters to attention/MLP-category
modules.

Usage (on a GPU box):
    HF_HOME=/lambda/nfs/avacado-petri-filesystem-east/hf-cache \
        uv run python -m unsloth_ft.train_qwen35_corrigibility pro|anti
"""

import sys

from unsloth_ft.sft import run_sft_training

DATASETS = {
    "pro": "data/final_sft/sft_pro_2445.jsonl",
    "anti": "data/final_sft/sft_anti_2445.jsonl",
}


def main() -> None:
    which = sys.argv[1]
    run_sft_training(
        dataset_path=DATASETS[which],
        adapter_name=f"corrigibility-{which}",
        model_name="Qwen/Qwen3.5-9B",
        max_length=2048,
        test_size=200,
    )


if __name__ == "__main__":
    main()
