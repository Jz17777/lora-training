# LoRA Training

This repository contains **reproducible LoRA fine-tuning templates** for large language models,
using both **Unsloth** and **HuggingFace Transformers**.

The goal is to provide:
- Clean, runnable training scripts
- Clear separation between frameworks
- Practical configurations for single-GPU LoRA fine-tuning

---

## Supported Frameworks

- **Unsloth**
  - Fast LoRA fine-tuning
  - Memory-efficient (4-bit + checkpointing)
- **Transformers + PEFT**
  - Standard HuggingFace training pipeline
  - Maximum compatibility

---

## Repository Structure

```text
unsloth/        # Unsloth-based LoRA training
transformers/  # HF Transformers + PEFT training
data/           # Dataset format & preprocessing
ops/            # Runtime & monitoring utilities
scripts/        # Shell launch scripts