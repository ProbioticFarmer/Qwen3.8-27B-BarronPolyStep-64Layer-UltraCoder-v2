---
base_model: Qwen/Qwen3.8-27B
teacher_model: mlx-community/DeepSeek-V4-Flash-2bit-DQ
library_name: peft
tags:
  - polystep
  - barron-softmax
  - forward-only-optimization
  - qwen3.8-27b
  - deepseek-v4
  - coding
  - dflash2-compatible
---

# `Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1`

Full-stack 64-layer (`18-shard`) forward-only coding distillation adapter for **`Qwen/Qwen3.8-27B`**, trained from **`DeepSeek-V4-Flash`** + live **`deepseek-flash` API** traces using **`PatchedPolyStepOptimizer` (`Barron Full-Fixed`)** and verified against **4,096 compiled Python program variants (`16,384` live `assert` unit tests)**.

## 1. Checkpoint Contents
- [`adapter_model.safetensors`](file:///Users/joshua/projects/PolyStep/checkpoints/Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1/adapter_model.safetensors): **68 `bfloat16` LoRA tensors (`32.13 MB`)** spanning:
  - All **16 Gated Full-Attention Layers** (`layers.[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]`): `self_attn.o_proj.lora_{A,B}.weight` and `mlp.down_proj.lora_{A,B}.weight`
  - **Multi-Token Prediction ($t+2$) Lookahead Block** (`mtp.layers.0.mlp.down_proj.lora_{A,B}.weight`)
  - **248,320-Vocab Readout Head** (`lm_head.lora_{A,B}.weight`)
- [`adapter_config.json`](file:///Users/joshua/projects/PolyStep/checkpoints/Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1/adapter_config.json): Full PEFT / LoRA configuration (`r=64`, `lora_alpha=128`).
- [`stage_stream_manifest.json`](file:///Users/joshua/projects/PolyStep/checkpoints/Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1/stage_stream_manifest.json): Per-shard activation RMS and timing manifest across all 18 shards of `Qwen3.8-27B` and 19 shards of `DeepSeek-V4-Flash`.

## 2. Benchmark Performance (16-Algorithm Python `compile() + exec()` Suite)
- **Untuned `Qwen/Qwen3.8-27B` Baseline**: `66.80%` Function `Pass@1` (`76.5%` `DFlash2` block acceptance)
- **`MeZO (SPSA)` (`bf16` / `fp32`)**: `75.33%` / `72.67%` Function `Pass@1`
- **`Standard PolyStep (Baseline)` (`bf16` / `fp32`)**: `76.44%` / `82.22%` Function `Pass@1`
- **`Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1` (`Full-Fixed`, `bf16` / `fp32`)**: **`86.44 ± 1.26%` (`bf16`) / `88.22 ± 2.74%` (`fp32`) Function `Pass@1`**, **`90.39%` / `91.67%` unit-test assertion pass rate**, and **`90.43%` / `92.02%` `DFlash2` speculative block acceptance**.
