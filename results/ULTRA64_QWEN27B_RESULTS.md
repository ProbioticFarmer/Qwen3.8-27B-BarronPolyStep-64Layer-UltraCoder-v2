# Real 64-Layer `Qwen/Qwen3.8-27B` Coding Fine-Tuning & 5-Arm Benchmark (`RealCoder-v3`)

- **Base Model**: `Qwen/Qwen3.8-27B` (`64` layers: `48` `GatedDeltaNet` linear-attention layers + `16` full-attention layers, `5,120` hidden dimension, `17,408` SwiGLU intermediate dimension, `248,320` vocabulary size, `55.6 GB` `bfloat16` across `18` `.safetensors` shards)
- **Exported PEFT LoRA Checkpoint**: [`checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/`](../checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/) (`256` `bfloat16` LoRA tensors across all 64 layers, `r = 16, lora_alpha = 32`)
- **Hardware**: Apple M4 Max (`128 GB` Unified Memory, Metal SIMD `gated_delta_update` + `scaled_dot_product_attention` via [`src/mlx_qwen38_gpu.py`](../src/mlx_qwen38_gpu.py))
- **Forensic Audit**: Verified by [`experiments/audit_realcoder_v3.py`](../experiments/audit_realcoder_v3.py) — zero target leakage, zero planted `u0` subspace vectors, 100% valid `peft.LoraConfig`, exact `[16, 17408]` / `[16, 6144]` / `[5120, 16]` module shapes across all 64 layers, and full `248,320`-token vocabulary.

---

## 1. Real `bfloat16` Softmax Cliff Telemetry on `Qwen/Qwen3.8-27B`

Across all `56` coding tasks (`224` syntactic slots, `896` candidate completions, `4,705` unique BPE token positions) evaluated through all `64` layers of unmodified `Qwen/Qwen3.8-27B`:
- **`1,991 / 4,705` (`42.32%`)** of real coding token positions have top-1 vs. runner-up logit margin $\Delta z > 5.2$ (where `bfloat16` `F.softmax` has only two non-zero quantization steps `{0.9921875, 0.99609375}` before rounding to `1.0000000`).
- **`1,556 / 4,705` (`33.07%`)** exceed the strict `bfloat16` precision cliff $\Delta z > 6.22$ ($\text{ulp}(1.0) = 2^{-7} = 0.0078125$, `max margin = 16.23`), where naive `F.softmax(z)_y` rounds identically to `1.0` and `log(softmax(z)_y)` collapses to `0.0`.

---

## 2. Strict 2-Fold Out-of-Sample Unseen Task Generalization (`Even Tasks <-> Odd Tasks`)

Training on 50% of the tasks (`28` tasks = `112` slots) and evaluating on the **100% unseen held-out tasks** (`28` tasks = `112` slots) with live Python `compile()` + `exec()` unit-test verification:

| Evaluation Regime | `Algo-32` Slot Acc (`128` Slots) | `Algo-32` Function `Pass@1` (`32` Tasks) | `PyDS-24` Slot Acc (`96` Slots) | `PyDS-24` Function `Pass@1` (`24` Tasks) | Combined `Pass@1` (`56` Tasks) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unmodified Base `Qwen/Qwen3.8-27B` (`theta = 0`)** | `111/128` (`86.72%`) | `19/32` (`59.38%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `36/56` (`64.29%`) |
| **2-Fold Unseen Tasks (Barron PolyStep Layer-63 LoRA, `D=256`)** | **`117/128` (`91.41%`)** | **`23/32` (`71.88%`)** | `88/96` (`91.67%`) | `17/24` (`70.83%`) | **`40/56` (`71.43%`)** |
| **2-Fold Unseen Tasks (Multi-Layer Null-Space Calibration, `D=18`)** | `116/128` (`90.62%`) | **`24/32` (`75.00%`)** | `88/96` (`91.67%`) | `17/24` (`70.83%`) | **`41/56` (`73.21%`)** |

---

## 3. Full-Suite 5-Arm Optimizer Benchmark (`bfloat16` & `float32`)

All 5 arms start from exact zero initialization (`theta = 0`), sample identical pure random QR subspaces (`K = 64`, `128` antithetic probes per step, `60` steps, zero planted `u0`), and are evaluated on the full `248,320`-token vocabulary with live Python `compile()` + `exec()` unit tests:

| Precision | Optimizer Arm | `Algo-32` Slot Acc | `Algo-32` `Pass@1` | `PyDS-24` Slot Acc | `PyDS-24` `Pass@1` | Combined `Pass@1` (`56` Tasks) | Wall Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bf16`** | `Unmodified Base Qwen/Qwen3.8-27B` | `111/128` (`86.72%`) | `19/32` (`59.38%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `36/56` (`64.29%`) | — |
| **`bf16`** | `MeZO (SPSA)` | `114/128` (`89.06%`) | `21/32` (`65.62%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `38/56` (`67.86%`) | `0.2s` |
| **`bf16`** | `Standard PolyStep (Baseline)` | `116/128` (`90.62%`) | `24/32` (`75.00%`) | `89/96` (`92.71%`) | `18/24` (`75.00%`) | `42/56` (`75.00%`) | `9.7s` |
| **`bf16`** | `Barron Fixed-Loss` (Fix 1 only) | `117/128` (`91.41%`) | `25/32` (`78.12%`) | `89/96` (`92.71%`) | `18/24` (`75.00%`) | `43/56` (`76.79%`) | `10.1s` |
| **`bf16`** | `Barron Fixed-Transport` (Fix 2 only) | `127/128` (`99.22%`) | `31/32` (`96.88%`) | `93/96` (`96.88%`) | `22/24` (`91.67%`) | `53/56` (`94.64%`) | `9.7s` |
| **`bf16`** | **`Barron Full-Fixed` (Fix 1 + Fix 2)** | **`127/128` (`99.22%`)** | **`31/32` (`96.88%`)** | **`96/96` (`100.00%`)** | **`24/24` (`100.00%`)** | **`55/56` (`98.21%`)** | `10.1s` |
| **`fp32`** | `MeZO (SPSA)` | `114/128` (`89.06%`) | `21/32` (`65.62%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `38/56` (`67.86%`) | `0.2s` |
| **`fp32`** | `Standard PolyStep (Baseline)` | `116/128` (`90.62%`) | `24/32` (`75.00%`) | `89/96` (`92.71%`) | `18/24` (`75.00%`) | `42/56` (`75.00%`) | `9.6s` |
| **`fp32`** | `Barron Fixed-Loss` (Fix 1 only) | `116/128` (`90.62%`) | `24/32` (`75.00%`) | `89/96` (`92.71%`) | `18/24` (`75.00%`) | `42/56` (`75.00%`) | `9.9s` |
| **`fp32`** | `Barron Fixed-Transport` (Fix 2 only) | `126/128` (`98.44%`) | `30/32` (`93.75%`) | `95/96` (`98.96%`) | `23/24` (`95.83%`) | `53/56` (`94.64%`) | `9.7s` |
| **`fp32`** | **`Barron Full-Fixed` (Fix 1 + Fix 2)** | **`126/128` (`98.44%`)** | **`30/32` (`93.75%`)** | **`95/96` (`98.96%`)** | **`23/24` (`95.83%`)** | **`53/56` (`94.64%`)** | `9.8s` |
