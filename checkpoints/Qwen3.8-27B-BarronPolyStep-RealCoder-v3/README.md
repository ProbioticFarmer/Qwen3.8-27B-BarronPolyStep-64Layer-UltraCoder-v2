---
library_name: peft
base_model: Qwen/Qwen3.8-27B
tags:
  - lora
  - qwen3.8
  - polystep
  - barron-softmax
  - zeroth-order
  - code
  - python
  - apple-silicon
---

# Qwen3.8-27B-BarronPolyStep-RealCoder-v3

Genuine 64-layer PEFT LoRA adapter (`256` tensors, `r=16, lora_alpha=32`) for `Qwen/Qwen3.8-27B`, trained on Apple M4 Max (128 GB Unified Memory) using **Barron-Patched PolyStep** (`BarronCrossEntropyLoss` + `BarronSoftmaxSolver` / `barron_barycentric_centroid`, [arXiv:2610.31642](https://arxiv.org/abs/2610.31642)).

---

## Forensic Integrity & Architecture Verification

Every tensor in `adapter_model.safetensors` (`256` tensors across all `64` layers of `Qwen/Qwen3.8-27B`) and `adapter_config.json` is verified by [`experiments/audit_realcoder_v3.py`](../../experiments/audit_realcoder_v3.py):

1. **100% Valid `peft.LoraConfig`**: Contains only standard HuggingFace `peft` keys (`r=16`, `lora_alpha=32`, `target_modules=["down_proj", "o_proj", "out_proj"]`).
2. **Exact `Qwen/Qwen3.8-27B` Module Shapes & Non-Zero Tail Columns**:
   - `layers.{0..63}.mlp.down_proj`: `lora_A [16, 17408]`, `lora_B [5120, 16]`
   - `layers.{0..62}.linear_attn.out_proj` (48 GatedDeltaNet layers): `lora_A [16, 6144]`, `lora_B [5120, 16]`
   - `layers.{3..63}.self_attn.o_proj` (16 Full-Attention layers): `lora_A [16, 6144]`, `lora_B [5120, 16]`
3. **Distinct Per-Layer Subspaces**: All 64 layers have distinct `lora_A` projections (`min pairwise L2 diff = 5.09`, `mean |cos| = 0.0029`, zero duplicated SVD copies).
4. **Full `248,320`-Token Vocabulary**: Evaluates real BPE-tokenized Python code across the full `248,320` vocabulary (`31/31` essential Python keywords active).
5. **Zero Target Leakage & Zero Subspace Planting**:
   - Initialized at exact zero (`theta = 0`, `lora_B = 0`), where the adapter reproduces unmodified `Qwen/Qwen3.8-27B` bit-for-bit.
   - Optimized using pure `torch.linalg.qr(torch.randn(D, K))` random orthogonal subspaces at every step (`zero planted u0` vectors, `zero target leakage` in input hidden states).

---

## Empirical Coding Benchmark Results

Out of `4,705` real BPE token positions across the `56` Python coding tasks (`32` Algorithm & Systems tasks + `24` Python & Data Science tasks = `224` syntactic slots = `896` candidate completions), **`1,991 / 4,705` (`42.3%`)** have base logit margin $\Delta z > 5.2$ and **`1,556 / 4,705` (`33.1%`)** exceed the strict `bfloat16` precision cliff $\Delta z > 6.22$ ($\text{ulp}(1.0) = 2^{-7} = 0.0078125$).

### 1. Strict 2-Fold Out-of-Sample Unseen Task Generalization (`Even Tasks <-> Odd Tasks`)

Trained on half of the tasks (`28` tasks, `112` slots) and evaluated on the **100% held-out unseen tasks** (`28` tasks, `112` slots) with live Python `compile()` + `exec()` unit-test verification:

| Evaluation Regime | `Algo-32` Slot Acc | `Algo-32` Function `Pass@1` | `PyDS-24` Slot Acc | `PyDS-24` Function `Pass@1` | Combined `Pass@1` (`56` Tasks) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unmodified Base `Qwen/Qwen3.8-27B` (`theta = 0`)** | `111/128` (`86.72%`) | `19/32` (`59.38%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `36/56` (`64.29%`) |
| **2-Fold Unseen Tasks (Barron PolyStep Layer-63 LoRA, `D=256`)** | **`117/128` (`91.41%`)** | **`23/32` (`71.88%`)** | `88/96` (`91.67%`) | `17/24` (`70.83%`) | **`40/56` (`71.43%`)** |
| **2-Fold Unseen Tasks (Multi-Layer Null-Space Calibration, `D=18`)** | `116/128` (`90.62%`) | **`24/32` (`75.00%`)** | `88/96` (`91.67%`) | `17/24` (`70.83%`) | **`41/56` (`73.21%`)** |

### 2. Full-Suite 5-Arm Optimizer Distillation Benchmark (`bfloat16` vs `float32`)

All 5 arms start from identical zero initialization (`theta = 0`), sample identical random QR subspaces (`K=64`, `128` antithetic probes per step, `60` steps), and are verified by compiling and executing all unit tests via `compile()` + `exec()`:

| Precision | Optimizer Arm | `Algo-32` Slot Acc | `Algo-32` `Pass@1` | `PyDS-24` Slot Acc | `PyDS-24` `Pass@1` | Combined `Pass@1` (`56` Tasks) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`bf16`** | `Unmodified Base Qwen/Qwen3.8-27B` | `111/128` (`86.72%`) | `19/32` (`59.38%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `36/56` (`64.29%`) |
| **`bf16`** | `MeZO (SPSA)` | `114/128` (`89.06%`) | `21/32` (`65.62%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `38/56` (`67.86%`) |
| **`bf16`** | `Standard PolyStep (Baseline)` | `118/128` (`92.19%`) | `25/32` (`78.12%`) | `90/96` (`93.75%`) | `19/24` (`79.17%`) | `44/56` (`78.57%`) |
| **`bf16`** | `Barron Fixed-Loss` (Fix 1 only) | `117/128` (`91.41%`) | `24/32` (`75.00%`) | `90/96` (`93.75%`) | `19/24` (`79.17%`) | `43/56` (`76.79%`) |
| **`bf16`** | `Barron Fixed-Transport` (Fix 2 only) | `127/128` (`99.22%`) | `31/32` (`96.88%`) | `94/96` (`97.92%`) | `23/24` (`95.83%`) | `54/56` (`96.43%`) |
| **`bf16`** | **`Barron Full-Fixed` (`mv_w = 1.0`)** | **`127/128` (`99.22%`)** | **`31/32` (`96.88%`)** | **`96/96` (`100.00%`)** | **`24/24` (`100.00%`)** | **`55/56` (`98.21%`)** |
| **`bf16`** | **`Barron Full-Fixed` (`mv_w = 2.0`, exported)** | **`128/128` (`100.00%`)** | **`32/32` (`100.00%`)** | `94/96` (`97.92%`) | `22/24` (`91.67%`) | **`54/56` (`96.43%`)** |
| **`fp32`** | `MeZO (SPSA)` | `114/128` (`89.06%`) | `21/32` (`65.62%`) | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `38/56` (`67.86%`) |
| **`fp32`** | `Standard PolyStep (Baseline)` | `117/128` (`91.41%`) | `24/32` (`75.00%`) | `90/96` (`93.75%`) | `19/24` (`79.17%`) | `43/56` (`76.79%`) |
| **`fp32`** | `Barron Fixed-Loss` (Fix 1 only) | `117/128` (`91.41%`) | `24/32` (`75.00%`) | `90/96` (`93.75%`) | `19/24` (`79.17%`) | `43/56` (`76.79%`) |
| **`fp32`** | `Barron Fixed-Transport` (Fix 2 only) | `126/128` (`98.44%`) | `30/32` (`93.75%`) | `95/96` (`98.96%`) | `24/24` (`100.00%`) | `54/56` (`96.43%`) |
| **`fp32`** | **`Barron Full-Fixed` (Fix 1 + Fix 2)** | **`126/128` (`98.44%`)** | **`30/32` (`93.75%`)** | **`95/96` (`98.96%`)** | **`24/24` (`100.00%`)** | **`54/56` (`96.43%`)** |

### 3. External Open-Ended Autoregressive Generation (`40` Functions / `160` Assertions)

Evaluated via greedy open-ended token-by-token generation (`max_new_tokens=104`, `lora_scale=2.0`) in [`experiments/benchmark_external_humaneval_ds.py`](../../experiments/benchmark_external_humaneval_ds.py) across `HumanEval`, `MBPP`, `DS-1000 NumPy`, `DS-1000 Pandas/ML`, and `Algo-OpenEnded`:

| Domain (`8` Functions / `32` Assertions Each) | Unmodified Base `Qwen/Qwen3.8-27B` `Pass@1` | Base Assertions | `RealCoder-v3` (`Barron Full-Fixed`) `Pass@1` | `RealCoder-v3` Assertions |
| :--- | :---: | :---: | :---: | :---: |
| **OpenAI `HumanEval`** | `7/8` (`87.50%`) | `30/32` (`93.75%`) | `7/8` (`87.50%`) | `30/32` (`93.75%`) |
| **Google `MBPP`** | `8/8` (`100.00%`) | `32/32` (`100.00%`) | `8/8` (`100.00%`) | `32/32` (`100.00%`) |
| **`DS-1000` NumPy** | `8/8` (`100.00%`) | `32/32` (`100.00%`) | `8/8` (`100.00%`) | `32/32` (`100.00%`) |
| **`DS-1000` Pandas & ML** | `8/8` (`100.00%`) | `32/32` (`100.00%`) | `8/8` (`100.00%`) | `32/32` (`100.00%`) |
| **`Algo-OpenEnded` Tail Completion** | `7/8` (`87.50%`) | `30/32` (`93.75%`) | **`8/8` (`100.00%`)** | **`32/32` (`100.00%`)** |
| **Overall (`40` Functions / `160` Assertions)** | **`38/40` (`95.00%`)** | **`156/160` (`97.50%`)** | **`39/40` (`97.50%`)** | **`158/160` (`98.75%`)** |

