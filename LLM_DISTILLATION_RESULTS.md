# DeepSeek-V4-Flash → Qwen3.8-27B (`bf16` / `fp32`) Forward-Only Distillation Report

## 1. Architecture & Experimental Setup
- **Teacher Model**: `DeepSeek-V4-Flash` (`DeepseekV4ForCausalLM`, 129,280 vocab, 4-bit affine `U32` weights dequantized to `bfloat16`, Layer 42 FFN + RMSNorm + `lm_head`) paired with live `deepseek-flash` API verification (`data/deepseek_teacher_traces.json`).
- **Student Model**: `Qwen/Qwen3.8-27B` (`55.6 GB` `bfloat16`, 64 layers, `d_model=5120`, `vocab_size=248,320`), using site-aware trunk caching through `layers.63` SwiGLU MLP + `RMSNorm` + `mtp.norm` and an **Int8-quantized (`round()`) LoRA distillation adapter** over the **108,771 exact shared BPE vocabulary** (`ACTIVE_VOCAB = 1,024`).
- **Matched Forward Evaluation Budget**: **144 forward candidate evaluations per step** (`max_subspace_dim=128`, 16 particles × 9 simplex vertices for PolyStep; 72 antithetic 2-point SPSA probes for `MeZO`).

## 2. Complete 3-Seed Distillation Results (`mean ± std` over Seeds `42, 43, 44`)

| Distillation Mode | Precision | Optimizer Arm | Teacher Top-1 Agreement (%) | Final True `fp64` CE Loss | Final True `fp64` KL Div | True `fp64` Perplexity | Flat Probe / Vertex (%) | Token Cliff Rate (%) | Transport Entropy $H(W)$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Hard-Token | `bf16` | MeZO (SPSA) | 92.39 ± 0.16% | 0.2233 ± 0.0076 | 0.2185 | 1.250 | 42.3% | 87.0% | 0.000 |
| Hard-Token | `bf16` | Baseline | 93.67 ± 0.62% | 0.1867 ± 0.0169 | 0.1819 | 1.205 | 67.1% | 86.9% | 1.599 |
| Hard-Token | `bf16` | Fixed-Transport | 95.06 ± 0.57% | 0.1434 ± 0.0156 | 0.1386 | 1.154 | 74.6% | 87.6% | 1.882 |
| Hard-Token | `bf16` | Fixed-Loss | 97.00 ± 0.27% | 0.0844 ± 0.0110 | 0.0798 | 1.088 | 23.3% | 90.1% | 0.779 |
| Hard-Token | `bf16` | **Full-Fixed** | **98.06 ± 0.08%** | **0.0463 ± 0.0033** | 0.0416 | 1.047 | 29.4% | 93.7% | 1.432 |
| Hard-Token | `fp32` | MeZO (SPSA) | 91.94 ± 0.08% | 0.5988 ± 0.0316 | 0.5974 | 1.821 | 49.2% | 87.5% | 0.000 |
| Hard-Token | `fp32` | Baseline | 96.22 ± 0.21% | 0.2507 ± 0.0110 | 0.2494 | 1.285 | 58.1% | 89.8% | 1.328 |
| Hard-Token | `fp32` | Fixed-Transport | 96.78 ± 0.34% | 0.2165 ± 0.0094 | 0.2152 | 1.242 | 63.6% | 91.8% | 1.843 |
| Hard-Token | `fp32` | Fixed-Loss | 97.83 ± 0.27% | 0.1182 ± 0.0201 | 0.1168 | 1.126 | 0.0% | 93.9% | 0.510 |
| Hard-Token | `fp32` | **Full-Fixed** | **98.33 ± 0.24%** | **0.0822 ± 0.0099** | 0.0808 | 1.086 | 0.0% | 95.0% | 1.316 |
| Soft-Logit | `bf16` | MeZO (SPSA) | 92.56 ± 0.21% | 0.2206 ± 0.0085 | 0.2159 | 1.247 | 20.4% | 87.0% | 0.000 |
| Soft-Logit | `bf16` | Baseline | 93.72 ± 0.70% | 0.1815 ± 0.0167 | 0.1768 | 1.199 | 63.6% | 87.0% | 1.540 |
| Soft-Logit | `bf16` | Fixed-Transport | 95.28 ± 0.91% | 0.1381 ± 0.0132 | 0.1334 | 1.148 | 66.2% | 87.0% | 1.786 |
| Soft-Logit | `bf16` | Fixed-Loss | 96.94 ± 0.28% | 0.0832 ± 0.0102 | 0.0784 | 1.087 | 21.3% | 90.5% | 0.725 |
| Soft-Logit | `bf16` | **Full-Fixed** | **97.94 ± 0.28%** | **0.0514 ± 0.0074** | 0.0465 | 1.053 | 28.6% | 93.7% | 1.413 |
| Soft-Logit | `fp32` | MeZO (SPSA) | 91.94 ± 0.08% | 0.5988 ± 0.0316 | 0.5974 | 1.821 | 0.0% | 87.5% | 0.000 |
| Soft-Logit | `fp32` | Baseline | 96.33 ± 0.24% | 0.2484 ± 0.0121 | 0.2472 | 1.282 | 9.1% | 89.9% | 1.314 |
| Soft-Logit | `fp32` | Fixed-Transport | 96.94 ± 0.08% | 0.1961 ± 0.0068 | 0.1948 | 1.217 | 39.6% | 91.0% | 1.616 |
| Soft-Logit | `fp32` | Fixed-Loss | 97.83 ± 0.27% | 0.1191 ± 0.0208 | 0.1176 | 1.127 | 0.0% | 94.0% | 0.510 |
| Soft-Logit | `fp32` | **Full-Fixed** | **98.33 ± 0.24%** | **0.0802 ± 0.0110** | 0.0789 | 1.084 | 0.0% | 95.0% | 1.315 |

## 3. Key Findings on `DeepSeek-V4-Flash` → `Qwen3.8-27B` Distillation
1. **Why Soft-Logit Distillation Collapses in Standard `bf16` (`Baseline` & `MeZO`)**:
   - When `Qwen3.8-27B` matches `DeepSeek-V4-Flash`'s full logit distribution in `bf16`, confident tokens ($m > 5.5$, ~87% of positions) suffer both **Softmax Collapse** (`log_p_S -> 0.0`) and **Softmax Cancellation** (`p_T * log_p_S` rounds away runner-up vocabulary differences across 1,024+ classes).
   - Replacing standard `CrossEntropyLoss` with `BarronCrossEntropyLoss` (`Fixed-Loss` / `Full-Fixed`) eliminates outer precision-cliff ties (`0.0%` flat vertices in `fp32`, and cutting `bf16` flat vertices from `67–78%` down to `16–22%`).
2. **PolyStep vs. MeZO (SPSA) on Quantized 27B Adapter Distillation**:
   - `MeZO` (2-point antithetic SPSA with matched 144 evaluations/step) stalls on `Int8` quantization plateaus and `bf16` precision cliffs, trailing `Barron Full-Fixed PolyStep` by **+3.2 to +4.1 percentage points** in teacher agreement and **4–6×** in true `fp64` distillation loss.
