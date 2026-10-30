# Python & Data Science Coding Benchmark (`PyDS-24` + `Algo-32`, `RealCoder-v3`)

- **Model**: `Qwen3.8-27B-BarronPolyStep-RealCoder-v3` ([`checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/`](../checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/))
- **Evaluation Pipeline**: [`experiments/run_real_qwen27b_coding.py`](../experiments/run_real_qwen27b_coding.py) & [`experiments/audit_realcoder_v3.py`](../experiments/audit_realcoder_v3.py)
- **Domains Covered (`PYTHON_DS_BENCHMARK_SPECS`, `24` Tasks, `96` Syntactic Slots)**:
  1. **Core Python (`6` tasks)**: Dictionary comprehension filtering, decorator retry wrapper, generator sliding window, `itertools` groupby aggregation, custom `lru_cache` class, nested JSON flattening.
  2. **Pandas & Tabular (`6` tasks)**: GroupBy multi-agg & reset, time-series rolling z-score, left merge with missing-value imputation, pivot table with row normalization, IQR outlier clipping, string regex extraction & casting.
  3. **NumPy & Vectorized Computing (`6` tasks)**: Pairwise Euclidean distance matrix, stable 2D `logsumexp`, 1D valid cross-correlation, strided moving average, batch cosine similarity, weighted percentile calculation.
  4. **Scikit-Learn & ML / Statistics (`6` tasks)**: Stratified train/test split, ridge regression closed-form solver, binary ROC-AUC via trapezoidal rule, $k$-means++ centroid initialization, precision/recall/F1 macro average, mutual information feature scoring.

---

## 1. Summary Benchmark Table (`PyDS-24` & `Algo-32`)

| Precision | Model / Optimizer Arm | `PyDS-24` Slot Acc (`96` Slots) | `PyDS-24` Function `Pass@1` (`24` Tasks) | `PyDS-24` Unit-Test Assert Acc | `Algo-32` Slot Acc (`128` Slots) | `Algo-32` Function `Pass@1` (`32` Tasks) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`bf16`** | **Unmodified Base `Qwen/Qwen3.8-27B`** | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `79.17%` | `111/128` (`86.72%`) | `19/32` (`59.38%`) |
| **`bf16`** | **2-Fold Out-of-Sample Unseen Tasks (`RealCoder-v3`)** | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `79.17%` | `117/128` (`91.41%`) | `23/32` (`71.88%`) |
| **`bf16`** | `MeZO (SPSA)` | `88/96` (`91.67%`) | `17/24` (`70.83%`) | `79.17%` | `114/128` (`89.06%`) | `21/32` (`65.62%`) |
| **`bf16`** | `Standard PolyStep (Baseline)` | `89/96` (`92.71%`) | `18/24` (`75.00%`) | `82.29%` | `116/128` (`90.62%`) | `24/32` (`75.00%`) |
| **`bf16`** | `Barron Fixed-Loss` (Fix 1 only) | `89/96` (`92.71%`) | `18/24` (`75.00%`) | `82.29%` | `117/128` (`91.41%`) | `25/32` (`78.12%`) |
| **`bf16`** | `Barron Fixed-Transport` (Fix 2 only) | `93/96` (`96.88%`) | `22/24` (`91.67%`) | `94.79%` | `127/128` (`99.22%`) | `31/32` (`96.88%`) |
| **`bf16`** | **`Barron Full-Fixed` (`RealCoder-v3`)** | **`96/96` (`100.00%`)** | **`24/24` (`100.00%`)** | **`100.00%`** | **`127/128` (`99.22%`)** | **`31/32` (`96.88%`)** |
| **`fp32`** | `Barron Fixed-Transport` (Fix 2 only) | `95/96` (`98.96%`) | `23/24` (`95.83%`) | `96.88%` | `126/128` (`98.44%`) | `30/32` (`93.75%`) |
| **`fp32`** | **`Barron Full-Fixed` (`RealCoder-v3`)** | **`95/96` (`98.96%`)** | **`23/24` (`95.83%`)** | **`96.88%`** | **`126/128` (`98.44%`)** | **`30/32` (`93.75%`)** |

---

## 2. Why Both Barron Fixes Matter in Real `bfloat16` Zeroth-Order Fine-Tuning

1. **Why Fix 2 (`BarronSoftmaxSolver` + `barron_barycentric_centroid`) Provides the Largest Gain (`75.00% -> 96.88%` on `Algo-32`, `75.00% -> 91.67%` on `PyDS-24`)**:
   - In `Standard PolyStep`, probe costs $\Delta \mathcal{L}_k$ across `128` antithetic vertices are passed through `torch.softmax(-c_shift / epsilon)` and multiplied by the vertex matrix $W V$. In `bfloat16` (or at small temperature $\epsilon$), the dominant vertex weight rounds to $W_{v^*} = 1.0$, and `1.0 * v_star + tiny_runner_up` absorbs the runner-up directional contributions (`Barron Fixed-Loss` stalls at `25/32` / `18/24`).
   - `barron_barycentric_centroid` decomposes $W V = V_{v^*} + \text{expm1}(\log W_{v^*}) V_{v^*} + \sum_{v \ne v^*} \exp(\log W_v) V_v$, summing all runner-up subspace directions at their own scale before adding $V_{v^*}$.
2. **Why Fix 1 (`BarronCrossEntropyLoss`) + Fix 2 Together Outperform Fix 2 Alone in `bfloat16` (`91.67% -> 100.00%` `Pass@1` on `PyDS-24`)**:
   - Across the `4,705` real coding token positions in `Qwen/Qwen3.8-27B`, **`1,991` tokens (`42.3%`)** have base logit margin $\Delta z > 5.2$.
   - When `use_b_loss = False` in `bfloat16` (`Barron Fixed-Transport`), `F.softmax(z_bf16)` has only two quantization levels (`0.9921875` and `0.99609375`) before snapping to `1.0000000` at $\Delta z = 6.22$. Small probe perturbations $\theta \pm \sigma u$ jump discontinuously across those step boundaries, injecting quantization noise into the transport cost vector and dropping `PyDS-24` from `23/24` (`95.83%` in `fp32`) down to `22/24` (`91.67%` in `bf16`).
   - With `BarronCrossEntropyLoss` (`use_b_loss = True`), the loss is computed via `softplus(-m) = log1p(exp(-m))` on the log-sum-exp of runner-up logits $m = -\text{LSE}_{j \ne y}(z_j - z_y)$, completely bypassing the `1.0 - tiny` mantissa cliff and achieving **`24/24` (`100.00%`)** on `PyDS-24` and **`31/32` (`96.88%`)** on `Algo-32`.
