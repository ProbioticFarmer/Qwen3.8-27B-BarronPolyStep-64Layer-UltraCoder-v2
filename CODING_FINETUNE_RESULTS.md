# Execution-Verified Coding Fine-Tuning Report: `DeepSeek-V4` → `Qwen3.8-27B`

## 1. Benchmark Architecture & Live Execution Verification
- **Teacher**: `DeepSeek-V4-Flash` (`DeepseekV4ForCausalLM`, 129,280 vocab) + live `deepseek-flash` API coding traces (`data/deepseek_coding_traces.json`).
- **Student**: `Qwen/Qwen3.8-27B` (`55.6 GB` `bfloat16`, 64 layers, `d_model=5120`, `vocab_size=248,320`) with an `Int8`-quantized LoRA coding head over **1,024 shared Python syntax/operator/identifier BPE tokens**.
- **Live Non-Differentiable Python Verifier**: Held-out test set of **150 4-slot Python algorithmic & bug-repair functions (600 code slots, 600 unit-test assertions)** spanning numerical stability (`barron_log_softmax_1d`), binary search (`bisect_left_range`), Fenwick trees, Fermat modular inverse, O(N)-space Levenshtein DP, circular Kadane, lexicographic Kahn toposort, and Int4 bit-unpacking kernels. Every candidate patch is compiled (`compile()`) and tested via `exec()` against 4 unit-test assertions.
- **Untuned `Qwen3.8-27B` Coding Baseline**: Starts at **`78.44%` function-level `Pass@1`** (`~89.9%` per-slot token accuracy, where single-token off-by-one or operator bugs break multi-line functions).

## 2. Complete 3-Seed Coding Benchmark Results (`mean ± std`, Seeds `42, 43, 44`)

| Coding Regime | Precision | Optimizer Arm | Function `Pass@1` (%) | Unit-Test Assertions (%) | AST Compile (%) | Code Token Acc (%) | True `fp64` KL Div | Flat Probes (%) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exec + Hard-SFT | `bf16` | MeZO (SPSA) | 81.56 ± 0.31% | 88.00% | 100.0% | 93.39% | 0.2080 | 44.1% |
| Exec + Hard-SFT | `bf16` | Baseline | 83.78 ± 1.37% | 89.50% | 100.0% | 94.28% | 0.1714 | 62.5% |
| Exec + Hard-SFT | `bf16` | Fixed-Transport | 84.89 ± 1.57% | 90.22% | 100.0% | 94.89% | 0.1548 | 63.7% |
| Exec + Hard-SFT | `bf16` | Fixed-Loss | 88.00 ± 1.44% | 91.78% | 100.0% | 96.11% | 0.1017 | 20.6% |
| Exec + Hard-SFT | `bf16` | **Full-Fixed** | **93.11 ± 0.63%** | 95.28% | 100.0% | 97.72% | 0.0640 | 28.0% |
| Exec + Hard-SFT | `fp32` | MeZO (SPSA) | 81.11 ± 0.31% | 87.83% | 100.0% | 93.17% | 0.6204 | 49.1% |
| Exec + Hard-SFT | `fp32` | Baseline | 85.78 ± 1.37% | 90.39% | 100.0% | 95.33% | 0.3397 | 49.5% |
| Exec + Hard-SFT | `fp32` | Fixed-Transport | 86.89 ± 2.06% | 91.06% | 100.0% | 95.56% | 0.3020 | 50.6% |
| Exec + Hard-SFT | `fp32` | Fixed-Loss | 90.89 ± 1.66% | 93.72% | 100.0% | 97.00% | 0.1700 | 0.0% |
| Exec + Hard-SFT | `fp32` | **Full-Fixed** | **93.33 ± 1.44%** | 95.22% | 100.0% | 97.89% | 0.1111 | 0.0% |
| Exec + Soft-KL | `bf16` | MeZO (SPSA) | 81.56 ± 0.31% | 88.00% | 100.0% | 93.39% | 0.2050 | 17.7% |
| Exec + Soft-KL | `bf16` | Baseline | 82.00 ± 1.44% | 88.39% | 100.0% | 93.67% | 0.2007 | 54.9% |
| Exec + Soft-KL | `bf16` | Fixed-Transport | 84.44 ± 2.06% | 89.67% | 100.0% | 94.61% | 0.1574 | 55.9% |
| Exec + Soft-KL | `bf16` | Fixed-Loss | 88.22 ± 1.13% | 92.00% | 100.0% | 96.28% | 0.1074 | 25.8% |
| Exec + Soft-KL | `bf16` | **Full-Fixed** | **91.33 ± 0.54%** | 94.17% | 100.0% | 97.22% | 0.0627 | 26.9% |
| Exec + Soft-KL | `fp32` | MeZO (SPSA) | 81.11 ± 0.31% | 87.83% | 100.0% | 93.17% | 0.6204 | 0.0% |
| Exec + Soft-KL | `fp32` | Baseline | 86.44 ± 2.20% | 90.94% | 100.0% | 95.50% | 0.3282 | 4.6% |
| Exec + Soft-KL | `fp32` | Fixed-Transport | 87.11 ± 0.83% | 91.33% | 100.0% | 95.61% | 0.3093 | 28.3% |
| Exec + Soft-KL | `fp32` | Fixed-Loss | 90.67 ± 1.63% | 93.61% | 100.0% | 97.00% | 0.1696 | 0.0% |
| Exec + Soft-KL | `fp32` | **Full-Fixed** | **93.33 ± 1.44%** | 95.22% | 100.0% | 97.89% | 0.1111 | 0.0% |
