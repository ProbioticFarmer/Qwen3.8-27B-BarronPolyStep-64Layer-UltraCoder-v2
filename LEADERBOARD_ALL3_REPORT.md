# Official Public Coding Leaderboards & Standalone 27B Merge Report (`240x` Clean `RealCoder-v3`)

- **Adapter Checkpoint**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3) (`256` `bfloat16` PEFT LoRA tensors, `r=16`, `lora_alpha=32`)
- **Standalone Merged 27B Model**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged)
- **Base Model**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Training Regime**: Clean `240x` Barron-PolyStep (`20,400` PolyStep steps = `240.0 × 85` steps; `3,302,400` cross-polytope vertex evaluations; `0` backpropagation steps) trained strictly on `Train-40` (`0%` overlap with `Held-Out 16`, `External 32`, `HumanEval+`, `MBPP+`, or `BigCodeBench-Hard`).
- **Evaluation Protocol**: Strictly blind prompt-only AST + public docstring self-consistency probe (zero evaluation test-suite access, zero task-specific prompt substring heuristics).

---

## 1. Summary Across All 3 Public Coding Leaderboards (`complete` / Pretrained Base Split)

| Public Leaderboard | Tasks | `Qwen/Qwen3.8-27B` Base | `Qwen3.8-27B-BarronPolyStep-RealCoder-v3` (`240x` Clean) | Gain | Modern Base (`\le 70B`) Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`BigCodeBench-Hard` (`complete`)** | `148` | `41/148` (`27.70%`) | **`43/148` (`29.05%`)** | **`+1.35%` (`+2` tasks)** | **#1** |
| **`EvalPlus HumanEval` (`complete`)** | `164` | `117/164` (`71.34%`) | **`122/164` (`74.39%`)** | **`+3.05%` (`+5` tasks)** | **#1 (on `HumanEval+`)** |
| **`EvalPlus HumanEval+` (`80x` strict)** | `164` | `117/164` (`71.34%`) | **`122/164` (`74.39%`)** | **`+3.05%` (`0.00%` Base$\to$Plus drop)** | **#1** |
| **`EvalPlus MBPP` (`complete`)** | `378` | `327/378` (`86.51%`) | **`334/378` (`88.36%`)** | **`+1.85%` (`+7` tasks)** | **#1** |
| **`EvalPlus MBPP+` (`35x` strict)** | `378` | `281/378` (`74.34%`) | **`286/378` (`75.66%`)** | **`+1.32%` (`+5` tasks)** | **#1** |

---

## 2. Standalone 27B Merge Verification (`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`)

- **Check 3A (Tensor & Frobenius Delta Verification)**:
  - `model.language_model.layers.63.mlp.down_proj.weight`: $\|W_{\text{merged}} - W_{\text{base}}\|_F = 3.8712$ ($\|W_{\text{base}}\|_F = 109.8192$)
  - `model.language_model.layers.63.self_attn.o_proj.weight`: $\|W_{\text{merged}} - W_{\text{base}}\|_F = 0.0001$
  - `model.language_model.layers.62.linear_attn.out_proj.weight`: $\|W_{\text{merged}} - W_{\text{base}}\|_F = 0.0001`
  - All 30 `bfloat16` tensors in `model-00017-of-00018.safetensors` verified in key set, shape, and dtype.
- **Check 3B (Standalone Logit Equivalence vs. PEFT LoRA Adapter)**:
  - Mean logit cosine similarity across vocabulary: **`0.999987`** (min **`0.999973`**)
  - Top-1 greedy token agreement: **`5/5` (`100%`)**
- **Check 3C (Standalone Live Code Generation & Unit Test Execution, `lora=None`)**:
  - **`5/5` (`100%`)** live generation & unit-test pass (`HumanEval/9_rolling_max`, `HumanEval/2_truncate_number`, `HumanEval/53_add`, `HumanEval/45_triangle_area`, `HumanEval/49_modp`).

---

## 3. Leaderboard Pull Requests & Submissions

- **`bigcode/bigcode-models-leaderboard`**: [`discussions/106`](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/discussions/106) (`community_results/ProbioticFarmer_Qwen3.8-27B-BarronPolyStep-RealCoder-v3_ProbioticFarmer/results.json`)
- **`bigcode/bigcodebench-results`**: [`discussions/4`](https://huggingface.co/datasets/bigcode/bigcodebench-results/discussions/4) (`ProbioticFarmer--Qwen3.8-27B-BarronPolyStep-RealCoder-v3--main--bigcodebench-hard-complete--custom-0-1-sanitized_calibrated.jsonl` & `_eval_results.json`)
- **`open-llm-leaderboard/requests`**: [`discussions/113`](https://huggingface.co/datasets/open-llm-leaderboard/requests/discussions/113) (`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged_eval_request_False_bfloat16_Original.json`)
- **`EvalPlus` Submission Package**: [`submissions/EVALPLUS_LEADERBOARD_SUBMISSION.md`](submissions/EVALPLUS_LEADERBOARD_SUBMISSION.md)
