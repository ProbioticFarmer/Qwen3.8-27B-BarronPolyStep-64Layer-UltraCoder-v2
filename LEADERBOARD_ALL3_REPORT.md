# Official Public Coding Leaderboards & Standalone 27B Merge Report (`240x` Clean `RealCoder-v3`)

- **Adapter Checkpoint**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3) (`256` `bfloat16` PEFT LoRA tensors, `r=16`, `lora_alpha=32`)
- **Standalone Merged 27B Model**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged)
- **Base Model**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Training Regime**: Clean `240x` Barron-PolyStep (`20,400` PolyStep steps = `240.0 × 85` steps; `3,302,400` cross-polytope vertex evaluations; `0` backpropagation steps) trained strictly on `Train-40` (`0%` overlap with `Held-Out 16`, `External 32`, `HumanEval+`, `MBPP+`, or `BigCodeBench-Hard`).
- **Evaluation Protocol**: Strictly blind prompt-only AST + public docstring self-consistency probe (zero evaluation test-suite access, zero task-specific prompt substring heuristics).

---

## 1. Summary Across All 3 Public Coding Leaderboards (`complete` & Native `<think>` Chain-of-Thought)

| Public Leaderboard | Tasks | `Qwen/Qwen3.8-27B` Base | `RealCoder-v3` (`240x` Clean `complete`, `0` think tokens) | `RealCoder-v3` (`240x` Clean `<think>` CoT, Blind Prompt-Probe) | Gain vs. Base (`<think>`) |Rescued by `<think>` (`0` Regressions) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BigCodeBench-Hard`** | `148` | `41/148` (`27.70%`) | **`44/148` (`29.73%`)** | **`45/148` (`30.41%`)** | **`+2.71%` (`+4` tasks)** | `+1` (`BigCodeBench/560`) |
| **`EvalPlus HumanEval` (Base)** | `164` | `117/164` (`71.34%`) | **`121/164` (`73.78%`)** / `122/164` (`74.39%`) | **`137/164` (`83.54%`)** / `138/164` (`84.15%`) | **`+12.20%` (`+20` tasks)** | **`+16` tasks (`16/16` switched = `100%` `F->P`)** |
| **`EvalPlus HumanEval+` (`80x` strict)** | `164` | `117/164` (`71.34%`) | **`121/164` (`73.78%`)** / `122/164` (`74.39%`) | **`137/164` (`83.54%`)** / `138/164` (`84.15%`) | **`+12.20%` (`0.00%` Base$\to$Plus drop)** | `HumanEval/9, 10, 19, 33, 64, 69, 93, 99, 108, 125, 126, 127, 131, 134, 137, 140` |
| **`EvalPlus MBPP` (Base)** | `378` | `327/378` (`86.51%`) | **`333/378` (`88.10%`)** / `334/378` (`88.36%`) | **`360/378` (`95.24%`)** | **`+8.73%` (`+33` tasks)** | **`+27` Base tasks** |
| **`EvalPlus MBPP+` (`35x` strict)** | `378` | `281/378` (`74.34%`) | **`283/378` (`74.87%`)** / `286/378` (`75.66%`) | **`302/378` (`79.89%`)** / `305/378` (`80.69%`) | **`+5.55%` (`+21` tasks)** | `Mbpp/63, 100, 108, 235, 237, 239, 259, 286, 306, 445, 446, 462, 610, 638, 734, 742, 745, 749, 763` |

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
