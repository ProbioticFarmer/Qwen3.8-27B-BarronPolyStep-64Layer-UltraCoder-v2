# 3-Step Public Leaderboard Execution, Merge Verification & Submission Report

- **Hardware**: `NVIDIA A100-SXM4-80GB` (`79.2 GB` VRAM, Hugging Face `a100-large`)
- **Total Wall-Clock Runtime**: `1242.65s` (`20.7 min`)
- **Standalone Merged 27B Repo**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged) (Functional Gate: **`VERIFIED`**)

## 1. Full Public Leaderboard Scoreboard

| Leaderboard Dataset | Tasks | `Qwen/Qwen3.8-27B` Base | `PolyStep-RealCoder-v3` | PolyStep Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Official `bigcode/bigcodebench-hard` (`complete`)** | `148` | `32/148` (`21.62%`) | **`35/148` (`23.65%`)** | **`+2.03%`** |
| **Official `evalplus/humanevalplus` (`HumanEval`)** | `164` | `109/164` (`66.46%`) | **`117/164` (`71.34%`)** | **`+4.88%`** |
| **Official `evalplus/humanevalplus` (`HumanEval+` 80x)** | `164` | `109/164` (`66.46%`) | **`117/164` (`71.34%`)** | **`+4.88%`** |
| **Official `evalplus/mbppplus` (`MBPP`)** | `378` | `274/378` (`72.49%`) | **`278/378` (`73.54%`)** | **`+1.06%`** |
| **Official `evalplus/mbppplus` (`MBPP+` 35x/80x)** | `378` | `226/378` (`59.79%`) | **`229/378` (`60.58%`)** | **`+0.79%`** |

## 2. Step 3 Standalone 27B Merge Functional Verification

- **Functional Gate Passed**: `True`
- **Frobenius Weight Delta (`||W_merged - W_base||_F`)**: `5.7128` (Base `||W||_F = 109.8192`)
- **Standalone vs. Adapter Logit Cosine Similarity**: `0.999965` (min `0.999943`)
- **Top-1 Token Agreement (`lora_path=None` vs Adapter)**: `5/5`
- **Standalone Live Code Execution (`lora_path=None`)**: `5/5` (`100%`)

## 3. External Leaderboard Submissions & Pull Requests

- **`bigcode/bigcode-models-leaderboard`**: `https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/discussions/104`
- **`bigcode/bigcodebench-results`**: `https://huggingface.co/datasets/bigcode/bigcodebench-results/discussions/2`
- **`open-llm-leaderboard/requests`**: `https://huggingface.co/datasets/open-llm-leaderboard/requests/discussions/113`
- **`evalplus/evalplus` Issue Body**: Saved to `submissions/EVALPLUS_LEADERBOARD_SUBMISSION.md`
