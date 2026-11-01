# 3-Step Public Leaderboard Execution, Merge Verification & Submission Report

- **Hardware**: `NVIDIA A100-SXM4-80GB` (`79.2 GB` VRAM, Hugging Face `a100-large`)
- **Total Wall-Clock Runtime**: `2244.64s` (`37.4 min`)
- **Standalone Merged 27B Repo**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged) (Functional Gate: **`VERIFIED`**)

## 1. Full Public Leaderboard Scoreboard

| Leaderboard Dataset | Tasks | `Qwen/Qwen3.8-27B` Base | `PolyStep-RealCoder-v3` | PolyStep Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Official `bigcode/bigcodebench-hard` (`complete`)** | `148` | `40/148` (`27.03%`) | **`43/148` (`29.05%`)** | **`+2.03%`** |
| **Official `evalplus/humanevalplus` (`HumanEval`)** | `164` | `125/164` (`76.22%`) | **`127/164` (`77.44%`)** | **`+1.22%`** |
| **Official `evalplus/humanevalplus` (`HumanEval+` 80x)** | `164` | `112/164` (`68.29%`) | **`120/164` (`73.17%`)** | **`+4.88%`** |
| **Official `evalplus/mbppplus` (`MBPP`)** | `378` | `278/378` (`73.54%`) | **`283/378` (`74.87%`)** | **`+1.32%`** |
| **Official `evalplus/mbppplus` (`MBPP+` 35x/80x)** | `378` | `230/378` (`60.85%`) | **`234/378` (`61.9%`)** | **`+1.06%`** |

## 2. Step 3 Standalone 27B Merge Functional Verification

- **Functional Gate Passed**: `True`
- **Frobenius Weight Delta (`||W_merged - W_base||_F`)**: `5.7128` (Base `||W||_F = 109.8192`)
- **Standalone vs. Adapter Logit Cosine Similarity**: `0.999965` (min `0.999943`)
- **Top-1 Token Agreement (`lora_path=None` vs Adapter)**: `5/5`
- **Standalone Live Code Execution (`lora_path=None`)**: `5/5` (`100%`)

## 3. External Leaderboard Submissions & Pull Requests

- **`bigcode/bigcode-models-leaderboard`**: `https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/discussions/105`
- **`bigcode/bigcodebench-results`**: `https://huggingface.co/datasets/bigcode/bigcodebench-results/discussions/3`
- **`open-llm-leaderboard/requests`**: `https://huggingface.co/datasets/open-llm-leaderboard/requests/discussions/113`
- **`evalplus/evalplus` Issue Body**: Saved to `submissions/EVALPLUS_LEADERBOARD_SUBMISSION.md`
