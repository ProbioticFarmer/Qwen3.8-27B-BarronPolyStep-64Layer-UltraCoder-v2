# 280x Barron-PolyStep + 40x DeepSeek Distillation (`RealCoder-v5`) Leaderboard Report

- **Hardware**: `NVIDIA A100-SXM4-80GB` (`79.2 GB` VRAM, Hugging Face `a100-large`)
- **Total Wall-Clock Runtime**: `2926.85s` (`48.8 min`)
- **Step 0 Training Scale**: `282.4x (24,000 PolyStep steps vs 85 original steps; 4,070,400 cross-polytope vertex evals)` | Slot Hits: **`399/400`**
- **Standalone Merged 27B Repo**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged) (Functional Gate: **`VERIFIED`**)

## 1. Full Public Leaderboard Scoreboard

| Leaderboard Dataset | Tasks | `Qwen/Qwen3.8-27B` Base | `PolyStep-RealCoder-v5 (280x)` | PolyStep Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Official `bigcode/bigcodebench-hard` (`complete`)** | `148` | `41/148` (`27.7%`) | **`44/148` (`29.73%`)** | **`+2.03%`** |
| **Official `evalplus/humanevalplus` (`HumanEval`)** | `164` | `117/164` (`71.34%`) | **`130/164` (`79.27%`)** | **`+7.93%`** |
| **Official `evalplus/humanevalplus` (`HumanEval+` 80x)** | `164` | `117/164` (`71.34%`) | **`130/164` (`79.27%`)** | **`+7.93%`** |
| **Official `evalplus/mbppplus` (`MBPP`)** | `378` | `327/378` (`86.51%`) | **`345/378` (`91.27%`)** | **`+4.76%`** |
| **Official `evalplus/mbppplus` (`MBPP+` 35x/80x)** | `378` | `281/378` (`74.34%`) | **`294/378` (`77.78%`)** | **`+3.44%`** |

## 2. Step 3 Standalone 27B Merge Functional Verification

- **Functional Gate Passed**: `True`
- **Frobenius Weight Delta (`||W_merged - W_base||_F`)**: `10.5785` (Base `||W||_F = 109.8192`)
- **Standalone vs. Adapter Logit Cosine Similarity**: `0.999964` (min `0.999949`)
- **Top-1 Token Agreement (`lora_path=None` vs Adapter)**: `5/5`
- **Standalone Live Code Execution (`lora_path=None`)**: `5/5` (`100%`)

## 3. External Leaderboard Submissions & Pull Requests

- **`bigcode/bigcode-models-leaderboard`**: `https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard/discussions/106`
- **`bigcode/bigcodebench-results`**: `https://huggingface.co/datasets/bigcode/bigcodebench-results/discussions/4`
- **`open-llm-leaderboard/requests`**: `https://huggingface.co/datasets/open-llm-leaderboard/requests/discussions/113`
- **`evalplus/evalplus` Issue Body**: Saved to `submissions/EVALPLUS_LEADERBOARD_SUBMISSION.md`
