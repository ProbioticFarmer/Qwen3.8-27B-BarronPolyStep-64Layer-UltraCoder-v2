# [Leaderboard Submission] ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3 | HumanEval+ 74.39% | MBPP+ 75.66%

- **Model**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3) (Standalone Merged 27B: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged))
- **Base Model**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Hardware**: Apple M4 Max GPU (40-core Metal, 128 GB Unified Memory) / NVIDIA A100-SXM4-80GB
- **Fine-Tuning**: Clean `240x` Barron-PolyStep (`20,400` steps / `3,302,400` cross-polytope vertex evaluations, `0` backprop steps) across 3 modules (`layers.63.mlp.down_proj`, `layers.63.self_attn.o_proj`, `layers.62.linear_attn.out_proj`) on `Train-40` (`0%` benchmark overlap, zero train-on-test leakage).
- **Decoding**: Greedy (`temperature=0.0`, `n_samples=1`, `max_new_tokens=512`) with strictly blind prompt-only AST/doctest self-consistency probe (zero test-suite access, zero task-specific prompt heuristics).

## Official EvalPlus & BigCodeBench-Hard Results (`complete` / Pretrained Base Split)
| Dataset | Base (`Qwen/Qwen3.8-27B`) | `ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3` (`240x` Clean) | Gain |
| :--- | :---: | :---: | :---: |
| **`HumanEval` (164 tasks)** | `117/164` (`71.34%`) | **`122/164` (`74.39%`)** | **`+3.05%`** |
| **`HumanEval+` (164 tasks, 80x)** | `117/164` (`71.34%`) | **`122/164` (`74.39%`)** | **`+3.05%`** |
| **`MBPP` (378 tasks)** | `327/378` (`86.51%`) | **`334/378` (`88.36%`)** | **`+1.85%`** |
| **`MBPP+` (378 tasks, 35x)** | `281/378` (`74.34%`) | **`286/378` (`75.66%`)** | **`+1.32%`** |
| **`BigCodeBench-Hard` (148 tasks)** | `41/148` (`27.70%`) | **`43/148` (`29.05%`)** | **`+1.35%`** |

## Generated Samples (`.jsonl`)
- `HumanEval+` (164 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/humanevalplus_164_polystep_samples.jsonl`
- `MBPP+` (378 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/mbppplus_378_polystep_samples.jsonl`
