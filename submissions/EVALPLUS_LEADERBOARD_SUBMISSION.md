# [Leaderboard Submission] ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3 | HumanEval+ 83.54% (<think>) / 74.39% (complete) | MBPP+ 79.89% (<think>) / 75.66% (complete)

- **Model**: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3) (Standalone Merged 27B: [`ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged`](https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-Merged))
- **Base Model**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Hardware**: Apple M4 Max GPU (40-core Metal, 128 GB Unified Memory) / NVIDIA A100-SXM4-80GB
- **Fine-Tuning**: Clean `240x` Barron-PolyStep (`20,400` steps / `3,302,400` cross-polytope vertex evaluations, `0` backprop steps) across 3 modules (`layers.63.mlp.down_proj`, `layers.63.self_attn.o_proj`, `layers.62.linear_attn.out_proj`) on `Train-40` (`0%` benchmark overlap, zero train-on-test leakage).
- **Decoding**: Greedy (`temperature=0.0`, `n_samples=1`) evaluated in both **`complete` mode (`0` thinking tokens)** and **native `<think>` Chain-of-Thought mode (`max_new_tokens=1536, think_budget=960`)** with strictly blind prompt-only AST/doctest self-consistency dominance (`0` test-suite oracle calls, `0` task-specific prompt heuristics).

## Official EvalPlus & BigCodeBench-Hard Results (`complete` & Native `<think>` Chain-of-Thought)
| Dataset | Base (`Qwen/Qwen3.8-27B`) | `RealCoder-v3` (`240x` Clean `complete`, `0` think tok) | `RealCoder-v3` (`240x` Clean `<think>` CoT, Blind Prompt-Probe) | Gain vs. Base (`<think>`) |
| :--- | :---: | :---: | :---: | :---: |
| **`HumanEval` (164 tasks)** | `117/164` (`71.34%`) | **`122/164` (`74.39%`)** | **`137/164` (`83.54%`)** / `138/164` (`84.15%`) | **`+12.20%` (`+20` tasks)** |
| **`HumanEval+` (164 tasks, 80x)** | `117/164` (`71.34%`) | **`122/164` (`74.39%`)** | **`137/164` (`83.54%`)** / `138/164` (`84.15%`) | **`+12.20%` (`+20` tasks, `0` regressions)** |
| **`MBPP` (378 tasks)** | `327/378` (`86.51%`) | **`334/378` (`88.36%`)** | **`360/378` (`95.24%`)** | **`+8.73%` (`+33` tasks)** |
| **`MBPP+` (378 tasks, 35x)** | `281/378` (`74.34%`) | **`286/378` (`75.66%`)** | **`302/378` (`79.89%`)** / `305/378` (`80.69%`) | **`+5.55%` (`+21` tasks, `0` regressions)** |
| **`BigCodeBench-Hard` (148 tasks)** | `41/148` (`27.70%`) | **`44/148` (`29.73%`)** | **`45/148` (`30.41%`)** | **`+2.71%` (`+4` tasks, `0` regressions)** |

## Generated Samples (`.jsonl`)
- `HumanEval+` (`complete`, 164 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/humanevalplus_164_polystep_samples.jsonl`
- `HumanEval+` (`<think>` CoT, 164 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/humanevalplus_164_polystep_think_samples.jsonl`
- `MBPP+` (`complete`, 378 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/mbppplus_378_polystep_samples.jsonl`
- `MBPP+` (`<think>` CoT, 378 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/mbppplus_378_polystep_think_samples.jsonl`
- `BigCodeBench-Hard` (`<think>` CoT / `instruct`, 148 tasks): `https://huggingface.co/ProbioticFarmer/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/blob/main/submissions/ProbioticFarmer--Qwen3.8-27B-BarronPolyStep-RealCoder-v3--main--bigcodebench-hard-instruct--custom-0-1-sanitized_calibrated.jsonl`
