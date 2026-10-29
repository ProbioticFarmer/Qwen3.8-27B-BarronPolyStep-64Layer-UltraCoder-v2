# Full-Stack 64-Layer Coding Fine-Tuning Report: `Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1`

## 1. Exported Model Checkpoint & Full-Stack Architecture
- **Exported Checkpoint Path**: [`checkpoints/Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1`](file:///Users/joshua/projects/PolyStep/checkpoints/Qwen3.8-27B-BarronPolyStep-FullStack-Coder-v1) (`adapter_model.safetensors`, `adapter_config.json`, `stage_stream_manifest.json`).
- **Full-Stack Depth Coverage**: Streamed all **18 `.safetensors` shards (`55.6 GB` `bfloat16`) of `Qwen/Qwen3.8-27B`**, adapting all **16 Gated Full-Attention Layers (`layers.[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]`)**, the **Multi-Token Prediction ($t+2$) Lookahead Block (`mtp.fc`, `mtp.layers.0`, `mtp.norm`)**, and **`lm_head`** against all **16 depth-matched Teacher Layers (`layers.[2, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29, 32, 34, 37, 40, 42]`)** of `DeepSeek-V4-Flash`.
- **16-Algorithm Live Python `compile() + exec()` Verifier**: **4,096 compiled Python program variants** and **16,384 live unit-test assertions** across 16 algorithmic & systems bug-repair tasks (`barron_log_softmax_1d`, `welford_online_variance`, `bisect_left_range`, `fenwick_prefix_sum`, `trie_prefix_count`, `mod_inverse_fermat`, `unpack_int4_nibbles`, `popcount_kernighan`, `edit_distance_dp`, `kadane_circular_max`, `longest_increasing_subseq_bs`, `coin_change_min_dp`, `kahn_lex_toposort`, `dijkstra_shortest_path`, `union_find_connected`, `lru_ring_buffer_idx`).

## 2. Full-Stack 18-Stage Results (`mean ± std` over Seeds `42, 43, 44`)

| Precision | Optimizer Arm | 16-Task Function `Pass@1` (%) | Unit-Test Assertions (%) | Code Token Acc (%) | `DFlash2` Block Accept (%) | True `fp64` KL Div | Flat Probes (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bf16` | MeZO (SPSA) | 75.33 ± 1.09% | 82.33% | 92.50% | 82.29% | 0.2553 | 20.0% |
| `bf16` | Baseline | 76.44 ± 0.31% | 83.11% | 92.83% | 83.03% | 0.2350 | 60.2% |
| `bf16` | Fixed-Transport | 80.22 ± 1.37% | 85.56% | 93.94% | 85.54% | 0.1876 | 64.5% |
| `bf16` | Fixed-Loss | 85.11 ± 1.37% | 88.94% | 95.28% | 88.61% | 0.1271 | 31.1% |
| `bf16` | **Full-Fixed** | **86.44 ± 1.26%** | 90.39% | 96.06% | 90.43% | 0.0796 | 38.7% |
| `fp32` | MeZO (SPSA) | 72.67 ± 0.94% | 80.50% | 91.72% | 80.57% | 0.7350 | 0.0% |
| `fp32` | Baseline | 82.22 ± 1.37% | 87.17% | 94.44% | 86.69% | 0.4257 | 0.1% |
| `fp32` | Fixed-Transport | 83.33 ± 2.18% | 87.89% | 94.78% | 87.46% | 0.3631 | 37.3% |
| `fp32` | Fixed-Loss | 86.00 ± 0.54% | 89.94% | 95.78% | 89.78% | 0.2611 | 0.0% |
| `fp32` | **Full-Fixed** | **88.22 ± 2.74%** | 91.67% | 96.72% | 92.02% | 0.1766 | 0.0% |
