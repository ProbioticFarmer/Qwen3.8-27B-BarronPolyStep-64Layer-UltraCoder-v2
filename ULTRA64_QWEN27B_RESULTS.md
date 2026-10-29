# Ultra-Scale 64-Layer Coding Distillation Report: `Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2`

## 1. Exported 64-Layer Rank-256 Checkpoint & Scale Specifications
- **Exported Checkpoint Directory**: [`checkpoints/Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2`](file:///Users/joshua/projects/PolyStep/checkpoints/Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2) (**260 `bfloat16` LoRA tensors, `~680 MB`**, `r=256`, `lora_alpha=512`).
- **Full 64-Layer Hybrid Architecture Streamed**:
  - **All 48 Gated Linear-Attention / Mamba-2 SSM Layers** (`layers.{0,1,2, 4,5,6, ..., 60,61,62}`: `in_proj_qkv`, `in_proj_z`, `A_log`, `dt_bias`, `out_proj`, SwiGLU MLP)
  - **All 16 Gated Full-Attention Retrieval Layers** (`layers.{3, 7, 11, ..., 59, 63}`: `q_proj`, `v_proj`, `o_proj`, SwiGLU MLP)
  - **Multi-Token Prediction ($t+2$) Lookahead Block** (`mtp.fc`, `mtp.layers.0`, `mtp.norm`) + **`lm_head`**
  - Paired against **all 43 Layers (`layers.0..42`) of `DeepSeek-V4-Flash`** + live flagship **`deepseek-v4-pro`** API coding traces (`data/deepseek_v4_pro_coding_traces.json`).
- **4× Larger Vocabulary & 2× Polytope Subspace**: **`4,096` shared coding BPE vocabulary**, **`128`-D active bottleneck (`r=256` exported)**, **`max_subspace_dim=256` (`32` simplex particles × `9` vertices = `288` forward candidates/step)**.
- **32-Algorithm Live Python `compile() + exec()` Verifier**: **`8,192` compiled Python program variants** and **`32,768` live unit-test assertions**.

## 2. Complete 3-Seed Results (`mean ± std`, Seeds `42, 43, 44`)

| Precision | Optimizer Arm | 32-Task Function `Pass@1` (%) | 32,768 Assertions (%) | 4,096-Vocab Token Acc (%) | `DFlash2` Block Accept (%) | True `fp64` KL Div | Flat Probes (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bf16` | MeZO (SPSA) | 86.53 ± 1.68% | 91.81% | 95.97% | 90.24% | 0.0870 | 34.8% |
| `bf16` | Baseline | 85.56 ± 1.37% | 91.04% | 95.59% | 89.34% | 0.0928 | 77.2% |
| `bf16` | Fixed-Transport | 91.94 ± 2.08% | 95.17% | 97.71% | 94.38% | 0.0463 | 75.4% |
| `bf16` | Fixed-Loss | 90.00 ± 1.02% | 93.92% | 97.08% | 92.87% | 0.0547 | 51.8% |
| `bf16` | **Full-Fixed** | **97.64 ± 1.04%** | 98.54% | 99.41% | 98.53% | 0.0252 | 51.5% |
| `fp32` | MeZO (SPSA) | 86.67 ± 1.56% | 91.84% | 95.87% | 89.99% | 0.1796 | 0.0% |
| `fp32` | Baseline | 97.92 ± 1.48% | 98.85% | 99.44% | 98.62% | 0.0170 | 0.6% |
| `fp32` | Fixed-Transport | 98.75 ± 0.59% | 99.20% | 99.69% | 99.22% | 0.0125 | 32.4% |
| `fp32` | Fixed-Loss | 99.44 ± 0.20% | 99.76% | 99.86% | 99.65% | 0.0082 | 0.0% |
| `fp32` | **Full-Fixed** | **98.61 ± 0.86%** | 99.03% | 99.65% | 99.14% | 0.0080 | 0.0% |
