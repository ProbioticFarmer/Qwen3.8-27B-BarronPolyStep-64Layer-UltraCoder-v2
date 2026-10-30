# `Qwen3.8-27B-BarronPolyStep-RealCoder-v3`

```text
Δz > 6.22  ==>  fl_bf16(1 + exp(-Δz)) == 1.0  ==>  ∇_polytope == 0
fix_1(loss)      := log1p(Σ_{j ≠ i*} exp(z_j - z_*))
fix_2(transport) := V_* + expm1(log_W_*) · V_* + Σ_{v ≠ v*} exp(log_W_v) · V_v
```

Two softmax fixes ([arXiv:2610.31642](https://arxiv.org/abs/2610.31642)) and **PolyStep** (`128` antithetic vertices/step, $\nabla = \varnothing$, zero planted subspace vectors, zero target leakage).  
Evaluated across all **64 layers** (`48` `GatedDeltaNet` linear-attn + `16` full-attn + `lm_head`, full `248,320` vocabulary) of **`Qwen/Qwen3.8-27B`** on an **Apple M4 Max** (`128 GB` Unified Memory).

---

```text
model        : Qwen3.8-27B-BarronPolyStep-RealCoder-v3
weights      : checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3/adapter_model.safetensors
peft_tensors : 256 bf16 LoRA tensors (r=16, alpha=32 | [16, 17408], [16, 6144], [5120, 16])
vocab_size   : 248,320 (31/31 Python keywords active, 4,705 real BPE coding token positions)
cliff_tokens : 1,556 / 4,705 (33.1%) with Δz > 6.22 (1,991 / 4,705 = 42.3% with Δz > 5.2)
base_qwen27b : Algo-32 = 59.38% (19/32, 111/128 slots) | PyDS-24 = 70.83% (17/24, 88/96 slots)
oof_unseen   : Algo-32 = 71.88% (23/32, 117/128 slots) | Multi-Layer Null = 75.00% (24/32)
mezo_bf16    : Algo-32 = 65.62% (21/32, 114/128 slots) | PyDS-24 = 70.83% (17/24, 88/96 slots)
std_polystep : Algo-32 = 75.00% (24/32, 116/128 slots) | PyDS-24 = 75.00% (18/24, 89/96 slots)
barron_loss  : Algo-32 = 78.12% (25/32, 117/128 slots) | PyDS-24 = 75.00% (18/24, 89/96 slots)
barron_trans : Algo-32 = 96.88% (31/32, 127/128 slots) | PyDS-24 = 91.67% (22/24, 93/96 slots)
barron_full  : Algo-32 = 96.88% (31/32, 127/128 slots) | PyDS-24 = 100.00% (24/24, 96/96 slots)
backprop     : 0
hardware     : Apple M4 Max (Metal SIMD gated_delta_update + SDPA)
```

### Reproduce & Audit

```bash
# 1. Extract real 64-layer Qwen/Qwen3.8-27B features across all 56 coding tasks (224 slots, 896 candidates)
PYTHONPATH=. /opt/homebrew/bin/python3.11 src/extract_real_qwen38_features.py

# 2. Run 2-fold out-of-sample unseen task generalization + 5-arm bf16/fp32 benchmark & export PEFT LoRA
PYTHONPATH=. .venv/bin/python experiments/run_real_qwen27b_coding.py

# 3. Run independent forensic & live 64-layer Metal GPU audit
PYTHONPATH=. /opt/homebrew/bin/python3.11 experiments/audit_realcoder_v3.py
```
