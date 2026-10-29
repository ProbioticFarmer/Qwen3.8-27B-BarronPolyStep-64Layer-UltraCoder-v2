# `Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2`

```text
Δz > 6.22  ==>  fl_bf16(1 + exp(-Δz)) == 1.0  ==>  ∇_polytope == 0
fix_1(loss)      := log1p(Σ_{j ≠ i*} exp(z_j - z_*))
fix_2(transport) := exp((c_* - c_i)/ε) / exp(log1p(Σ_{j ≠ i*} exp((c_* - c_j)/ε)))
```

Two softmax fixes and **PolyStep** (`288` vertices/step, $\nabla = \varnothing$).  
Recapitulated common coding fine-tunes and distilled **DeepSeek-V4.1-Flash** into **Qwen3.8-27B** (`48` linear-attn + `16` full-attn + `mtp.0` + `lm_head`).  
All on an **M4 Max**.

---

```text
model      : Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2
weights    : checkpoints/Qwen3.8-27B-BarronPolyStep-64Layer-UltraCoder-v2/adapter_model.safetensors
footprint  : 681.54 MB (260 bf16 tensors, r=256)
dead_probes: 77.2% -> 0.0%
kl_fp64    : 0.1306 -> 0.0150
pass@1_32  : 82.92% -> 98.61% (32,768 assertions)
pass@1_ds  : 82.08% -> 99.58% (24,576 assertions | 95% CI [98.75, 100.00])
backprop   : 0
hardware   : Apple M4 Max
```
