"""Apple Silicon M4 Max Metal GPU Engine for `Qwen/Qwen3.8-27B` (`55.6 GB` `bfloat16`, All 64 Layers).

Uses zero-copy Unified Memory mapping (`mx.load`) and custom Metal GPU kernels:
- `mlx_lm.models.gated_delta.gated_delta_update` (Metal SIMD kernel for 48 GatedDeltaNet layers)
- `mx.fast.scaled_dot_product_attention` (Metal FlashAttention for 16 Gated Full-Attention layers)
- `mx.fast.rms_norm` & `mx.fast.rope` (Metal fused RMSNorm and partial RoPE)
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from tokenizers import Tokenizer


def get_qwen38_snapshot() -> str:
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    return sorted(glob.glob(os.path.join(hub, "models--Qwen--Qwen3.8-27B", "snapshots", "*")))[0]


class MLXQwen38Engine:
    """Zero-copy M4 Max Metal GPU engine for full 64-layer `Qwen/Qwen3.8-27B`."""

    def __init__(self):
        t0 = time.time()
        self.snap = get_qwen38_snapshot()
        self.tokenizer = Tokenizer.from_file(os.path.join(self.snap, "tokenizer.json"))
        with open(os.path.join(self.snap, "model.safetensors.index.json"), encoding="utf-8") as f:
            wmap = json.load(f)["weight_map"]

        shard_files = sorted(set(wmap.values()))
        self.weights: Dict[str, mx.array] = {}
        for sf in shard_files:
            self.weights.update(mx.load(os.path.join(self.snap, sf)))

        # Pre-adjust conv1d weights [10240, 1, 4] -> [10240, 4, 1] and (1 + w) RMSNorm weights
        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.language_model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.norm.weight",
        )
        for k in list(self.weights.keys()):
            if k.startswith("model.visual."):
                del self.weights[k]
                continue
            v = self.weights[k]
            if "conv1d.weight" in k and v.shape[-1] != 1:
                self.weights[k] = v.moveaxis(2, 1)
            elif any(k.endswith(sfx) for sfx in norm_suffixes):
                self.weights[k] = (v.astype(mx.float32) + 1.0).astype(mx.bfloat16)

        print(
            f"[MLXQwen38Engine] Memory-mapped all 18 shards ({len(self.weights)} tensors) "
            f"onto M4 Max Metal GPU in {(time.time() - t0)*1000:.1f} ms.",
            flush=True,
        )

    def forward_linear_attn(self, x: mx.array, l: int, lora: Optional[Dict[str, mx.array]] = None, lora_scale: float = 2.0) -> mx.array:
        p = f"model.language_model.layers.{l}.linear_attn"
        peft_p = f"base_model.model.language_model.layers.{l}.linear_attn"
        B, S, _ = x.shape
        nk, nv, dk, dv = 16, 48, 128, 128
        key_dim = nk * dk  # 2048
        val_dim = nv * dv  # 6144

        mixed_qkv = x @ self.weights[f"{p}.in_proj_qkv.weight"].T  # [B, S, 10240]
        z = (x @ self.weights[f"{p}.in_proj_z.weight"].T).reshape(B, S, nv, dv)
        b = x @ self.weights[f"{p}.in_proj_b.weight"].T  # [B, S, 48]
        a = x @ self.weights[f"{p}.in_proj_a.weight"].T  # [B, S, 48]

        pad_zeros = mx.zeros((B, 3, 10240), dtype=x.dtype)
        conv_in = mx.concatenate([pad_zeros, mixed_qkv], axis=1)
        conv_out = nn.silu(mx.conv1d(conv_in, self.weights[f"{p}.conv1d.weight"], groups=10240))

        q_raw, k_raw, v_raw = mx.split(conv_out, [key_dim, 2 * key_dim], axis=-1)
        q = q_raw.reshape(B, S, nk, dk)
        k = k_raw.reshape(B, S, nk, dk)
        v = v_raw.reshape(B, S, nv, dv)

        inv_scale = dk ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        y, _ = gated_delta_update(
            q, k, v, a, b, self.weights[f"{p}.A_log"], self.weights[f"{p}.dt_bias"]
        )
        y_norm = mx.fast.rms_norm(y, self.weights[f"{p}.norm.weight"], 1e-6) * nn.silu(z)
        y_flat = y_norm.reshape(B, S, val_dim)
        out = y_flat @ self.weights[f"{p}.out_proj.weight"].T
        if lora and f"{peft_p}.out_proj.lora_A.weight" in lora:
            la = lora[f"{peft_p}.out_proj.lora_A.weight"]
            lb = lora[f"{peft_p}.out_proj.lora_B.weight"]
            out = out + ((y_flat @ la.T) @ lb.T) * lora_scale
        return out

    def forward_full_attn(self, x: mx.array, l: int, mask: Optional[mx.array], lora: Optional[Dict[str, mx.array]] = None, lora_scale: float = 2.0) -> mx.array:
        p = f"model.language_model.layers.{l}.self_attn"
        peft_p = f"base_model.model.language_model.layers.{l}.self_attn"
        B, L, _ = x.shape
        nh, nkv, hd = 24, 4, 256
        scale = hd ** -0.5

        q_out = (x @ self.weights[f"{p}.q_proj.weight"].T).reshape(B, L, nh, hd * 2)
        queries, gate = mx.split(q_out, [hd], axis=-1)
        gate = gate.reshape(B, L, nh * hd)

        keys = (x @ self.weights[f"{p}.k_proj.weight"].T).reshape(B, L, nkv, hd)
        values = (x @ self.weights[f"{p}.v_proj.weight"].T).reshape(B, L, nkv, hd)

        queries = mx.fast.rms_norm(queries, self.weights[f"{p}.q_norm.weight"], 1e-6).transpose(0, 2, 1, 3)
        keys = mx.fast.rms_norm(keys, self.weights[f"{p}.k_norm.weight"], 1e-6).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = mx.fast.rope(queries, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=0)
        keys = mx.fast.rope(keys, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=0)

        attn_out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask=mask)
        attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(B, L, nh * hd)
        gated = attn_flat * mx.sigmoid(gate)
        out = gated @ self.weights[f"{p}.o_proj.weight"].T
        if lora and f"{peft_p}.o_proj.lora_A.weight" in lora:
            la = lora[f"{peft_p}.o_proj.lora_A.weight"]
            lb = lora[f"{peft_p}.o_proj.lora_B.weight"]
            out = out + ((gated @ la.T) @ lb.T) * lora_scale
        return out

    def forward_mlp(self, x: mx.array, l: int, lora: Optional[Dict[str, mx.array]] = None, lora_scale: float = 2.0) -> mx.array:
        p = f"model.language_model.layers.{l}.mlp"
        peft_p = f"base_model.model.language_model.layers.{l}.mlp"
        act = nn.silu(x @ self.weights[f"{p}.gate_proj.weight"].T) * (x @ self.weights[f"{p}.up_proj.weight"].T)
        out = act @ self.weights[f"{p}.down_proj.weight"].T
        if lora and f"{peft_p}.down_proj.lora_A.weight" in lora:
            la = lora[f"{peft_p}.down_proj.lora_A.weight"]
            lb = lora[f"{peft_p}.down_proj.lora_B.weight"]
            out = out + ((act @ la.T) @ lb.T) * lora_scale
        return out

    def forward_hidden(self, input_ids: mx.array, lora: Optional[Dict[str, mx.array]] = None, lora_scale: float = 2.0) -> mx.array:
        """Run all 64 layers on `input_ids` (`[B, L]`) on M4 Max Metal GPU and return final normed hidden states (`[B, L, 5120]`)."""
        B, L = input_ids.shape
        h = self.weights["model.language_model.embed_tokens.weight"][input_ids]
        mask = "causal" if L > 1 else None

        for l in range(64):
            lp = f"model.language_model.layers.{l}"
            h_in = mx.fast.rms_norm(h, self.weights[f"{lp}.input_layernorm.weight"], 1e-6)
            if l % 4 != 3:
                h = h + self.forward_linear_attn(h_in, l, lora=lora, lora_scale=lora_scale)
            else:
                h = h + self.forward_full_attn(h_in, l, mask=mask, lora=lora, lora_scale=lora_scale)
            h_post = mx.fast.rms_norm(h, self.weights[f"{lp}.post_attention_layernorm.weight"], 1e-6)
            h = h + self.forward_mlp(h_post, l, lora=lora, lora_scale=lora_scale)
            if l % 8 == 7:
                mx.eval(h)

        final_h = mx.fast.rms_norm(h, self.weights["model.language_model.norm.weight"], 1e-6)
        mx.eval(final_h)
        return final_h

    def forward_branched_slots(
        self,
        pref_ids: mx.array,
        pref_pad_lens: List[int],
        suff_ids: mx.array,
        lora: Optional[Dict[str, mx.array]] = None,
        lora_scale: float = 2.0,
    ) -> dict:
        """Run left-padded shared prefixes `[B, L_p]` and 4-way branched suffixes `[4*B, L_s]` through all 64 layers."""
        B, L_p = pref_ids.shape
        B4, L_s = suff_ids.shape
        assert B4 == B * 4

        pos_p = mx.arange(L_p)[None, :]
        pad_arr = mx.array(pref_pad_lens, dtype=mx.int32)[:, None]
        valid_p = (pos_p >= pad_arr).astype(mx.bfloat16)[:, :, None]

        k_idx_p = mx.arange(L_p)[None, None, None, :]
        q_idx_p = mx.arange(L_p)[None, None, :, None]
        pad_4d = pad_arr[:, None, None, :]
        allow_p = (k_idx_p >= pad_4d) & (k_idx_p <= q_idx_p)
        attn_mask_p = mx.where(allow_p, mx.array(0.0, dtype=mx.bfloat16), mx.array(-1e9, dtype=mx.bfloat16))

        pad_4d_b4 = mx.repeat(pad_4d, 4, axis=0)
        k_idx_s = mx.arange(L_p + L_s)[None, None, None, :]
        q_idx_s = (L_p + mx.arange(L_s))[None, None, :, None]
        allow_s = (k_idx_s >= pad_4d_b4) & (k_idx_s <= q_idx_s)
        attn_mask_s = mx.where(allow_s, mx.array(0.0, dtype=mx.bfloat16), mx.array(-1e9, dtype=mx.bfloat16))

        h_p = self.weights["model.language_model.embed_tokens.weight"][pref_ids] * valid_p
        h_s = self.weights["model.language_model.embed_tokens.weight"][suff_ids]

        nk, nv, dk, dv = 16, 48, 128, 128
        key_dim = nk * dk
        val_dim = nv * dv
        inv_scale = dk ** -0.5
        nh, nkv, hd = 24, 4, 256
        attn_scale = hd ** -0.5

        act63_pref_last = None
        act63_suff = None
        gated63_pref_last = None
        gated63_suff = None
        y62_pref_last = None
        y62_suff = None

        for l in range(64):
            lp = f"model.language_model.layers.{l}"
            peft_lp = f"base_model.model.language_model.layers.{l}"
            w_in_norm = self.weights[f"{lp}.input_layernorm.weight"]
            h_p_in = mx.fast.rms_norm(h_p, w_in_norm, 1e-6) * valid_p
            h_s_in = mx.fast.rms_norm(h_s, w_in_norm, 1e-6)

            if l % 4 != 3:
                p = f"{lp}.linear_attn"
                peft_p = f"{peft_lp}.linear_attn"
                w_qkv = self.weights[f"{p}.in_proj_qkv.weight"].T
                w_z = self.weights[f"{p}.in_proj_z.weight"].T
                w_b = self.weights[f"{p}.in_proj_b.weight"].T
                w_a = self.weights[f"{p}.in_proj_a.weight"].T
                w_conv = self.weights[f"{p}.conv1d.weight"]
                A_log = self.weights[f"{p}.A_log"]
                dt_bias = self.weights[f"{p}.dt_bias"]
                w_norm = self.weights[f"{p}.norm.weight"]
                w_out = self.weights[f"{p}.out_proj.weight"].T

                mqkv_p = (h_p_in @ w_qkv) * valid_p
                z_p = (h_p_in @ w_z).reshape(B, L_p, nv, dv)
                b_p = h_p_in @ w_b
                a_p = h_p_in @ w_a

                conv_in_p = mx.concatenate([mx.zeros((B, 3, 10240), dtype=h_p.dtype), mqkv_p], axis=1)
                conv_out_p = nn.silu(mx.conv1d(conv_in_p, w_conv, groups=10240)) * valid_p
                q_p, k_p, v_p = mx.split(conv_out_p, [key_dim, 2 * key_dim], axis=-1)
                q_p = (inv_scale ** 2) * mx.fast.rms_norm(q_p.reshape(B, L_p, nk, dk), None, 1e-6)
                k_p = inv_scale * mx.fast.rms_norm(k_p.reshape(B, L_p, nk, dk), None, 1e-6)
                v_p = v_p.reshape(B, L_p, nv, dv) * valid_p[:, :, :, None]

                y_p, delta_state = gated_delta_update(q_p, k_p, v_p, a_p, b_p, A_log, dt_bias)
                y_p_norm = (mx.fast.rms_norm(y_p, w_norm, 1e-6) * nn.silu(z_p)).reshape(B, L_p, val_dim)
                if l == 62:
                    y62_pref_last = y_p_norm[:, -1:, :]
                out_p = y_p_norm @ w_out
                if lora and f"{peft_p}.out_proj.lora_A.weight" in lora:
                    la = lora[f"{peft_p}.out_proj.lora_A.weight"]
                    lb = lora[f"{peft_p}.out_proj.lora_B.weight"]
                    out_p = out_p + ((y_p_norm @ la.T) @ lb.T) * lora_scale
                h_p = (h_p + out_p) * valid_p

                if L_s > 0:
                    conv_state_4 = mx.repeat(conv_in_p[:, -3:, :], 4, axis=0)
                    delta_state_4 = mx.repeat(delta_state, 4, axis=0)
                    mqkv_s = h_s_in @ w_qkv
                    z_s = (h_s_in @ w_z).reshape(B4, L_s, nv, dv)
                    b_s = h_s_in @ w_b
                    a_s = h_s_in @ w_a

                    conv_in_s = mx.concatenate([conv_state_4, mqkv_s], axis=1)
                    conv_out_s = nn.silu(mx.conv1d(conv_in_s, w_conv, groups=10240))
                    q_s, k_s, v_s = mx.split(conv_out_s, [key_dim, 2 * key_dim], axis=-1)
                    q_s = (inv_scale ** 2) * mx.fast.rms_norm(q_s.reshape(B4, L_s, nk, dk), None, 1e-6)
                    k_s = inv_scale * mx.fast.rms_norm(k_s.reshape(B4, L_s, nk, dk), None, 1e-6)
                    v_s = v_s.reshape(B4, L_s, nv, dv)

                    y_s, _ = gated_delta_update(q_s, k_s, v_s, a_s, b_s, A_log, dt_bias, state=delta_state_4)
                    y_s_norm = (mx.fast.rms_norm(y_s, w_norm, 1e-6) * nn.silu(z_s)).reshape(B4, L_s, val_dim)
                    if l == 62:
                        y62_suff = y_s_norm
                    out_s = y_s_norm @ w_out
                    if lora and f"{peft_p}.out_proj.lora_A.weight" in lora:
                        la = lora[f"{peft_p}.out_proj.lora_A.weight"]
                        lb = lora[f"{peft_p}.out_proj.lora_B.weight"]
                        out_s = out_s + ((y_s_norm @ la.T) @ lb.T) * lora_scale
                    h_s = h_s + out_s
            else:
                p = f"{lp}.self_attn"
                peft_p = f"{peft_lp}.self_attn"
                w_q = self.weights[f"{p}.q_proj.weight"].T
                w_k = self.weights[f"{p}.k_proj.weight"].T
                w_v = self.weights[f"{p}.v_proj.weight"].T
                w_qn = self.weights[f"{p}.q_norm.weight"]
                w_kn = self.weights[f"{p}.k_norm.weight"]
                w_o = self.weights[f"{p}.o_proj.weight"].T

                q_out_p = (h_p_in @ w_q).reshape(B, L_p, nh, hd * 2)
                queries_p, gate_p = mx.split(q_out_p, [hd], axis=-1)
                gate_p = gate_p.reshape(B, L_p, nh * hd)
                keys_p = (h_p_in @ w_k).reshape(B, L_p, nkv, hd)
                values_p = (h_p_in @ w_v).reshape(B, L_p, nkv, hd)

                queries_p = mx.fast.rms_norm(queries_p, w_qn, 1e-6).transpose(0, 2, 1, 3)
                keys_p = mx.fast.rms_norm(keys_p, w_kn, 1e-6).transpose(0, 2, 1, 3)
                values_p = values_p.transpose(0, 2, 1, 3)

                queries_p = mx.fast.rope(queries_p, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=0)
                keys_p = mx.fast.rope(keys_p, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=0)

                attn_p = mx.fast.scaled_dot_product_attention(queries_p, keys_p, values_p, scale=attn_scale, mask=attn_mask_p)
                gated_p = attn_p.transpose(0, 2, 1, 3).reshape(B, L_p, nh * hd) * mx.sigmoid(gate_p)
                if l == 63:
                    gated63_pref_last = gated_p[:, -1:, :]
                out_p = gated_p @ w_o
                if lora and f"{peft_p}.o_proj.lora_A.weight" in lora:
                    la = lora[f"{peft_p}.o_proj.lora_A.weight"]
                    lb = lora[f"{peft_p}.o_proj.lora_B.weight"]
                    out_p = out_p + ((gated_p @ la.T) @ lb.T) * lora_scale
                h_p = (h_p + out_p) * valid_p

                if L_s > 0:
                    q_out_s = (h_s_in @ w_q).reshape(B4, L_s, nh, hd * 2)
                    queries_s, gate_s = mx.split(q_out_s, [hd], axis=-1)
                    gate_s = gate_s.reshape(B4, L_s, nh * hd)
                    keys_s = (h_s_in @ w_k).reshape(B4, L_s, nkv, hd)
                    values_s = (h_s_in @ w_v).reshape(B4, L_s, nkv, hd)

                    queries_s = mx.fast.rms_norm(queries_s, w_qn, 1e-6).transpose(0, 2, 1, 3)
                    keys_s = mx.fast.rms_norm(keys_s, w_kn, 1e-6).transpose(0, 2, 1, 3)
                    values_s = values_s.transpose(0, 2, 1, 3)

                    queries_s = mx.fast.rope(queries_s, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=L_p)
                    keys_s = mx.fast.rope(keys_s, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=L_p)

                    keys_all = mx.concatenate([mx.repeat(keys_p, 4, axis=0), keys_s], axis=2)
                    values_all = mx.concatenate([mx.repeat(values_p, 4, axis=0), values_s], axis=2)

                    attn_s = mx.fast.scaled_dot_product_attention(queries_s, keys_all, values_all, scale=attn_scale, mask=attn_mask_s)
                    gated_s = attn_s.transpose(0, 2, 1, 3).reshape(B4, L_s, nh * hd) * mx.sigmoid(gate_s)
                    if l == 63:
                        gated63_suff = gated_s
                    out_s = gated_s @ w_o
                    if lora and f"{peft_p}.o_proj.lora_A.weight" in lora:
                        la = lora[f"{peft_p}.o_proj.lora_A.weight"]
                        lb = lora[f"{peft_p}.o_proj.lora_B.weight"]
                        out_s = out_s + ((gated_s @ la.T) @ lb.T) * lora_scale
                    h_s = h_s + out_s

            # MLP
            w_post_norm = self.weights[f"{lp}.post_attention_layernorm.weight"]
            h_p_post = mx.fast.rms_norm(h_p, w_post_norm, 1e-6) * valid_p
            p_mlp = f"{lp}.mlp"
            peft_mlp = f"{peft_lp}.mlp"
            w_gate = self.weights[f"{p_mlp}.gate_proj.weight"].T
            w_up = self.weights[f"{p_mlp}.up_proj.weight"].T
            w_down = self.weights[f"{p_mlp}.down_proj.weight"].T

            act_p = nn.silu(h_p_post @ w_gate) * (h_p_post @ w_up)
            if l == 63:
                act63_pref_last = act_p[:, -1:, :]
            out_mlp_p = act_p @ w_down
            h_p = (h_p + out_mlp_p) * valid_p
            if lora and f"{peft_mlp}.down_proj.lora_A.weight" in lora:
                la = lora[f"{peft_mlp}.down_proj.lora_A.weight"]
                lb = lora[f"{peft_mlp}.down_proj.lora_B.weight"]
                if l == 63:
                    lora_dp = ((act_p.astype(mx.float32) @ la.astype(mx.float32).T) @ lb.astype(mx.float32).T) * lora_scale
                    h_p = (h_p.astype(mx.float32) + lora_dp) * valid_p.astype(mx.float32)
                else:
                    h_p = (h_p + ((act_p @ la.T) @ lb.T) * lora_scale) * valid_p

            if L_s > 0:
                h_s_post = mx.fast.rms_norm(h_s, w_post_norm, 1e-6)
                act_s = nn.silu(h_s_post @ w_gate) * (h_s_post @ w_up)
                if l == 63:
                    act63_suff = act_s
                out_mlp_s = act_s @ w_down
                h_s = h_s + out_mlp_s
                if lora and f"{peft_mlp}.down_proj.lora_A.weight" in lora:
                    la = lora[f"{peft_mlp}.down_proj.lora_A.weight"]
                    lb = lora[f"{peft_mlp}.down_proj.lora_B.weight"]
                    if l == 63:
                        lora_ds = ((act_s.astype(mx.float32) @ la.astype(mx.float32).T) @ lb.astype(mx.float32).T) * lora_scale
                        h_s = h_s.astype(mx.float32) + lora_ds
                    else:
                        h_s = h_s + ((act_s @ la.T) @ lb.T) * lora_scale

            if l % 8 == 7:
                mx.eval(h_p, h_s)

        h63_base_pref_last = h_p[:, -1:, :]
        h63_base_suff = h_s
        w_final = self.weights["model.language_model.norm.weight"].astype(h63_base_pref_last.dtype)
        final_pref_last = mx.fast.rms_norm(h63_base_pref_last, w_final, 1e-6).astype(mx.bfloat16)
        final_suff = mx.fast.rms_norm(h63_base_suff, w_final, 1e-6).astype(mx.bfloat16) if L_s > 0 else h63_base_suff.astype(mx.bfloat16)
        mx.eval(
            final_pref_last,
            final_suff,
            act63_pref_last,
            act63_suff,
            gated63_pref_last,
            gated63_suff,
            y62_pref_last,
            y62_suff,
            h63_base_pref_last,
            h63_base_suff,
        )
        return {
            "final_pref_last": final_pref_last,
            "final_suff": final_suff,
            "act63_pref_last": act63_pref_last,
            "act63_suff": act63_suff,
            "gated63_pref_last": gated63_pref_last,
            "gated63_suff": gated63_suff,
            "y62_pref_last": y62_pref_last,
            "y62_suff": y62_suff,
            "h63_base_pref_last": h63_base_pref_last,
            "h63_base_suff": h63_base_suff,
        }

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        lora: Optional[Dict[str, mx.array]] = None,
        lora_scale: float = 1.0,
    ) -> List[str]:
        """Fast batched autoregressive generation with GatedDeltaNet state caching and Full-Attn KV caching."""
        B = len(prompts)
        tok_lists = [self.tokenizer.encode(p).ids for p in prompts]
        L_p = max(len(x) for x in tok_lists)
        pref_pad_lens = [L_p - len(x) for x in tok_lists]
        pref_arr = mx.zeros((B, L_p), dtype=mx.int32)
        for b, ids in enumerate(tok_lists):
            pad_l = pref_pad_lens[b]
            pref_arr[b, pad_l:] = mx.array(ids, dtype=mx.int32)

        idx_p = mx.arange(L_p)[None, :]
        pad_arr = mx.array(pref_pad_lens, dtype=mx.int32)[:, None]
        valid_p_2d = idx_p >= pad_arr
        valid_p = valid_p_2d[:, :, None].astype(mx.bfloat16)

        causal_p = mx.tril(mx.ones((L_p, L_p), dtype=mx.bool_))[None, None, :, :]
        valid_k_p = valid_p_2d[:, None, None, :]
        attn_mask_p = mx.where(causal_p & valid_k_p, mx.array(0.0, dtype=mx.bfloat16), mx.array(-1e9, dtype=mx.bfloat16))

        h = self.weights["model.language_model.embed_tokens.weight"][pref_arr] * valid_p
        nk, dk, nv, dv = 16, 128, 48, 128
        key_dim = nk * dk
        val_dim = nv * dv
        inv_scale = dk ** -0.5
        nh, nkv, hd = 24, 4, 256
        attn_scale = hd ** -0.5

        conv_states: Dict[int, mx.array] = {}
        delta_states: Dict[int, mx.array] = {}
        k_caches: Dict[int, mx.array] = {}
        v_caches: Dict[int, mx.array] = {}

        # 1. Prefill all 64 layers
        for l in range(64):
            lp = f"model.language_model.layers.{l}"
            peft_lp = f"base_model.model.language_model.layers.{l}"
            w_in_norm = self.weights[f"{lp}.input_layernorm.weight"]
            h_in = mx.fast.rms_norm(h, w_in_norm, 1e-6) * valid_p

            if l % 4 != 3:
                p = f"{lp}.linear_attn"
                peft_p = f"{peft_lp}.linear_attn"
                w_qkv = self.weights[f"{p}.in_proj_qkv.weight"].T
                w_z = self.weights[f"{p}.in_proj_z.weight"].T
                w_b = self.weights[f"{p}.in_proj_b.weight"].T
                w_a = self.weights[f"{p}.in_proj_a.weight"].T
                w_conv = self.weights[f"{p}.conv1d.weight"]
                A_log = self.weights[f"{p}.A_log"]
                dt_bias = self.weights[f"{p}.dt_bias"]
                w_norm = self.weights[f"{p}.norm.weight"]
                w_out = self.weights[f"{p}.out_proj.weight"].T

                mqkv_p = (h_in @ w_qkv) * valid_p
                z_p = (h_in @ w_z).reshape(B, L_p, nv, dv)
                b_p = h_in @ w_b
                a_p = h_in @ w_a

                conv_in_p = mx.concatenate([mx.zeros((B, 3, 10240), dtype=h.dtype), mqkv_p], axis=1)
                conv_states[l] = conv_in_p[:, -3:, :]
                conv_out_p = nn.silu(mx.conv1d(conv_in_p, w_conv, groups=10240)) * valid_p
                q_p, k_p, v_p = mx.split(conv_out_p, [key_dim, 2 * key_dim], axis=-1)
                q_p = (inv_scale ** 2) * mx.fast.rms_norm(q_p.reshape(B, L_p, nk, dk), None, 1e-6)
                k_p = inv_scale * mx.fast.rms_norm(k_p.reshape(B, L_p, nk, dk), None, 1e-6)
                v_p = v_p.reshape(B, L_p, nv, dv) * valid_p[:, :, :, None]

                y_p, d_state = gated_delta_update(q_p, k_p, v_p, a_p, b_p, A_log, dt_bias)
                delta_states[l] = d_state
                y_p_norm = (mx.fast.rms_norm(y_p, w_norm, 1e-6) * nn.silu(z_p)).reshape(B, L_p, val_dim)
                out_p = y_p_norm @ w_out
                if lora and f"{peft_p}.out_proj.lora_A.weight" in lora:
                    la = lora[f"{peft_p}.out_proj.lora_A.weight"]
                    lb = lora[f"{peft_p}.out_proj.lora_B.weight"]
                    out_p = out_p + ((y_p_norm @ la.T) @ lb.T) * lora_scale
                h = (h + out_p) * valid_p
            else:
                p = f"{lp}.self_attn"
                peft_p = f"{peft_lp}.self_attn"
                w_q = self.weights[f"{p}.q_proj.weight"].T
                w_k = self.weights[f"{p}.k_proj.weight"].T
                w_v = self.weights[f"{p}.v_proj.weight"].T
                w_qn = self.weights[f"{p}.q_norm.weight"]
                w_kn = self.weights[f"{p}.k_norm.weight"]
                w_o = self.weights[f"{p}.o_proj.weight"].T

                q_out_p = (h_in @ w_q).reshape(B, L_p, nh, hd * 2)
                queries_p, gate_p = mx.split(q_out_p, [hd], axis=-1)
                gate_p = gate_p.reshape(B, L_p, nh * hd)
                keys_p = (h_in @ w_k).reshape(B, L_p, nkv, hd)
                values_p = (h_in @ w_v).reshape(B, L_p, nkv, hd)

                queries_p = mx.fast.rms_norm(queries_p, w_qn, 1e-6).transpose(0, 2, 1, 3)
                keys_p = mx.fast.rms_norm(keys_p, w_kn, 1e-6).transpose(0, 2, 1, 3)
                values_p = values_p.transpose(0, 2, 1, 3)

                queries_p = mx.fast.rope(queries_p, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=0)
                keys_p = mx.fast.rope(keys_p, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=0)
                k_caches[l] = keys_p
                v_caches[l] = values_p

                attn_p = mx.fast.scaled_dot_product_attention(queries_p, keys_p, values_p, scale=attn_scale, mask=attn_mask_p)
                gated_p = attn_p.transpose(0, 2, 1, 3).reshape(B, L_p, nh * hd) * mx.sigmoid(gate_p)
                out_p = gated_p @ w_o
                if lora and f"{peft_p}.o_proj.lora_A.weight" in lora:
                    la = lora[f"{peft_p}.o_proj.lora_A.weight"]
                    lb = lora[f"{peft_p}.o_proj.lora_B.weight"]
                    out_p = out_p + ((gated_p @ la.T) @ lb.T) * lora_scale
                h = (h + out_p) * valid_p

            w_post_norm = self.weights[f"{lp}.post_attention_layernorm.weight"]
            h_post = mx.fast.rms_norm(h, w_post_norm, 1e-6) * valid_p
            p_mlp = f"{lp}.mlp"
            peft_mlp = f"{peft_lp}.mlp"
            w_gate = self.weights[f"{p_mlp}.gate_proj.weight"].T
            w_up = self.weights[f"{p_mlp}.up_proj.weight"].T
            w_down = self.weights[f"{p_mlp}.down_proj.weight"].T

            act_p = nn.silu(h_post @ w_gate) * (h_post @ w_up)
            out_mlp_p = act_p @ w_down
            h = (h + out_mlp_p) * valid_p
            if lora and f"{peft_mlp}.down_proj.lora_A.weight" in lora:
                la = lora[f"{peft_mlp}.down_proj.lora_A.weight"]
                lb = lora[f"{peft_mlp}.down_proj.lora_B.weight"]
                if l == 63:
                    lora_dp = ((act_p.astype(mx.float32) @ la.astype(mx.float32).T) @ lb.astype(mx.float32).T) * lora_scale
                    h = (h.astype(mx.float32) + lora_dp) * valid_p.astype(mx.float32)
                else:
                    h = (h + ((act_p @ la.T) @ lb.T) * lora_scale) * valid_p

            if l % 8 == 7:
                mx.eval(h)

        w_final = self.weights["model.language_model.norm.weight"]
        w_lm = self.weights["lm_head.weight"]
        h_last = mx.fast.rms_norm(h[:, -1:, :], w_final.astype(h.dtype), 1e-6).astype(mx.bfloat16)
        next_tok = mx.argmax(h_last[:, 0, :] @ w_lm.T, axis=-1)
        mx.eval(next_tok)

        generated_ids: List[List[int]] = [[int(next_tok[b])] for b in range(B)]

        # 2. Decode single token at a time across all B prompts
        for step in range(1, max_new_tokens):
            cur_pos = L_p + step - 1
            tok_in = next_tok[:, None]  # [B, 1]
            h_step = self.weights["model.language_model.embed_tokens.weight"][tok_in]  # [B, 1, 5120]

            idx_all = mx.arange(cur_pos + 1)[None, :]
            valid_k_step = (idx_all >= pad_arr)[:, None, None, :]  # [B, 1, 1, cur_pos + 1]
            attn_mask_step = mx.where(valid_k_step, mx.array(0.0, dtype=mx.bfloat16), mx.array(-1e9, dtype=mx.bfloat16))

            for l in range(64):
                lp = f"model.language_model.layers.{l}"
                peft_lp = f"base_model.model.language_model.layers.{l}"
                w_in_norm = self.weights[f"{lp}.input_layernorm.weight"]
                h_in = mx.fast.rms_norm(h_step, w_in_norm, 1e-6)

                if l % 4 != 3:
                    p = f"{lp}.linear_attn"
                    peft_p = f"{peft_lp}.linear_attn"
                    w_qkv = self.weights[f"{p}.in_proj_qkv.weight"].T
                    w_z = self.weights[f"{p}.in_proj_z.weight"].T
                    w_b = self.weights[f"{p}.in_proj_b.weight"].T
                    w_a = self.weights[f"{p}.in_proj_a.weight"].T
                    w_conv = self.weights[f"{p}.conv1d.weight"]
                    A_log = self.weights[f"{p}.A_log"]
                    dt_bias = self.weights[f"{p}.dt_bias"]
                    w_norm = self.weights[f"{p}.norm.weight"]
                    w_out = self.weights[f"{p}.out_proj.weight"].T

                    mqkv_s = h_in @ w_qkv
                    z_s = (h_in @ w_z).reshape(B, 1, nv, dv)
                    b_s = h_in @ w_b
                    a_s = h_in @ w_a

                    conv_in_s = mx.concatenate([conv_states[l], mqkv_s], axis=1)
                    conv_states[l] = conv_in_s[:, -3:, :]
                    conv_out_s = nn.silu(mx.conv1d(conv_in_s, w_conv, groups=10240))
                    q_s, k_s, v_s = mx.split(conv_out_s, [key_dim, 2 * key_dim], axis=-1)
                    q_s = (inv_scale ** 2) * mx.fast.rms_norm(q_s.reshape(B, 1, nk, dk), None, 1e-6)
                    k_s = inv_scale * mx.fast.rms_norm(k_s.reshape(B, 1, nk, dk), None, 1e-6)
                    v_s = v_s.reshape(B, 1, nv, dv)

                    y_s, d_state = gated_delta_update(q_s, k_s, v_s, a_s, b_s, A_log, dt_bias, state=delta_states[l])
                    delta_states[l] = d_state
                    y_s_norm = (mx.fast.rms_norm(y_s, w_norm, 1e-6) * nn.silu(z_s)).reshape(B, 1, val_dim)
                    out_s = y_s_norm @ w_out
                    if lora and f"{peft_p}.out_proj.lora_A.weight" in lora:
                        la = lora[f"{peft_p}.out_proj.lora_A.weight"]
                        lb = lora[f"{peft_p}.out_proj.lora_B.weight"]
                        out_s = out_s + ((y_s_norm @ la.T) @ lb.T) * lora_scale
                    h_step = h_step + out_s
                else:
                    p = f"{lp}.self_attn"
                    peft_p = f"{peft_lp}.self_attn"
                    w_q = self.weights[f"{p}.q_proj.weight"].T
                    w_k = self.weights[f"{p}.k_proj.weight"].T
                    w_v = self.weights[f"{p}.v_proj.weight"].T
                    w_qn = self.weights[f"{p}.q_norm.weight"]
                    w_kn = self.weights[f"{p}.k_norm.weight"]
                    w_o = self.weights[f"{p}.o_proj.weight"].T

                    q_out_s = (h_in @ w_q).reshape(B, 1, nh, hd * 2)
                    queries_s, gate_s = mx.split(q_out_s, [hd], axis=-1)
                    gate_s = gate_s.reshape(B, 1, nh * hd)
                    keys_s = (h_in @ w_k).reshape(B, 1, nkv, hd)
                    values_s = (h_in @ w_v).reshape(B, 1, nkv, hd)

                    queries_s = mx.fast.rms_norm(queries_s, w_qn, 1e-6).transpose(0, 2, 1, 3)
                    keys_s = mx.fast.rms_norm(keys_s, w_kn, 1e-6).transpose(0, 2, 1, 3)
                    values_s = values_s.transpose(0, 2, 1, 3)

                    queries_s = mx.fast.rope(queries_s, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=cur_pos)
                    keys_s = mx.fast.rope(keys_s, dims=64, traditional=False, base=10_000_000.0, scale=1.0, offset=cur_pos)

                    k_caches[l] = mx.concatenate([k_caches[l], keys_s], axis=2)
                    v_caches[l] = mx.concatenate([v_caches[l], values_s], axis=2)

                    attn_s = mx.fast.scaled_dot_product_attention(
                        queries_s, k_caches[l], v_caches[l], scale=attn_scale, mask=attn_mask_step
                    )
                    gated_s = attn_s.transpose(0, 2, 1, 3).reshape(B, 1, nh * hd) * mx.sigmoid(gate_s)
                    out_s = gated_s @ w_o
                    if lora and f"{peft_p}.o_proj.lora_A.weight" in lora:
                        la = lora[f"{peft_p}.o_proj.lora_A.weight"]
                        lb = lora[f"{peft_p}.o_proj.lora_B.weight"]
                        out_s = out_s + ((gated_s @ la.T) @ lb.T) * lora_scale
                    h_step = h_step + out_s

                w_post_norm = self.weights[f"{lp}.post_attention_layernorm.weight"]
                h_post = mx.fast.rms_norm(h_step, w_post_norm, 1e-6)
                p_mlp = f"{lp}.mlp"
                peft_mlp = f"{peft_lp}.mlp"
                w_gate = self.weights[f"{p_mlp}.gate_proj.weight"].T
                w_up = self.weights[f"{p_mlp}.up_proj.weight"].T
                w_down = self.weights[f"{p_mlp}.down_proj.weight"].T

                act_s = nn.silu(h_post @ w_gate) * (h_post @ w_up)
                out_mlp_s = act_s @ w_down
                h_step = h_step + out_mlp_s
                if lora and f"{peft_mlp}.down_proj.lora_A.weight" in lora:
                    la = lora[f"{peft_mlp}.down_proj.lora_A.weight"]
                    lb = lora[f"{peft_mlp}.down_proj.lora_B.weight"]
                    if l == 63:
                        lora_ds = ((act_s.astype(mx.float32) @ la.astype(mx.float32).T) @ lb.astype(mx.float32).T) * lora_scale
                        h_step = h_step.astype(mx.float32) + lora_ds
                    else:
                        h_step = h_step + ((act_s @ la.T) @ lb.T) * lora_scale

                if l % 16 == 15:
                    mx.eval(h_step)

            h_last = mx.fast.rms_norm(h_step, w_final.astype(h_step.dtype), 1e-6).astype(mx.bfloat16)
            next_tok = mx.argmax(h_last[:, 0, :] @ w_lm.T, axis=-1)
            mx.eval(next_tok)
            for b in range(B):
                generated_ids[b].append(int(next_tok[b]))
            if step >= 12 and step % 4 == 3:
                all_done = True
                for b in range(B):
                    txt = self.tokenizer.decode(generated_ids[b])
                    lines = txt.splitlines()
                    kept = []
                    finished_b = False
                    for line in lines:
                        st = line.strip()
                        if kept and st and not line.startswith((" ", "\t")):
                            finished_b = True
                            break
                        if st.startswith(("def ", "class ", "if __name__", "# Test", "# Example", "print(", "assert ")):
                            if kept:
                                finished_b = True
                                break
                        kept.append(line)
                    if not finished_b and "\n\n" in txt and "return " in txt:
                        finished_b = True
                    if not finished_b:
                        all_done = False
                        break
                if all_done:
                    break

        return [self.tokenizer.decode(ids) for ids in generated_ids]
