"""Extract real 64-layer `Qwen/Qwen3.8-27B` features for the 16 new multi-library & math tasks
(`EXPANDED_V4_SPECS` = 64 slots = 256 candidates) on Apple M4 Max Metal GPU and combine with
`/tmp/qwen38_real_56tasks_cache.npz` into `/tmp/qwen38_real_72tasks_cache.npz` (72 tasks = 288 slots = 1,152 candidates).
"""

import os
import sys
import time
from typing import List, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, "/Users/joshua/evaluation/PiQwen")
sys.path.insert(0, "/Users/joshua/.gemini/antigravity/brain/146baa7b-6e90-4cae-ab58-824b7d75b10c/scratch")
from test_new_16_specs import EXPANDED_V4_SPECS
from mlx_qwen38_gpu import MLXQwen38Engine, gated_delta_update


def get_slot_prefix_and_options(spec: dict, k: int) -> Tuple[str, List[str]]:
    sname = spec["name"]
    code = f"# Function: {sname}\n" + spec["template"]
    tag = f"{{s{k}}}"
    for _ in range(4):
        if tag in code:
            break
        for j in range(4):
            if j != k and f"{{s{j}}}" in code and tag in spec["slot_options"][j][0]:
                code = code.replace(f"{{s{j}}}", spec["slot_options"][j][0])
    assert tag in code, f"{tag} not found in {sname}"
    prefix = code.split(tag)[0]
    for _ in range(4):
        for j in range(4):
            if f"{{s{j}}}" in prefix:
                prefix = prefix.replace(f"{{s{j}}}", spec["slot_options"][j][0])
    opts = []
    for c in spec["slot_options"][k]:
        c_exp = c
        for _ in range(4):
            for j in range(4):
                if f"{{s{j}}}" in c_exp:
                    c_exp = c_exp.replace(f"{{s{j}}}", spec["slot_options"][j][0])
        opts.append(c_exp)
    return prefix, opts


def forward_branched_slots(eng: MLXQwen38Engine, pref_ids: mx.array, pref_pad_lens: List[int], suff_ids: mx.array) -> dict:
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

    h_p = eng.weights["model.language_model.embed_tokens.weight"][pref_ids] * valid_p
    h_s = eng.weights["model.language_model.embed_tokens.weight"][suff_ids]

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
        w_in_norm = eng.weights[f"{lp}.input_layernorm.weight"]
        h_p_in = mx.fast.rms_norm(h_p, w_in_norm, 1e-6) * valid_p
        h_s_in = mx.fast.rms_norm(h_s, w_in_norm, 1e-6)

        if l % 4 != 3:
            p = f"{lp}.linear_attn"
            w_qkv = eng.weights[f"{p}.in_proj_qkv.weight"].T
            w_z = eng.weights[f"{p}.in_proj_z.weight"].T
            w_b = eng.weights[f"{p}.in_proj_b.weight"].T
            w_a = eng.weights[f"{p}.in_proj_a.weight"].T
            w_conv = eng.weights[f"{p}.conv1d.weight"]
            A_log = eng.weights[f"{p}.A_log"]
            dt_bias = eng.weights[f"{p}.dt_bias"]
            w_norm = eng.weights[f"{p}.norm.weight"]
            w_out = eng.weights[f"{p}.out_proj.weight"].T

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
                h_s = h_s + out_s
        else:
            p = f"{lp}.self_attn"
            w_q = eng.weights[f"{p}.q_proj.weight"].T
            w_k = eng.weights[f"{p}.k_proj.weight"].T
            w_v = eng.weights[f"{p}.v_proj.weight"].T
            w_qn = eng.weights[f"{p}.q_norm.weight"]
            w_kn = eng.weights[f"{p}.k_norm.weight"]
            w_o = eng.weights[f"{p}.o_proj.weight"].T

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
                h_s = h_s + out_s

        w_post_norm = eng.weights[f"{lp}.post_attention_layernorm.weight"]
        h_p_post = mx.fast.rms_norm(h_p, w_post_norm, 1e-6) * valid_p
        p_mlp = f"{lp}.mlp"
        w_gate = eng.weights[f"{p_mlp}.gate_proj.weight"].T
        w_up = eng.weights[f"{p_mlp}.up_proj.weight"].T
        w_down = eng.weights[f"{p_mlp}.down_proj.weight"].T

        act_p = nn.silu(h_p_post @ w_gate) * (h_p_post @ w_up)
        if l == 63:
            act63_pref_last = act_p[:, -1:, :]
        out_mlp_p = act_p @ w_down
        h_p = (h_p + out_mlp_p) * valid_p

        if L_s > 0:
            h_s_post = mx.fast.rms_norm(h_s, w_post_norm, 1e-6)
            act_s = nn.silu(h_s_post @ w_gate) * (h_s_post @ w_up)
            if l == 63:
                act63_suff = act_s
            out_mlp_s = act_s @ w_down
            h_s = h_s + out_mlp_s

        if l % 8 == 7:
            mx.eval(h_p, h_s)

    h63_base_pref_last = h_p[:, -1:, :]
    h63_base_suff = h_s
    w_final = eng.weights["model.language_model.norm.weight"].astype(h63_base_pref_last.dtype)
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


def main():
    t0 = time.time()
    base_cache = np.load("/tmp/qwen38_real_56tasks_cache.npz")
    eng = MLXQwen38Engine()
    tok = eng.tokenizer
    w_lm = eng.weights["lm_head.weight"]
    w_final_norm_mx = eng.weights["model.language_model.norm.weight"].astype(mx.float32)

    slot_records = []
    for t_i, spec in enumerate(EXPANDED_V4_SPECS):
        for k in range(4):
            prefix, opts = get_slot_prefix_and_options(spec, k)
            p_ids = tok.encode(prefix).ids
            c_entries = []
            for c_i, c in enumerate(opts):
                f_ids = tok.encode(prefix + c).ids
                common = 0
                while common < len(p_ids) and common < len(f_ids) and p_ids[common] == f_ids[common]:
                    common += 1
                start_pos = max(1, min(common, len(f_ids) - 1))
                c_entries.append((start_pos, f_ids))
            slot_start = min(x[0] for x in c_entries)
            pref_ids = c_entries[0][1][:slot_start]
            c_suffs = [f_ids[slot_start:-1] for _, f_ids in c_entries]
            c_targets = [f_ids[slot_start:] for _, f_ids in c_entries]
            slot_records.append({
                "slot_idx": len(slot_records),
                "task_idx": t_i,
                "task_name": spec["name"],
                "slot_k": k,
                "pref_ids": pref_ids,
                "c_suffs": c_suffs,
                "c_targets": c_targets,
            })

    sorted_order = sorted(range(len(slot_records)), key=lambda idx: len(slot_records[idx]["pref_ids"]))
    batch_size = 32
    num_new_slots = len(slot_records)
    slot_outputs = {}

    for b_idx in range(0, num_new_slots, batch_size):
        b_t0 = time.time()
        chunk_indices = sorted_order[b_idx : b_idx + batch_size]
        B = len(chunk_indices)
        L_p = max(len(slot_records[i]["pref_ids"]) for i in chunk_indices)
        L_s = max(max(len(s) for s in slot_records[i]["c_suffs"]) for i in chunk_indices)

        pref_arr = mx.zeros((B, L_p), dtype=mx.int32)
        pref_pad_lens = []
        suff_arr = mx.zeros((B * 4, max(1, L_s)), dtype=mx.int32)

        for local_b, orig_i in enumerate(chunk_indices):
            p = slot_records[orig_i]["pref_ids"]
            pad_l = L_p - len(p)
            pref_pad_lens.append(pad_l)
            pref_arr[local_b, pad_l:] = mx.array(p, dtype=mx.int32)
            for c_i in range(4):
                s = slot_records[orig_i]["c_suffs"][c_i]
                if len(s) > 0:
                    suff_arr[local_b * 4 + c_i, :len(s)] = mx.array(s, dtype=mx.int32)

        out = forward_branched_slots(
            eng,
            pref_arr,
            pref_pad_lens,
            suff_arr if L_s > 0 else mx.zeros((B * 4, 0), dtype=mx.int32),
        )
        print(
            f"[Expanded-16 Batch {b_idx//batch_size + 1}/2] Ran {B} slots (L_p={L_p}, L_s={L_s}) through all 64 layers in {time.time() - b_t0:.2f}s",
            flush=True,
        )

        for local_b, orig_i in enumerate(chunk_indices):
            rec = slot_records[orig_i]
            pos_final_h = [out["final_pref_last"][local_b, 0]]
            pos_h63_base = [out["h63_base_pref_last"][local_b, 0]]
            pos_act63 = [out["act63_pref_last"][local_b, 0]]
            pos_gated63 = [out["gated63_pref_last"][local_b, 0]]
            pos_y62 = [out["y62_pref_last"][local_b, 0]]

            cand_pos_indices = []
            for c_i in range(4):
                t_toks = rec["c_targets"][c_i]
                n_t = len(t_toks)
                c_pos = [0]
                for m in range(1, n_t):
                    p_idx = len(pos_final_h)
                    c_pos.append(p_idx)
                    b4 = local_b * 4 + c_i
                    pos_final_h.append(out["final_suff"][b4, m - 1])
                    pos_h63_base.append(out["h63_base_suff"][b4, m - 1])
                    pos_act63.append(out["act63_suff"][b4, m - 1])
                    pos_gated63.append(out["gated63_suff"][b4, m - 1])
                    pos_y62.append(out["y62_suff"][b4, m - 1])
                cand_pos_indices.append(c_pos)

            stack_final_h = mx.stack(pos_final_h, axis=0)
            logits_full = (stack_final_h @ w_lm.T).astype(mx.float32)
            lse_full = mx.logsumexp(logits_full, axis=-1)
            top8_idx = mx.argpartition(-logits_full, kth=8, axis=-1)[:, :8]

            cand_base_lps = []
            for c_i in range(4):
                t_toks = rec["c_targets"][c_i]
                c_pos = mx.array(cand_pos_indices[c_i], dtype=mx.int32)
                t_arr = mx.array(t_toks, dtype=mx.int32)
                tok_lps = logits_full[c_pos, t_arr] - lse_full[c_pos]
                mx.eval(tok_lps)
                cand_base_lps.append(np.array(tok_lps, dtype=np.float32))

            mx.eval(lse_full, top8_idx)
            slot_outputs[orig_i] = {
                "h63_base": np.array(mx.stack(pos_h63_base, axis=0).astype(mx.float32), dtype=np.float16),
                "act63": np.array(mx.stack(pos_act63, axis=0).astype(mx.float32), dtype=np.float32),
                "gated63": np.array(mx.stack(pos_gated63, axis=0).astype(mx.float32), dtype=np.float32),
                "y62": np.array(mx.stack(pos_y62, axis=0).astype(mx.float32), dtype=np.float32),
                "lse_full": np.array(lse_full, dtype=np.float32),
                "top8_idx": np.array(top8_idx, dtype=np.int32),
                "cand_pos_indices": cand_pos_indices,
                "cand_targets": rec["c_targets"],
                "cand_base_lps": cand_base_lps,
            }

    slot_correct = 0
    task_pass = 0
    for t_i in range(16):
        all_4 = True
        for k in range(4):
            s_idx = t_i * 4 + k
            means = [float(x.mean()) for x in slot_outputs[s_idx]["cand_base_lps"]]
            if int(np.argmax(means)) == 0:
                slot_correct += 1
            else:
                all_4 = False
        if all_4:
            task_pass += 1
    print(
        f"[Base Qwen/Qwen3.8-27B] EXPANDED_V4_SPECS (16 tasks) | Slot Acc: {slot_correct}/64 ({100.0*slot_correct/64:.2f}%) "
        f"| Function Pass@1: {task_pass}/16 ({100.0*task_pass/16:.2f}%)",
        flush=True,
    )

    old_active_vocab = base_cache["active_vocab_ids"].astype(np.int32)
    active_vocab_set = set(int(x) for x in old_active_vocab)

    old_h63 = base_cache["h63_base"]
    pos_offset = old_h63.shape[0]

    new_h63_list, new_act63_list, new_gated63_list, new_y62_list, new_lse_full_list = [], [], [], [], []
    new_cand_tok_pos = []
    new_cand_tok_target_full = []
    new_cand_offsets = [int(base_cache["cand_offsets"][-1])]
    new_cand_base_mean_lp = np.zeros((num_new_slots, 4), dtype=np.float32)

    for s_idx in range(num_new_slots):
        out_s = slot_outputs[s_idx]
        n_pos = out_s["h63_base"].shape[0]
        new_h63_list.append(out_s["h63_base"])
        new_act63_list.append(out_s["act63"])
        new_gated63_list.append(out_s["gated63"])
        new_y62_list.append(out_s["y62"])
        new_lse_full_list.append(out_s["lse_full"])

        for tid in out_s["top8_idx"].flatten():
            active_vocab_set.add(int(tid))

        for c_i in range(4):
            new_cand_base_mean_lp[s_idx, c_i] = float(out_s["cand_base_lps"][c_i].mean())
            for local_p, t_id in zip(out_s["cand_pos_indices"][c_i], out_s["cand_targets"][c_i]):
                new_cand_tok_pos.append(pos_offset + local_p)
                new_cand_tok_target_full.append(int(t_id))
                active_vocab_set.add(int(t_id))
            new_cand_offsets.append(int(base_cache["cand_offsets"][-1]) + len(new_cand_tok_pos))

        pos_offset += n_pos

    old_cand_target_full = old_active_vocab[base_cache["cand_tok_target_active"]]
    combined_active_vocab_ids = np.array(sorted(active_vocab_set), dtype=np.int32)
    vocab_map = {int(tid): i for i, tid in enumerate(combined_active_vocab_ids)}

    all_cand_target_full = np.concatenate([old_cand_target_full, np.array(new_cand_tok_target_full, dtype=np.int32)])
    combined_cand_tok_target_active = np.array([vocab_map[int(t)] for t in all_cand_target_full], dtype=np.int32)

    u_act63 = base_cache["u_act63"].astype(np.float32)
    u_gated63 = base_cache["u_gated63"].astype(np.float32)
    u_y62 = base_cache["u_y62"].astype(np.float32)

    new_act63_proj = np.concatenate(new_act63_list, axis=0) @ u_act63.T
    new_gated63_proj = np.concatenate(new_gated63_list, axis=0) @ u_gated63.T
    new_y62_proj = np.concatenate(new_y62_list, axis=0) @ u_y62.T

    combined_h63 = np.concatenate([old_h63, np.concatenate(new_h63_list, axis=0)], axis=0)
    combined_act63_proj = np.concatenate([base_cache["act63_proj"], new_act63_proj.astype(np.float32)], axis=0)
    combined_gated63_proj = np.concatenate([base_cache["gated63_proj"], new_gated63_proj.astype(np.float32)], axis=0)
    combined_y62_proj = np.concatenate([base_cache["y62_proj"], new_y62_proj.astype(np.float32)], axis=0)
    combined_lse_full = np.concatenate([base_cache["lse_full"], np.concatenate(new_lse_full_list, axis=0)], axis=0)

    w_lm_active_mx = w_lm[mx.array(combined_active_vocab_ids, dtype=mx.int32)].astype(mx.float32)
    mx.eval(w_lm_active_mx, w_final_norm_mx)
    w_lm_active = np.array(w_lm_active_mx, dtype=np.float32)

    h63_mx = mx.array(combined_h63, dtype=mx.float32)
    final_h_mx = mx.fast.rms_norm(h63_mx, w_final_norm_mx, 1e-6)
    logits_act_mx = final_h_mx @ w_lm_active_mx.T
    lse_act_mx = mx.logsumexp(logits_act_mx, axis=-1)
    mx.eval(lse_act_mx)
    lse_act = np.array(lse_act_mx, dtype=np.float32)
    diff_exp = np.maximum(1e-12, np.exp(combined_lse_full.astype(np.float64)) - np.exp(lse_act.astype(np.float64)))
    combined_lse_rest = np.log(diff_exp).astype(np.float32)

    out_path = "/tmp/qwen38_real_72tasks_cache.npz"
    np.savez(
        out_path,
        h63_base=combined_h63,
        act63_proj=combined_act63_proj,
        gated63_proj=combined_gated63_proj,
        y62_proj=combined_y62_proj,
        u_act63=u_act63,
        u_gated63=u_gated63,
        u_y62=u_y62,
        layer_down_a=base_cache["layer_down_a"],
        layer_attn_a=base_cache["layer_attn_a"],
        w_lm_active=w_lm_active,
        w_final_norm=base_cache["w_final_norm"],
        active_vocab_ids=combined_active_vocab_ids,
        lse_full=combined_lse_full,
        lse_rest=combined_lse_rest,
        cand_tok_pos=np.concatenate([base_cache["cand_tok_pos"], np.array(new_cand_tok_pos, dtype=np.int32)]),
        cand_tok_target_active=combined_cand_tok_target_active,
        cand_offsets=np.concatenate([base_cache["cand_offsets"], np.array(new_cand_offsets[1:], dtype=np.int32)]),
        cand_base_mean_lp=np.concatenate([base_cache["cand_base_mean_lp"], new_cand_base_mean_lp], axis=0),
    )
    print(
        f"[Done] Saved {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB) in {time.time() - t0:.2f}s! "
        f"Total Slots=288 (72 tasks), N_pos={combined_h63.shape[0]}, V_active={len(combined_active_vocab_ids)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
