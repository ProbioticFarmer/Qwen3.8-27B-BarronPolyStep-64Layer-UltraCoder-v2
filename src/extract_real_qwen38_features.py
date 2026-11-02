"""Extract 100% real, unmodified 64-layer `Qwen/Qwen3.8-27B` features and exact 248,320-vocab log-probabilities
across all 56 coding tasks (`32` Algorithm tasks + `24` Python/Data Science tasks = `224` slots = `896` candidates)
on Apple M4 Max Metal GPU.

Enforces strict train-only PCA/SVD basis extraction (Zero Transductive Leakage):
- `u_act63_train40`, `u_gated63_train40`, `u_y62_train40`, and `layer_down_a` / `layer_attn_a` are computed
  strictly from the 40 Training Tasks (`160` slots), with zero positions from the 16 Held-Out Test Tasks (`64` slots).
- `u_act63_even` / `u_act63_odd` (and `gated63`, `y62`) are computed strictly from Even / Odd tasks respectively
  for 2-Fold Out-of-Sample Cross-Validation.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np

# Support running from either PolyStep root or PiQwen root
try:
    from src.coding_tasks import PYTHON_DS_BENCHMARK_SPECS, ULTRA_32_SPECS
    from src.mlx_qwen38_gpu import MLXQwen38Engine
except ImportError:
    from coding_tasks import PYTHON_DS_BENCHMARK_SPECS, ULTRA_32_SPECS
    from mlx_qwen38_gpu import MLXQwen38Engine

# Canonical 40-Train / 16-Held-Out Test Split (8 Algo + 8 PyDS held out)
HELD_OUT_TEST_TASKS = sorted(
    [2, 5, 7, 12, 13, 18, 27, 29] + [32 + i for i in [2, 5, 8, 11, 14, 17, 20, 23]]
)
TRAIN_40_TASKS = [t for t in range(56) if t not in set(HELD_OUT_TEST_TASKS)]
HELDOUT_16_TASK_INDICES = HELD_OUT_TEST_TASKS
TRAIN_40_TASK_INDICES = TRAIN_40_TASKS


def get_slot_prefix_and_options(spec: dict, k: int) -> Tuple[str, List[str]]:
    """Expand parent and preceding slots so `{s_k}` is at its exact syntactic position."""
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


def top_right_singular_train_only(
    mat_all: np.ndarray,
    train_mask: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute top-k right singular vectors strictly on `mat_all[train_mask]`, then project `mat_all`."""
    mat_tr = mat_all[train_mask]
    gram = mat_tr @ mat_tr.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    idx = np.argsort(eigvals)[::-1][:k]
    s_inv = 1.0 / np.sqrt(np.maximum(eigvals[idx], 1e-8))
    vt = (eigvecs[:, idx] * s_inv[None, :]).T @ mat_tr  # [k, D]
    vt = vt / np.maximum(np.linalg.norm(vt, axis=-1, keepdims=True), 1e-8)
    proj_all = mat_all @ vt.T  # [N_pos, k]
    return vt.astype(np.float32), proj_all.astype(np.float32)


def main():
    t_start = time.time()
    try:
        eng = MLXQwen38Engine(subspace_gguf_path=None)
    except TypeError:
        eng = MLXQwen38Engine()
    tok = eng.tokenizer

    all_specs = []
    for t_i, s in enumerate(ULTRA_32_SPECS):
        all_specs.append(("algo32", t_i, s))
    for t_i, s in enumerate(PYTHON_DS_BENCHMARK_SPECS):
        all_specs.append(("pyds24", t_i, s))

    slot_records = []
    for suite_name, t_i, spec in all_specs:
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
            slot_idx = len(slot_records)
            slot_records.append({
                "slot_idx": slot_idx,
                "suite": suite_name,
                "task_idx": t_i,
                "task_name": spec["name"],
                "slot_k": k,
                "pref_ids": pref_ids,
                "c_suffs": c_suffs,
                "c_targets": c_targets,
            })

    sorted_order = sorted(range(len(slot_records)), key=lambda idx: len(slot_records[idx]["pref_ids"]))
    batch_size = 56
    num_slots = len(slot_records)

    slot_outputs: Dict[int, dict] = {}
    w_lm = eng.weights["lm_head.weight"]

    for b_idx in range(0, num_slots, batch_size):
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

        out = eng.forward_branched_slots(pref_arr, pref_pad_lens, suff_arr if L_s > 0 else mx.zeros((B * 4, 0), dtype=mx.int32))
        print(
            f"[Batch {b_idx//batch_size + 1}/4] Ran {B} slots (L_p={L_p}, L_s={L_s}) through all 64 layers in {time.time() - b_t0:.2f}s",
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
                "act63": np.array(mx.stack(pos_act63, axis=0).astype(mx.float32), dtype=np.float16),
                "gated63": np.array(mx.stack(pos_gated63, axis=0).astype(mx.float32), dtype=np.float16),
                "y62": np.array(mx.stack(pos_y62, axis=0).astype(mx.float32), dtype=np.float16),
                "lse_full": np.array(lse_full, dtype=np.float32),
                "top8_idx": np.array(top8_idx, dtype=np.int32),
                "cand_pos_indices": cand_pos_indices,
                "cand_targets": rec["c_targets"],
                "cand_base_lps": cand_base_lps,
            }

    for s_name, label, n_tasks in [("algo32", "ULTRA_32_SPECS (Algo-32)", 32), ("pyds24", "PYTHON_DS_BENCHMARK_SPECS (PyDS-24)", 24)]:
        slot_correct = 0
        task_pass = 0
        for t_i in range(n_tasks):
            all_4 = True
            for k in range(4):
                s_idx = (0 if s_name == "algo32" else 128) + t_i * 4 + k
                out_s = slot_outputs[s_idx]
                means = [float(x.mean()) for x in out_s["cand_base_lps"]]
                pred = int(np.argmax(means))
                if pred == 0:
                    slot_correct += 1
                else:
                    all_4 = False
            if all_4:
                task_pass += 1
        print(
            f"[Base Qwen/Qwen3.8-27B] {label:36s} | Slot Acc: {slot_correct}/{n_tasks*4} ({100.0*slot_correct/(n_tasks*4):.2f}%) "
            f"| Function Pass@1: {task_pass}/{n_tasks} ({100.0*task_pass/n_tasks:.2f}%)",
            flush=True,
        )

    active_vocab_set = set()
    python_kw = [
        "def", "return", "if", "else", "elif", "for", "while", "in", "not",
        "and", "or", "True", "False", "None", "lambda", "import", "from",
        "class", "try", "except", "len", "range", "max", "min", "sum",
        "sorted", "list", "dict", "set", "np", "pd",
    ]
    for kw in python_kw:
        for variant in [kw, " " + kw, "\n" + kw]:
            for tid in tok.encode(variant).ids:
                active_vocab_set.add(int(tid))

    all_h63 = []
    all_act63 = []
    all_gated63 = []
    all_y62 = []
    all_lse_full = []
    pos_slot_id_list = []

    cand_tok_pos = []
    cand_tok_target_full = []
    cand_offsets = [0]
    cand_base_mean_lp = np.zeros((num_slots, 4), dtype=np.float32)

    pos_offset = 0
    for s_idx in range(num_slots):
        out_s = slot_outputs[s_idx]
        n_pos = out_s["h63_base"].shape[0]
        all_h63.append(out_s["h63_base"])
        all_act63.append(out_s["act63"])
        all_gated63.append(out_s["gated63"])
        all_y62.append(out_s["y62"])
        all_lse_full.append(out_s["lse_full"])
        pos_slot_id_list.extend([s_idx] * n_pos)

        for tid in out_s["top8_idx"].flatten():
            active_vocab_set.add(int(tid))

        for c_i in range(4):
            cand_base_mean_lp[s_idx, c_i] = float(out_s["cand_base_lps"][c_i].mean())
            for local_p, t_id in zip(out_s["cand_pos_indices"][c_i], out_s["cand_targets"][c_i]):
                cand_tok_pos.append(pos_offset + local_p)
                cand_tok_target_full.append(int(t_id))
                active_vocab_set.add(int(t_id))
            cand_offsets.append(len(cand_tok_pos))

        pos_offset += n_pos

    active_vocab_ids = np.array(sorted(active_vocab_set), dtype=np.int32)
    vocab_map = {int(tid): i for i, tid in enumerate(active_vocab_ids)}
    cand_tok_target_active = np.array([vocab_map[t] for t in cand_tok_target_full], dtype=np.int32)
    pos_slot_id = np.array(pos_slot_id_list, dtype=np.int32)
    pos_task_id = pos_slot_id // 4

    h63_mat = np.concatenate(all_h63, axis=0)
    act63_mat = np.concatenate(all_act63, axis=0).astype(np.float32)
    gated63_mat = np.concatenate(all_gated63, axis=0).astype(np.float32)
    y62_mat = np.concatenate(all_y62, axis=0).astype(np.float32)
    lse_full_vec = np.concatenate(all_lse_full, axis=0)

    w_lm_active_mx = w_lm[mx.array(active_vocab_ids, dtype=mx.int32)].astype(mx.float32)
    w_final_norm_mx = eng.weights["model.language_model.norm.weight"].astype(mx.float32)
    mx.eval(w_lm_active_mx, w_final_norm_mx)
    w_lm_active = np.array(w_lm_active_mx, dtype=np.float32)
    w_final_norm = np.array(w_final_norm_mx, dtype=np.float32)

    h63_mx = mx.array(h63_mat, dtype=mx.float32)
    final_h_mx = mx.fast.rms_norm(h63_mx, w_final_norm_mx, 1e-6)
    logits_act_mx = final_h_mx @ w_lm_active_mx.T
    lse_act_mx = mx.logsumexp(logits_act_mx, axis=-1)
    mx.eval(lse_act_mx)
    lse_act = np.array(lse_act_mx, dtype=np.float32)
    diff_exp = np.maximum(1e-12, np.exp(lse_full_vec.astype(np.float64)) - np.exp(lse_act.astype(np.float64)))
    lse_rest = np.log(diff_exp).astype(np.float32)

    # STRICT TRAIN-ONLY SVD BASES (ZERO TEST-SET LEAKAGE!)
    print("[PCA/SVD] Computing strictly train-isolated principal bases (train40, even, odd)...", flush=True)
    rank_basis = 64
    train40_set = set(TRAIN_40_TASKS)
    train40_pos_mask = np.array([int(t) in train40_set for t in pos_task_id], dtype=bool)
    even_pos_mask = (pos_task_id % 2) == 0
    odd_pos_mask = (pos_task_id % 2) == 1

    # Primary bases are computed strictly on the 40 Train Tasks (zero positions from the 16 Held-Out Test Tasks!)
    u_act63, act63_proj = top_right_singular_train_only(act63_mat, train40_pos_mask, rank_basis)
    u_gated63, gated63_proj = top_right_singular_train_only(gated63_mat, train40_pos_mask, rank_basis)
    u_y62, y62_proj = top_right_singular_train_only(y62_mat, train40_pos_mask, rank_basis)

    # Fold-isolated bases for 2-Fold Out-of-Sample Cross-Validation
    u_act63_even, act63_proj_even = top_right_singular_train_only(act63_mat, even_pos_mask, 16)
    u_act63_odd, act63_proj_odd = top_right_singular_train_only(act63_mat, odd_pos_mask, 16)
    _, gated63_proj_even = top_right_singular_train_only(gated63_mat, even_pos_mask, 6)
    _, gated63_proj_odd = top_right_singular_train_only(gated63_mat, odd_pos_mask, 6)
    _, y62_proj_even = top_right_singular_train_only(y62_mat, even_pos_mask, 6)
    _, y62_proj_odd = top_right_singular_train_only(y62_mat, odd_pos_mask, 6)

    # Extract distinct unsupervised right singular directions for all 64 layers using ONLY train40 probe states
    r_peft = 16
    layer_down_a = np.zeros((64, r_peft, 17408), dtype=np.float16)
    layer_attn_a = np.zeros((64, r_peft, 6144), dtype=np.float16)
    probe_vec = mx.array(h63_mat[train40_pos_mask][:64], dtype=mx.bfloat16)
    all_true_64 = np.ones(64, dtype=bool)

    for l in range(64):
        lp = f"model.language_model.layers.{l}"
        w_d = eng.weights[f"{lp}.mlp.down_proj.weight"]
        v_d = (probe_vec @ w_d).astype(mx.float32)
        mx.eval(v_d)
        vt_d, _ = top_right_singular_train_only(np.array(v_d, dtype=np.float32), all_true_64, r_peft)
        layer_down_a[l] = vt_d.astype(np.float16)

        if l % 4 != 3:
            w_a = eng.weights[f"{lp}.linear_attn.out_proj.weight"]
        else:
            w_a = eng.weights[f"{lp}.self_attn.o_proj.weight"]
        v_a = (probe_vec @ w_a).astype(mx.float32)
        mx.eval(v_a)
        vt_a, _ = top_right_singular_train_only(np.array(v_a, dtype=np.float32), all_true_64, r_peft)
        layer_attn_a[l] = vt_a.astype(np.float16)

    layer_down_a[63] = u_act63[:r_peft].astype(np.float16)
    layer_attn_a[63] = u_gated63[:r_peft].astype(np.float16)
    layer_attn_a[62] = u_y62[:r_peft].astype(np.float16)

    cache_path = "/tmp/qwen38_real_56tasks_cache.npz"
    np.savez(
        cache_path,
        h63_base=h63_mat,
        act63_proj=act63_proj,
        gated63_proj=gated63_proj,
        y62_proj=y62_proj,
        u_act63=u_act63,
        u_gated63=u_gated63,
        u_y62=u_y62,
        act63_proj_even=act63_proj_even,
        act63_proj_odd=act63_proj_odd,
        u_act63_even=u_act63_even,
        u_act63_odd=u_act63_odd,
        gated63_proj_even=gated63_proj_even,
        gated63_proj_odd=gated63_proj_odd,
        y62_proj_even=y62_proj_even,
        y62_proj_odd=y62_proj_odd,
        pos_slot_id=pos_slot_id,
        train40_tasks=np.array(TRAIN_40_TASKS, dtype=np.int32),
        heldout16_tasks=np.array(HELD_OUT_TEST_TASKS, dtype=np.int32),
        layer_down_a=layer_down_a,
        layer_attn_a=layer_attn_a,
        w_lm_active=w_lm_active,
        w_final_norm=w_final_norm,
        active_vocab_ids=active_vocab_ids,
        lse_full=lse_full_vec,
        lse_rest=lse_rest,
        cand_tok_pos=np.array(cand_tok_pos, dtype=np.int32),
        cand_tok_target_active=cand_tok_target_active,
        cand_offsets=np.array(cand_offsets, dtype=np.int32),
        cand_base_mean_lp=cand_base_mean_lp,
    )
    print(
        f"[Done] Saved leak-free {cache_path} ({os.path.getsize(cache_path)/(1024*1024):.1f} MB) in {time.time() - t_start:.2f}s! "
        f"N_pos={h63_mat.shape[0]} (Train40_pos={int(train40_pos_mask.sum())}, HeldOut16_pos={int((~train40_pos_mask).sum())}), V_active={len(active_vocab_ids)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
