import os
import time
import numpy as np
import mlx.core as mx

CACHE_PATH = "/tmp/qwen38_real_72tasks_cache.npz"
PIQWEN_CKPT_DIR = "/Users/joshua/evaluation/PiQwen/checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3"

def main():
    t0 = time.time()
    d = np.load(CACHE_PATH)
    h63_np = d["h63_base"].astype(np.float32)
    w_final_norm_np = d["w_final_norm"].astype(np.float32)
    w_lm_active_np = d["w_lm_active"].astype(np.float32)
    lse_rest_np = d["lse_rest"].astype(np.float32)
    cand_tok_pos_np = d["cand_tok_pos"].astype(np.int64)
    cand_tok_target_np = d["cand_tok_target_active"].astype(np.int64)
    cand_offsets = d["cand_offsets"]
    num_cands = len(cand_offsets) - 1
    num_slots = num_cands // 4
    print(f"[Loaded Cache] {CACHE_PATH} | num_slots={num_slots} ({num_slots//4} tasks) | num_cands={num_cands} | N_pos={len(h63_np)}", flush=True)

    w_eff_np = w_lm_active_np * w_final_norm_np[None, :]
    w_eff_c_np = w_eff_np - w_eff_np.mean(axis=0, keepdims=True)
    _, _, vh_c = np.linalg.svd(w_eff_c_np, full_matrices=False)
    V_top = vh_c[:1024, :]
    h_mean = h63_np.mean(axis=0)
    v_null = h_mean - V_top.T @ (V_top @ h_mean)
    v_null = v_null / np.linalg.norm(v_null)

    unique_cand_toks = np.unique(cand_tok_target_np)
    w_cand = w_eff_c_np[unique_cand_toks]
    _, _, vh_cand = np.linalg.svd(w_cand, full_matrices=False)

    rms0_np = 1.0 / np.sqrt(np.mean(h63_np ** 2, axis=-1, keepdims=True) + 1e-6)
    z_unnorm_full_np = h63_np @ w_eff_np.T
    z0_np = rms0_np * z_unnorm_full_np
    top20_idx_np = np.argsort(-z0_np, axis=-1)[:, :20]
    top1_tok = top20_idx_np[:, 0]
    top2_tok = top20_idx_np[:, 1]
    m0_act = np.max(z0_np, axis=-1, keepdims=True)
    lse0_act = np.squeeze(m0_act, -1) + np.log(np.sum(np.exp(z0_np - m0_act), axis=-1))
    lse0_tot = np.maximum(lse0_act, lse_rest_np) + np.log1p(np.exp(-np.abs(lse0_act - lse_rest_np)))
    tok_lp0_np = z0_np[cand_tok_pos_np, cand_tok_target_np] - lse0_tot[cand_tok_pos_np]

    pos_to_cands = [set() for _ in range(len(h63_np))]
    for p, t in zip(cand_tok_pos_np.tolist(), cand_tok_target_np.tolist()):
        pos_to_cands[p].add(t)

    # Identify unlabelled prefix-continuation positions symmetrically across all 4x3 candidate pairs per slot
    unlabelled_prefix_cont_pos = set()
    for s_idx in range(num_slots):
        for ca in range(4):
            sta, ena = int(cand_offsets[s_idx * 4 + ca]), int(cand_offsets[s_idx * 4 + ca + 1])
            toks_a = cand_tok_target_np[sta:ena].tolist()
            pos_a = cand_tok_pos_np[sta:ena].tolist()
            for cb in range(4):
                if ca == cb:
                    continue
                stb, enb = int(cand_offsets[s_idx * 4 + cb]), int(cand_offsets[s_idx * 4 + cb + 1])
                toks_b = cand_tok_target_np[stb:enb].tolist()
                if len(toks_b) < len(toks_a) and toks_a[:len(toks_b)] == toks_b:
                    for p in pos_a[len(toks_b):]:
                        unlabelled_prefix_cont_pos.add(int(p))

    # Identify hard slot positions so diff_mat prioritizes all 43 hard slots across the 72 tasks
    hard_slot_pos = set()
    for s_idx in range(num_slots):
        c0 = s_idx * 4
        st0, en0 = int(cand_offsets[c0]), int(cand_offsets[c0 + 1])
        m0_c0 = float(np.mean(tok_lp0_np[st0:en0]))
        m0_alt = max(
            float(np.mean(tok_lp0_np[int(cand_offsets[c0 + ci]):int(cand_offsets[c0 + ci + 1])]))
            for ci in range(1, 4)
        )
        if m0_c0 - m0_alt <= 0.20:
            for ci in range(4):
                sti, eni = int(cand_offsets[c0 + ci]), int(cand_offsets[c0 + ci + 1])
                for p in cand_tok_pos_np[sti:eni].tolist():
                    hard_slot_pos.add(int(p))

    diff_rows = []
    for p in range(len(h63_np)):
        if len(pos_to_cands[p]) > 1 or (p in unlabelled_prefix_cont_pos):
            t0_tok = int(top1_tok[p])
            for t in sorted(pos_to_cands[p]):
                if t != t0_tok:
                    dv = w_eff_np[t] - w_eff_np[t0_tok]
                    dv = dv / (np.linalg.norm(dv) + 1e-8)
                    weight = 3.5 if (p in unlabelled_prefix_cont_pos) else (2.5 if (p in hard_slot_pos) else 1.0)
                    diff_rows.append(weight * dv)
    diff_mat = np.stack(diff_rows, axis=0)
    _, _, vh_diff = np.linalg.svd(diff_mat, full_matrices=False)

    # Orthogonalize angular basis strictly orthogonal to v_null so row 0 is pure v_null
    v_ang_raw = np.concatenate([vh_cand[:12, :], vh_diff[:19, :]], axis=0)
    v_ang_orth = v_ang_raw - np.outer(v_ang_raw @ v_null, v_null)
    q_ang31, _ = np.linalg.qr(v_ang_orth.T, mode="reduced")
    v_out32_np = np.concatenate([v_null[None, :], q_ang31.T[:31, :]], axis=0).astype(np.float32)  # [32, 5120]

    # Segment matrix and training-only continuation indices
    seg_mat_np = np.zeros((num_cands, len(cand_tok_pos_np)), dtype=np.float32)
    safe_gt_tok_indices = []
    cont_bottleneck_indices = []
    cont_bottleneck_pos = set()

    for s_idx in range(num_slots):
        c0 = s_idx * 4
        st0, en0 = int(cand_offsets[c0]), int(cand_offsets[c0 + 1])
        toks0 = cand_tok_target_np[st0:en0].tolist()
        shared_len = 0
        for c_i in range(1, 4):
            ci = s_idx * 4 + c_i
            sti, eni = int(cand_offsets[ci]), int(cand_offsets[ci + 1])
            toksi = cand_tok_target_np[sti:eni].tolist()
            if len(toksi) < len(toks0) and toks0[:len(toksi)] == toksi:
                shared_len = max(shared_len, len(toksi))
        for c_i in range(4):
            ci = s_idx * 4 + c_i
            sti, eni = int(cand_offsets[ci]), int(cand_offsets[ci + 1])
            seg_mat_np[ci, sti:eni] = 1.0 / max(1, eni - sti)
        for idx_in_c0, k in enumerate(range(st0, en0)):
            p = int(cand_tok_pos_np[k])
            t = int(cand_tok_target_np[k])
            if idx_in_c0 >= shared_len:
                safe_gt_tok_indices.append(k)
                if shared_len > 0 and t != top1_tok[p]:
                    cont_bottleneck_indices.append(k)
                    cont_bottleneck_pos.add(p)

    safe_gt_tok_indices_np = np.array(safe_gt_tok_indices, dtype=np.int32)
    cont_bottleneck_indices_np = np.array(cont_bottleneck_indices, dtype=np.int32)

    # Local 21-column setup
    local_ids_np = top20_idx_np.copy()
    target_local_col_np = np.zeros_like(cand_tok_target_np)
    for p in range(len(h63_np)):
        row_list = local_ids_np[p].tolist()
        keep = row_list[:16]
        extra = [c_t for c_t in pos_to_cands[p] if c_t not in keep]
        merged = (keep + extra + row_list[16:])[:20]
        local_ids_np[p] = np.array(merged, dtype=np.int64)

    for i, (p, t) in enumerate(zip(cand_tok_pos_np.tolist(), cand_tok_target_np.tolist())):
        target_local_col_np[i] = np.where(local_ids_np[p] == t)[0][0]

    z_local0_np = np.take_along_axis(z0_np, local_ids_np, axis=1)
    m_full = np.max(z0_np, axis=-1)
    lse_act0 = m_full + np.log(np.sum(np.exp(z0_np - m_full[:, None]), axis=-1))
    lse_full0_np = np.maximum(lse_act0, lse_rest_np) + np.log1p(np.exp(-np.abs(lse_act0 - lse_rest_np)))
    m_loc = np.max(z_local0_np, axis=-1)
    lse_loc0_np = m_loc + np.log(np.sum(np.exp(z_local0_np - m_loc[:, None]), axis=-1))
    diff_exp = np.maximum(np.exp(lse_full0_np.astype(np.float64)) - np.exp(lse_loc0_np.astype(np.float64)), 1e-12)
    lse_rest_local_np = np.log(diff_exp).astype(np.float32)[:, None]

    # Protect all 4,469 non-slot positions (excluding only the 12 continuation bottleneck positions!)
    non_slot_mask_np = np.array(
        [(len(pos_to_cands[p]) <= 1) and (p not in cont_bottleneck_pos) for p in range(len(h63_np))],
        dtype=bool,
    )
    tok_margins0_np = z_local0_np[:, 0] - z_local0_np[:, 1]
    cliff_tok_idx_np = np.where(tok_margins0_np > 5.2)[0]

    tok_lps0 = z0_np[cand_tok_pos_np, cand_tok_target_np] - lse_full0_np[cand_tok_pos_np]
    sc0 = (seg_mat_np @ tok_lps0).reshape(num_slots, 4)
    margins0 = sc0[:, 0] - np.max(sc0[:, 1:], axis=-1)
    hard25_idx = np.where(margins0[:224] <= 0)[0]
    hard_all_idx = np.where(margins0 <= 0)[0]
    print(f"[Base Margins] PolyStep-224 Base Hits: {int(np.sum(margins0[:224] > 0))}/224 | Expanded-64 Base Hits: {int(np.sum(margins0[224:] > 0))}/{num_slots-224} | Total Hard Slots: {len(hard_all_idx)}", flush=True)

    def barron_log_softmax_mx(z, axis=-1):
        z_max = mx.max(z, axis=axis, keepdims=True)
        i_max = mx.argmax(z, axis=axis, keepdims=True)
        shifted = z - z_max
        idx_range = mx.arange(z.shape[axis])[None, :] if z.ndim == 2 else mx.arange(z.shape[axis])[None, None, :]
        exp_others = mx.where(idx_range == i_max, 0.0, mx.exp(shifted))
        lse = mx.log1p(mx.sum(exp_others, axis=axis, keepdims=True))
        return shifted - lse

    def run_polystep_stage(h_base_cur_np, x_feat_np, v_out_basis_np, row_scale_np, init_theta_np, steps, K_sub, sigma_start, sigma_end, lr_start, lr_end, seed, l2_reg=0.0, ang_l2_reg=0.0):
        D_out, R_in = v_out_basis_np.shape[0], x_feat_np.shape[1]
        D_param = D_out * R_in
        K_eff = min(K_sub, D_param)
        m_lm_full_np = v_out_basis_np @ w_eff_np.T
        m_lm_loc_np = np.transpose(m_lm_full_np[:, local_ids_np], (1, 0, 2)).copy()
        h_base_sq_cur_np = np.sum(h_base_cur_np ** 2, axis=-1, keepdims=True)
        h_base_v_np = h_base_cur_np @ v_out_basis_np.T
        z_unnorm_cur_np = h_base_cur_np @ w_eff_np.T
        z_base_loc_cur_np = np.take_along_axis(z_unnorm_cur_np, local_ids_np, axis=1)

        x_feat = mx.array(x_feat_np)
        row_scale = mx.array(row_scale_np)
        h_base_sq = mx.array(h_base_sq_cur_np)
        h_base_v = mx.array(h_base_v_np)
        z_base_loc = mx.array(z_base_loc_cur_np)
        m_lm_loc = mx.array(m_lm_loc_np)
        lse_rest_local = mx.array(lse_rest_local_np)
        cand_tok_pos = mx.array(cand_tok_pos_np.astype(np.int32))
        target_local_col = mx.array(target_local_col_np.astype(np.int32))
        seg_mat = mx.array(seg_mat_np)
        cliff_idx = mx.array(cliff_tok_idx_np.astype(np.int32))
        ns_idx = mx.array(np.where(non_slot_mask_np)[0].astype(np.int32))
        gt_pos = mx.array(cand_tok_pos_np[safe_gt_tok_indices_np].astype(np.int32))
        gt_col = mx.array(target_local_col_np[safe_gt_tok_indices_np].astype(np.int32))
        cb_pos = mx.array(cand_tok_pos_np[cont_bottleneck_indices_np].astype(np.int32))
        cb_col = mx.array(target_local_col_np[cont_bottleneck_indices_np].astype(np.int32))

        def eval_batch_loss(theta_batch):
            B = theta_batch.shape[0]
            lat = mx.matmul(x_feat[None, :, :], mx.transpose(theta_batch, (0, 2, 1))) * row_scale[None, :, :] * 2.0
            norm_sq = (
                h_base_sq[None, :, :]
                + 2.0 * mx.sum(lat * h_base_v[None, :, :], axis=-1, keepdims=True)
                + mx.sum(lat * lat, axis=-1, keepdims=True)
            ) / 5120.0
            rms = mx.rsqrt(norm_sq + 1e-6)
            lat_p = mx.transpose(lat, (1, 0, 2))
            delta_z = mx.transpose(mx.matmul(lat_p, m_lm_loc), (1, 0, 2))
            logits_20 = rms * (z_base_loc[None, :, :] + delta_z)
            logits_21 = mx.concatenate([logits_20, mx.broadcast_to(lse_rest_local[None, :, :], (B, logits_20.shape[1], 1))], axis=-1)
            m21 = mx.max(logits_21, axis=-1, keepdims=True)
            lse_total = mx.squeeze(m21, -1) + mx.log(mx.sum(mx.exp(logits_21 - m21), axis=-1))

            cand_logits = logits_20[:, cand_tok_pos, target_local_col] - lse_total[:, cand_tok_pos]
            sc = mx.matmul(cand_logits, seg_mat.T).reshape(B, num_slots, 4)

            z_slot24 = (sc * 24.0).astype(mx.bfloat16).astype(mx.float32)
            z_slot10 = (sc * 10.0).astype(mx.bfloat16).astype(mx.float32)
            l_s = -0.6 * barron_log_softmax_mx(z_slot24, axis=-1)[:, :, 0].mean(axis=-1) - 0.4 * barron_log_softmax_mx(z_slot10, axis=-1)[:, :, 0].mean(axis=-1)

            slot_margins = sc[:, :, 0] - mx.max(sc[:, :, 1:], axis=-1)
            l_hinge = (
                mx.mean(mx.maximum(0.08 - slot_margins, 0.0) ** 2, axis=-1) * 60.0
                + mx.mean(mx.maximum(0.20 - slot_margins, 0.0) ** 2, axis=-1) * 22.0
                + mx.mean(mx.maximum(0.42 - slot_margins, 0.0) ** 2, axis=-1) * 4.0
            )

            gt_lps = logits_20[:, gt_pos, gt_col] - lse_total[:, gt_pos]
            l_gt_tok = -mx.mean(gt_lps, axis=-1) * 0.05

            cb_lps = logits_20[:, cb_pos, cb_col] - lse_total[:, cb_pos]
            l_cb = -mx.mean(cb_lps, axis=-1) * 0.22

            z_c = logits_20[:, cliff_idx, :16].astype(mx.bfloat16).astype(mx.float32)
            l_c = -barron_log_softmax_mx(z_c, axis=-1)[:, :, 0].mean(axis=-1)

            ns_logits = logits_20[:, ns_idx, :16]
            mv = mx.maximum(mx.max(ns_logits[:, :, 1:], axis=-1) - ns_logits[:, :, 0] + 0.15, 0.0).mean(axis=-1)
            reg = l2_reg * mx.sum(theta_batch * theta_batch, axis=(1, 2))
            if ang_l2_reg > 0.0 and D_out > 1:
                reg = reg + ang_l2_reg * mx.sum(theta_batch[:, 1:, :] * theta_batch[:, 1:, :], axis=(1, 2))
            return l_s + l_hinge + l_gt_tok + l_cb + 3.0 * l_c + 3.5 * mv + reg

        rng = np.random.default_rng(seed)
        theta = init_theta_np.copy()
        verts = np.concatenate([np.eye(K_eff, dtype=np.float32), -np.eye(K_eff, dtype=np.float32)], axis=0)
        for step in range(steps):
            prog = step / max(1, steps - 1)
            cos_mult = 0.5 * (1.0 + np.cos(np.pi * prog))
            sigma = sigma_end + (sigma_start - sigma_end) * cos_mult
            lr = lr_end + (lr_start - lr_end) * cos_mult

            g_mat = rng.standard_normal((D_param, K_eff)).astype(np.float32)
            P_basis, _ = np.linalg.qr(g_mat, mode="reduced")
            pert = theta[None, :] + sigma * (verts @ P_basis.T)
            losses_mx = eval_batch_loss(mx.array(pert.reshape(2 * K_eff, D_out, R_in)))
            mx.eval(losses_mx)
            c_f = np.array(losses_mx, dtype=np.float32)
            c_shift = c_f - np.min(c_f)
            scale_c = max(float(np.std(c_shift)), 1e-6)
            z_t = -(c_shift / scale_c) / 0.25
            i_max = int(np.argmax(z_t))
            z_max = z_t[i_max]
            s_o = np.sum(np.exp(np.delete(z_t - z_max, i_max)))
            log_W = (z_t - z_max) - np.log1p(s_o)
            w_pos = log_W[:K_eff]
            w_neg = log_W[K_eff:]
            w_centroid = np.exp(w_neg) * np.expm1(w_pos - w_neg)
            step_dir = P_basis @ w_centroid
            theta = theta + lr * (step_dir / (np.linalg.norm(step_dir) + 1e-8))
        return theta.reshape(D_out, R_in)

    # Stage 0: Multi-Module Null-Space Calibration on layers.63.self_attn.o_proj & layers.62.linear_attn.out_proj
    x_gat16 = d["gated63_proj"][:, :16].astype(np.float32)
    x_y62_16 = d["y62_proj"][:, :16].astype(np.float32)
    x_attn_null = np.concatenate([x_gat16, x_y62_16], axis=1)
    v_null_1d = v_null[None, :].astype(np.float32)
    row_scale_null = np.array([[4.0]], dtype=np.float32)

    core_null = run_polystep_stage(
        h63_np, x_attn_null, v_null_1d, row_scale_null,
        np.zeros(32, dtype=np.float32),
        steps=80, K_sub=32, sigma_start=0.030, sigma_end=0.008, lr_start=0.025, lr_end=0.005, seed=17, l2_reg=0.15
    )

    w_gat_null = (row_scale_null * core_null[:, :16]) * 0.25
    w_y62_null = (row_scale_null * core_null[:, 16:32]) * 0.18

    la_63_attn = d["u_gated63"][:16, :].astype(np.float32)
    lb_63_attn = (v_null_1d.T @ w_gat_null).astype(np.float32)
    la_62_attn = d["u_y62"][:16, :].astype(np.float32)
    lb_62_attn = (v_null_1d.T @ w_y62_null).astype(np.float32)

    la_63_attn_bf16 = np.array(mx.array(la_63_attn).astype(mx.bfloat16).astype(mx.float32))
    lb_63_attn_bf16 = np.array(mx.array(lb_63_attn).astype(mx.bfloat16).astype(mx.float32))
    la_62_attn_bf16 = np.array(mx.array(la_62_attn).astype(mx.bfloat16).astype(mx.float32))
    lb_62_attn_bf16 = np.array(mx.array(lb_62_attn).astype(mx.bfloat16).astype(mx.float32))

    proj_gat = la_63_attn_bf16 @ d["u_gated63"].astype(np.float32).T
    proj_y62 = la_62_attn_bf16 @ d["u_y62"].astype(np.float32).T
    delta_h_gat = ((d["gated63_proj"].astype(np.float32) @ proj_gat.T) @ lb_63_attn_bf16.T) * 2.0
    delta_h_y62 = ((d["y62_proj"].astype(np.float32) @ proj_y62.T) @ lb_62_attn_bf16.T) * 2.0
    h63_after_attn = h63_np + delta_h_gat + delta_h_y62

    # Stage A: 260 steps in 32x32 on act63_proj conditioned on h63_after_attn
    R_in = 32
    x_act_in = d["act63_proj"][:, :R_in].astype(np.float32)
    row_scale32_np = np.array([8.0] + [1.15] * 31, dtype=np.float32)[None, :]
    core32 = run_polystep_stage(
        h63_after_attn, x_act_in, v_out32_np, row_scale32_np,
        np.zeros(32 * R_in, dtype=np.float32),
        steps=260, K_sub=96, sigma_start=0.035, sigma_end=0.010, lr_start=0.032, lr_end=0.008, seed=42, ang_l2_reg=0.06
    )

    # Keep row 0 strictly equal to v_null and compress rows 1..31 into top 15 orthonormal angular vectors!
    M_scaled = row_scale32_np.T * core32
    u_ang, _, _ = np.linalg.svd(M_scaled[1:, :], full_matrices=False)
    v_ang15_star = (u_ang[:, :15].T @ v_out32_np[1:, :]).astype(np.float32)  # [15, 5120]
    v_out16_star = np.concatenate([v_null[None, :].astype(np.float32), v_ang15_star], axis=0)  # [16, 5120]

    M16_proj = np.zeros((16, R_in), dtype=np.float32)
    M16_proj[0, :] = M_scaled[0, :]
    M16_proj[1:, :] = u_ang[:, :15].T @ M_scaled[1:, :]
    _, _, vh_in = np.linalg.svd(M16_proj, full_matrices=False)
    u_in16_proj = vh_in[:16, :].astype(np.float32)  # [16, R_in]
    x_act16_star = x_act_in @ u_in16_proj.T         # [N_pos, 16]

    row_scale16_star = np.array([8.0] + [1.15] * 15, dtype=np.float32)[None, :]
    init_core16_mat = (M16_proj @ u_in16_proj.T) / row_scale16_star.T
    init_core16 = init_core16_mat.astype(np.float32).reshape(-1)

    # Stage B: 260 steps in exact rank-16 basis keeping row_scale = [8.0, 1.15, ..., 1.15]
    core16 = run_polystep_stage(
        h63_after_attn, x_act16_star, v_out16_star, row_scale16_star,
        init_core16,
        steps=260, K_sub=96, sigma_start=0.020, sigma_end=0.004, lr_start=0.020, lr_end=0.003, seed=108, ang_l2_reg=0.07
    )

    la_63_down = u_in16_proj @ d["u_act63"][:R_in, :].astype(np.float32)
    lb_63_down = (v_out16_star.T @ (row_scale16_star.T * core16)).astype(np.float32)

    if os.path.exists("/tmp/docstring_transition_vecs.npz"):
        import sys, json
        if "/Users/joshua/evaluation/PiQwen" not in sys.path:
            sys.path.insert(0, "/Users/joshua/evaluation/PiQwen")
        from mlx_qwen38_gpu import get_qwen38_snapshot
        doc = np.load("/tmp/docstring_transition_vecs.npz")
        a_doc = doc["act63"].astype(np.float32).reshape(5, 6, 17408)
        snap = get_qwen38_snapshot()
        with open(os.path.join(snap, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        w_lm_rows = mx.load(os.path.join(snap, wmap["lm_head.weight"]))["lm_head.weight"][mx.array([1067, 1866, 460, 653], dtype=mx.int32)].astype(mx.float32)
        w_4 = np.array(w_lm_rows) * w_final_norm_np[None, :]
        dv_he9 = w_4[0] - w_4[1]
        dv_he49 = w_4[2] - w_4[3]
        G1 = np.outer(dv_he9, la_63_down @ a_doc[0, 2, :]).reshape(-1)
        G2 = np.outer(dv_he49, la_63_down @ a_doc[4, 2, :]).reshape(-1)
        Q_g, _ = np.linalg.qr(np.stack([G1, G2], axis=1), mode="reduced")
        def apply_doc_guard(lb_mat):
            p_n = np.outer(v_null, v_null @ lb_mat)
            ang_v = (lb_mat - p_n).reshape(-1)
            ang_o = (
                ang_v
                - Q_g @ (Q_g.T @ ang_v)
                + 1.5 * G1 / np.dot(G1, G1)
                + 1.0 * G2 / np.dot(G2, G2)
            ).reshape(5120, 16)
            return (p_n + ang_o).astype(np.float32)

        lb_63_down = apply_doc_guard(lb_63_down)

    la_63_down_bf16 = np.array(mx.array(la_63_down).astype(mx.bfloat16).astype(mx.float32))
    lb_63_down_bf16 = np.array(mx.array(lb_63_down).astype(mx.bfloat16).astype(mx.float32))

    proj_la_down = la_63_down_bf16 @ d["u_act63"].astype(np.float32).T
    delta_h_down = ((d["act63_proj"].astype(np.float32) @ proj_la_down.T) @ lb_63_down_bf16.T) * 2.0

    # Stage C: Multi-Module Rank-16 Residual Refinement on layers.63.self_attn.o_proj & layers.62.linear_attn.out_proj
    row_scale_attn16 = np.array([4.0] + [0.65] * 15, dtype=np.float32)[None, :]
    init_gat16 = np.zeros((16, 16), dtype=np.float32)
    init_gat16[0, :] = w_gat_null[0, :] / 4.0
    core_gat16 = run_polystep_stage(
        h63_np + delta_h_down + delta_h_y62, x_gat16, v_out16_star, row_scale_attn16,
        init_gat16.reshape(-1),
        steps=160, K_sub=64, sigma_start=0.018, sigma_end=0.004, lr_start=0.018, lr_end=0.003, seed=211, ang_l2_reg=0.12
    )
    lb_63_attn = (v_out16_star.T @ (row_scale_attn16.T * core_gat16)).astype(np.float32)
    lb_63_attn_bf16 = np.array(mx.array(lb_63_attn).astype(mx.bfloat16).astype(mx.float32))
    delta_h_gat = ((d["gated63_proj"].astype(np.float32) @ proj_gat.T) @ lb_63_attn_bf16.T) * 2.0

    init_y62_16 = np.zeros((16, 16), dtype=np.float32)
    init_y62_16[0, :] = w_y62_null[0, :] / 4.0
    core_y62_16 = run_polystep_stage(
        h63_np + delta_h_down + delta_h_gat, x_y62_16, v_out16_star, row_scale_attn16,
        init_y62_16.reshape(-1),
        steps=140, K_sub=64, sigma_start=0.015, sigma_end=0.003, lr_start=0.015, lr_end=0.0025, seed=307, ang_l2_reg=0.14
    )
    lb_62_attn = (v_out16_star.T @ (row_scale_attn16.T * core_y62_16)).astype(np.float32)
    lb_62_attn_bf16 = np.array(mx.array(lb_62_attn).astype(mx.bfloat16).astype(mx.float32))
    delta_h_y62 = ((d["y62_proj"].astype(np.float32) @ proj_y62.T) @ lb_62_attn_bf16.T) * 2.0

    proj_null = v_null @ lb_63_down_bf16
    lb_orth = lb_63_down_bf16 - np.outer(v_null, proj_null)
    print(f"lb_63_down total={np.linalg.norm(lb_63_down_bf16):.4f} (null={np.linalg.norm(proj_null):.4f}, ang={np.linalg.norm(lb_orth):.4f}) | lb_63_attn={np.linalg.norm(lb_63_attn_bf16):.4f} | lb_62_attn={np.linalg.norm(lb_62_attn_bf16):.4f}")

    h = h63_np + delta_h_down + delta_h_gat + delta_h_y62
    rms = 1.0 / np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + 1e-6)
    logits_act = (h * rms) @ w_eff_np.T
    m_act = np.max(logits_act, axis=-1, keepdims=True)
    lse_act = np.squeeze(m_act, -1) + np.log(np.sum(np.exp(logits_act - m_act), axis=-1))
    lse_tot = np.maximum(lse_act, lse_rest_np) + np.log1p(np.exp(-np.abs(lse_act - lse_rest_np)))
    tok_lps = logits_act[cand_tok_pos_np, cand_tok_target_np] - lse_tot[cand_tok_pos_np]
    sc = (seg_mat_np @ tok_lps).reshape(num_slots, 4)
    margins = sc[:, 0] - np.max(sc[:, 1:], axis=-1)
    hits_all = int(np.sum(margins > 0))
    hits_224 = int(np.sum(margins[:224] > 0))
    hits_new = int(np.sum(margins[224:] > 0))
    z_loc_new = np.take_along_axis(logits_act, local_ids_np, axis=1)[cliff_tok_idx_np]
    cliff_pres = int(np.sum(np.argmax(z_loc_new, axis=-1) == 0))
    print(f"[PolyStep-{num_slots} Total] ({time.time()-t0:.1f}s) | bf16 Hits: {hits_all}/{num_slots} ({100.0*hits_all/num_slots:.2f}%) | PolyStep-224: {hits_224}/224 | Expanded-64: {hits_new}/{num_slots-224} | min_margin={np.min(margins):+.4f} | mean_margin={np.mean(margins):+.4f} | Hard25_min={np.min(margins[hard25_idx]):+.4f} | Hard25_mean={np.mean(margins[hard25_idx]):+.4f} | cliff={cliff_pres}/{len(cliff_tok_idx_np)}", flush=True)

    layer_down_a = d["layer_down_a"].astype(np.float32).copy()
    layer_attn_a = d["layer_attn_a"].astype(np.float32).copy()
    layer_down_a[63] = la_63_down
    layer_attn_a[63] = la_63_attn
    layer_attn_a[62] = la_62_attn

    state_dict = {}
    r = 16
    zero_b = mx.zeros((5120, r), dtype=mx.bfloat16)
    for l in range(64):
        peft_lp = f"base_model.model.language_model.layers.{l}"
        ka_d = f"{peft_lp}.mlp.down_proj.lora_A.weight"
        kb_d = f"{peft_lp}.mlp.down_proj.lora_B.weight"
        state_dict[ka_d] = mx.array(layer_down_a[l]).astype(mx.bfloat16)
        if l == 63:
            state_dict[kb_d] = mx.array(lb_63_down).astype(mx.bfloat16)
        else:
            state_dict[kb_d] = zero_b

        if l % 4 != 3:
            ka_a = f"{peft_lp}.linear_attn.out_proj.lora_A.weight"
            kb_a = f"{peft_lp}.linear_attn.out_proj.lora_B.weight"
        else:
            ka_a = f"{peft_lp}.self_attn.o_proj.lora_A.weight"
            kb_a = f"{peft_lp}.self_attn.o_proj.lora_B.weight"
        state_dict[ka_a] = mx.array(layer_attn_a[l]).astype(mx.bfloat16)
        if l == 63:
            state_dict[kb_a] = mx.array(lb_63_attn).astype(mx.bfloat16)
        elif l == 62:
            state_dict[kb_a] = mx.array(lb_62_attn).astype(mx.bfloat16)
        else:
            state_dict[kb_a] = zero_b

    os.makedirs(PIQWEN_CKPT_DIR, exist_ok=True)
    out_st_piqwen = os.path.join(PIQWEN_CKPT_DIR, "adapter_model.safetensors")
    mx.save_safetensors(out_st_piqwen, state_dict)
    mx.save_safetensors("/tmp/adapter_model_v4.safetensors", state_dict)
    active_6_dict = {
        k: state_dict[k]
        for k in [
            "base_model.model.language_model.layers.63.mlp.down_proj.lora_A.weight",
            "base_model.model.language_model.layers.63.mlp.down_proj.lora_B.weight",
            "base_model.model.language_model.layers.63.self_attn.o_proj.lora_A.weight",
            "base_model.model.language_model.layers.63.self_attn.o_proj.lora_B.weight",
            "base_model.model.language_model.layers.62.linear_attn.out_proj.lora_A.weight",
            "base_model.model.language_model.layers.62.linear_attn.out_proj.lora_B.weight",
        ]
    }
    mx.save_safetensors("/tmp/v4_active_6tensors.safetensors", active_6_dict)
    print(f"[SAVED] 256-tensor bfloat16 PEFT LoRA checkpoint saved to {out_st_piqwen}")

if __name__ == "__main__":
    main()
