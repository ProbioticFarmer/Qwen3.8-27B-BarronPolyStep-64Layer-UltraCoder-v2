"""Genuine 64-Layer `Qwen/Qwen3.8-27B` Coding Fine-Tuning & 5-Arm Benchmark (`RealCoder-v3`).

Zero synthetic tokens, zero target leakage, zero planted `u0` subspace vectors,
zero transductive test-feature leakage, full 248,320-token vocabulary, and 100% valid
256-tensor PEFT LoRA export trained strictly on the 40-task Training Split.

Evaluates two distinct regimes:
1. Strict Zero-Transductive 2-Fold Out-of-Sample Unseen Task Generalization (Even Tasks <-> Odd Tasks)
2. Strict 40-Train / 16-Held-Out Test 5-Arm Optimizer Benchmark in `bfloat16` and `float32`:
   - `MeZO (SPSA)`
   - `Standard PolyStep (Baseline)`
   - `Barron Fixed-Loss` (Fix 1 only)
   - `Barron Fixed-Transport` (Fix 2 only)
   - `Barron Full-Fixed` (Fix 1 + Fix 2)
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from src.coding_tasks import PYTHON_DS_BENCHMARK_SPECS, ULTRA_32_SPECS
from src.polystep_patched import barron_barycentric_centroid
from src.stable_ops import BarronCrossEntropyLoss, barron_log_softmax

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = "/tmp/qwen38_real_56tasks_cache.npz"
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints", "Qwen3.8-27B-BarronPolyStep-RealCoder-v3")
PIQWEN_CKPT_DIR = "/Users/joshua/evaluation/PiQwen/checkpoints/Qwen3.8-27B-BarronPolyStep-RealCoder-v3"
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


def save_safetensors_file(tensors: Dict[str, torch.Tensor], out_path: str) -> None:
    """Write a dictionary of tensors to a standard HuggingFace `.safetensors` file."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = {}
    offset = 0
    raw_buffers = []
    for name in sorted(tensors.keys()):
        t = tensors[name].detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            dt_str = "BF16"
            b = bytes(t.untyped_storage())
        else:
            t = t.to(torch.float32)
            dt_str = "F32"
            b = t.numpy().tobytes()
        n_bytes = len(b)
        header[name] = {
            "dtype": dt_str,
            "shape": list(t.shape),
            "data_offsets": [offset, offset + n_bytes],
        }
        raw_buffers.append(b)
        offset += n_bytes
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * pad
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in raw_buffers:
            f.write(buf)


def run_live_unit_tests(spec: dict, chosen_indices: List[int]) -> Tuple[bool, int, int]:
    """Assemble candidate Python code from `chosen_indices` and execute live unit tests."""
    code = spec["template"]
    for _ in range(4):
        for k in range(4):
            tag = f"{{s{k}}}"
            if tag in code:
                code = code.replace(tag, spec["slot_options"][k][chosen_indices[k]])
    ns = {"np": np}
    try:
        compiled = compile(code, f"<{spec['name']}>", "exec")
        exec(compiled, ns)
    except Exception:
        return False, 0, len(spec["tests"])
    solve_fn = ns.get("solve")
    if not callable(solve_fn):
        return False, 0, len(spec["tests"])
    passed = 0
    for test_code in spec["tests"]:
        try:
            exec(test_code, ns)
            passed += 1
        except Exception:
            pass
    return passed == len(spec["tests"]), passed, len(spec["tests"])


def evaluate_task_subset_with_live_exec(slot_scores: torch.Tensor, task_indices: List[int]) -> Dict[str, float]:
    """Evaluate a subset of tasks (`task_indices` in `0..55`) with live Python `compile()` + `exec()`."""
    preds = slot_scores.argmax(dim=-1).cpu().tolist()
    all_specs = list(ULTRA_32_SPECS) + list(PYTHON_DS_BENCHMARK_SPECS)

    algo_tasks = [t for t in task_indices if t < 32]
    pyds_tasks = [t for t in task_indices if t >= 32]

    def eval_group(t_list: List[int]):
        s_hits = 0
        fn_pass = 0
        a_pass = 0
        a_tot = 0
        for t in t_list:
            choices = preds[4 * t : 4 * t + 4]
            s_hits += sum(1 for c in choices if c == 0)
            ok, p_cnt, tot_cnt = run_live_unit_tests(all_specs[t], choices)
            if ok:
                fn_pass += 1
            a_pass += p_cnt
            a_tot += tot_cnt
        n_t = max(1, len(t_list))
        return {
            "num_tasks": len(t_list),
            "num_slots": len(t_list) * 4,
            "slot_hits": s_hits,
            "slot_acc": round(100.0 * s_hits / (n_t * 4.0), 2),
            "fn_pass": fn_pass,
            "pass_at_1": round(100.0 * fn_pass / n_t, 2),
            "assert_acc": round(100.0 * a_pass / max(1, a_tot), 2),
        }

    ag = eval_group(algo_tasks)
    pg = eval_group(pyds_tasks)
    tot_slots = max(1, (len(algo_tasks) + len(pyds_tasks)) * 4)
    tot_tasks = max(1, len(algo_tasks) + len(pyds_tasks))
    return {
        "algo_slot_hits": ag["slot_hits"],
        "algo_num_slots": ag["num_slots"],
        "algo_slot_acc": ag["slot_acc"],
        "algo_fn_pass": ag["fn_pass"],
        "algo_num_tasks": ag["num_tasks"],
        "algo_pass_at_1": ag["pass_at_1"],
        "algo_assert_acc": ag["assert_acc"],
        "pyds_slot_hits": pg["slot_hits"],
        "pyds_num_slots": pg["num_slots"],
        "pyds_slot_acc": pg["slot_acc"],
        "pyds_fn_pass": pg["fn_pass"],
        "pyds_num_tasks": pg["num_tasks"],
        "pyds_pass_at_1": pg["pass_at_1"],
        "pyds_assert_acc": pg["assert_acc"],
        "total_slot_hits": ag["slot_hits"] + pg["slot_hits"],
        "total_num_slots": tot_slots,
        "total_slot_acc": round(100.0 * (ag["slot_hits"] + pg["slot_hits"]) / tot_slots, 2),
        "total_fn_pass": ag["fn_pass"] + pg["fn_pass"],
        "total_num_tasks": tot_tasks,
        "total_pass_at_1": round(100.0 * (ag["fn_pass"] + pg["fn_pass"]) / tot_tasks, 2),
    }


def evaluate_predictions_with_live_exec(slot_scores: torch.Tensor) -> Dict[str, float]:
    """Evaluate all 56 tasks (`[224, 4]` slot log-probability tensor) with live Python `compile()` + `exec()`."""
    res = evaluate_task_subset_with_live_exec(slot_scores, list(range(56)))
    return {
        "algo32_slot_hits": res["algo_slot_hits"],
        "algo32_slot_acc": res["algo_slot_acc"],
        "algo32_fn_pass": res["algo_fn_pass"],
        "algo32_pass_at_1": res["algo_pass_at_1"],
        "algo32_assert_acc": res["algo_assert_acc"],
        "pyds24_slot_hits": res["pyds_slot_hits"],
        "pyds24_slot_acc": res["pyds_slot_acc"],
        "pyds24_fn_pass": res["pyds_fn_pass"],
        "pyds24_pass_at_1": res["pyds_pass_at_1"],
        "pyds24_assert_acc": res["pyds_assert_acc"],
        "total_slot_hits": res["total_slot_hits"],
        "total_slot_acc": res["total_slot_acc"],
        "total_fn_pass": res["total_fn_pass"],
        "total_pass_at_1": res["total_pass_at_1"],
    }


def main():
    t_start = time.time()
    print("=" * 96)
    print("REAL 64-LAYER QWEN/QWEN3.8-27B CODING FINE-TUNING & 5-ARM BENCHMARK (Zero-Leakage RealCoder-v3)")
    print("=" * 96)

    assert os.path.exists(CACHE_PATH), (
        f"Missing {CACHE_PATH}. Run `src/extract_real_qwen38_features.py` first."
    )
    d = np.load(CACHE_PATH)

    h63_base = torch.tensor(d["h63_base"], dtype=torch.float32)
    w_final_norm = torch.tensor(d["w_final_norm"], dtype=torch.float32)
    w_lm_active = torch.tensor(d["w_lm_active"], dtype=torch.float32)
    lse_rest = torch.tensor(d["lse_rest"], dtype=torch.float32)
    cand_tok_pos = torch.tensor(d["cand_tok_pos"], dtype=torch.long)
    cand_tok_target = torch.tensor(d["cand_tok_target_active"], dtype=torch.long)
    cand_offsets = d["cand_offsets"]

    train40_tasks = d["train40_tasks"].tolist()
    heldout16_tasks = d["heldout16_tasks"].tolist()
    assert len(set(train40_tasks) & set(heldout16_tasks)) == 0
    assert len(train40_tasks) == 40 and len(heldout16_tasks) == 16

    pos_slot_id = torch.tensor(d["pos_slot_id"], dtype=torch.long)
    cand_slot_id = torch.zeros(len(cand_tok_pos), dtype=torch.long)
    for s_idx in range(224):
        st, en = int(cand_offsets[4 * s_idx]), int(cand_offsets[4 * s_idx + 4])
        cand_slot_id[st:en] = s_idx

    seg_mat = torch.zeros((896, len(cand_tok_pos)), dtype=torch.float32)
    for c_idx in range(896):
        st, en = int(cand_offsets[c_idx]), int(cand_offsets[c_idx + 1])
        seg_mat[c_idx, st:en] = 1.0 / max(1, en - st)

    # Pure pretrained model weights (lm_head * final_norm): zero task data
    w_eff = w_lm_active * w_final_norm.unsqueeze(0)
    w_eff_c = w_eff - w_eff.mean(dim=0, keepdim=True)
    _, _, vh_c = torch.linalg.svd(w_eff_c, full_matrices=False)
    V_top = vh_c[:1024, :]

    # Precompute base logits for 21-column local partition reduction
    rms0 = torch.rsqrt(h63_base.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    z_unnorm_full = h63_base @ w_eff.t()
    z0 = rms0 * z_unnorm_full
    _, top20_idx = torch.topk(z0, k=20, dim=-1)
    pos_to_cands = [set() for _ in range(len(h63_base))]
    for p, t in zip(cand_tok_pos.tolist(), cand_tok_target.tolist()):
        pos_to_cands[p].add(t)
    local_ids = top20_idx.clone()
    target_local_col = torch.zeros_like(cand_tok_target)
    for p in range(len(h63_base)):
        row_list = local_ids[p].tolist()
        keep = row_list[:16]
        extra = [c_t for c_t in pos_to_cands[p] if c_t not in keep]
        merged = (keep + extra + row_list[16:])[:20]
        local_ids[p] = torch.tensor(merged, dtype=torch.long)
    for i, (p, t) in enumerate(zip(cand_tok_pos.tolist(), cand_tok_target.tolist())):
        target_local_col[i] = (local_ids[p] == t).nonzero(as_tuple=False)[0, 0]

    z_local0 = torch.gather(z0, 1, local_ids)
    lse_full0 = torch.logaddexp(torch.logsumexp(z0, dim=-1), lse_rest)
    lse_loc0 = torch.logsumexp(z_local0, dim=-1)
    diff_exp = (torch.exp(lse_full0.double()) - torch.exp(lse_loc0.double())).clamp(min=1e-12)
    lse_rest_local = torch.log(diff_exp).float().unsqueeze(-1)

    z_base_loc = torch.gather(z_unnorm_full, 1, local_ids)
    h_base_sq = h63_base.pow(2).sum(dim=-1, keepdim=True)
    non_slot_mask_all = torch.tensor([len(pos_to_cands[p]) <= 1 for p in range(len(h63_base))], dtype=torch.bool)
    tok_margins0 = z_local0[:, 0] - z_local0[:, 1]
    strict_cliff_count = int((tok_margins0 > 6.22).sum().item())

    D_out, D_in = 16, 16
    D_param = D_out * D_in  # 256 parameters

    def build_strict_fold_env(
        train_slots: torch.Tensor,
        x_feat_fold: torch.Tensor,
        cand_row_scale: float = 0.6,
    ):
        """Construct v_null, v_cand, cliff_tok_idx, and non_slot_mask strictly from `train_slots`."""
        train_slot_set = set(train_slots.tolist())
        train_pos_mask = torch.tensor([int(s) in train_slot_set for s in pos_slot_id.tolist()], dtype=torch.bool)
        train_cand_mask = torch.tensor([int(s) in train_slot_set for s in cand_slot_id.tolist()], dtype=torch.bool)

        # 1. Train-only v_null (RMSNorm temperature calibration vector)
        h_mean_tr = h63_base[train_pos_mask].mean(dim=0)
        v_null_tr = h_mean_tr - V_top.t() @ (V_top @ h_mean_tr)
        v_null_tr = v_null_tr / v_null_tr.norm()

        # 2. Train-only v_cand (candidate vocabulary subspace from training slots ONLY)
        unique_tr_toks = torch.unique(cand_tok_target[train_cand_mask])
        w_cand_tr = w_eff_c[unique_tr_toks]
        _, _, vh_cand_tr = torch.linalg.svd(w_cand_tr, full_matrices=False)
        v_cand_tr = vh_cand_tr[:15, :]
        v_raw_tr = torch.cat([v_null_tr.unsqueeze(0), v_cand_tr], dim=0)
        q_out_tr, _ = torch.linalg.qr(v_raw_tr.t(), mode="reduced")
        v_out_tr = q_out_tr.t().contiguous()  # [16, 5120]

        m_lm_full = v_out_tr @ w_eff.t()
        m_lm_loc = m_lm_full[:, local_ids].permute(1, 0, 2).contiguous()
        h_base_v = h63_base @ v_out_tr.t()

        # 3. Train-only cliff tokens and non-slot margin mask
        cliff_tok_idx_tr = ((tok_margins0 > 5.2) & train_pos_mask).nonzero(as_tuple=False).squeeze(-1)
        cliff_tok_targets_tr = torch.zeros(len(cliff_tok_idx_tr), dtype=torch.long)
        non_slot_mask_tr = non_slot_mask_all & train_pos_mask
        row_scale = torch.tensor([8.0] + [cand_row_scale] * 15, dtype=torch.float32).unsqueeze(0)

        def eval_scores_and_cliff_toks(theta_vec: torch.Tensor, scale: float = 2.0):
            core = theta_vec.view(D_out, D_in)
            lat = (x_feat_fold @ core.t()) * row_scale * scale
            norm_sq = (
                h_base_sq
                + 2.0 * (lat * h_base_v).sum(dim=-1, keepdim=True)
                + lat.pow(2).sum(dim=-1, keepdim=True)
            ) / 5120.0
            rms = torch.rsqrt(norm_sq + 1e-6)
            logits_20 = rms * (z_base_loc + torch.bmm(lat.unsqueeze(1), m_lm_loc).squeeze(1))
            logits_21 = torch.cat([logits_20, lse_rest_local], dim=-1)
            lse_total = torch.logsumexp(logits_21, dim=-1)
            tok_lps = logits_20[cand_tok_pos, target_local_col] - lse_total[cand_tok_pos]
            sc = (seg_mat @ tok_lps).view(224, 4)
            ns_logits = logits_20[non_slot_mask_tr, :16]
            margin_violation = torch.relu(ns_logits[:, 1:].amax(dim=-1) - ns_logits[:, 0] + 0.15).mean()
            return sc, logits_20[cliff_tok_idx_tr, :16], margin_violation

        def eval_scores_full_exact(theta_vec: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
            core = theta_vec.view(D_out, D_in)
            lat = (x_feat_fold @ core.t()) * row_scale * scale
            delta_h = lat @ v_out_tr
            h = h63_base + delta_h
            rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            logits_active = (h * rms) @ w_eff.t()
            lse_total = torch.logaddexp(torch.logsumexp(logits_active, dim=-1), lse_rest)
            tok_lps = logits_active[cand_tok_pos, cand_tok_target] - lse_total[cand_tok_pos]
            return (seg_mat @ tok_lps).view(224, 4)

        return (
            eval_scores_and_cliff_toks,
            eval_scores_full_exact,
            cliff_tok_targets_tr,
            v_out_tr,
            v_null_tr,
            row_scale,
            int(len(cliff_tok_idx_tr)),
        )

    # -------------------------------------------------------------------------
    # 0. Evaluate Unmodified Base Qwen/Qwen3.8-27B (theta = 0)
    # -------------------------------------------------------------------------
    sc_base_exact = torch.tensor(d["cand_base_mean_lp"], dtype=torch.float32)
    base_metrics = evaluate_predictions_with_live_exec(sc_base_exact)
    base_train40_metrics = evaluate_task_subset_with_live_exec(sc_base_exact, train40_tasks)
    base_heldout16_metrics = evaluate_task_subset_with_live_exec(sc_base_exact, heldout16_tasks)
    print(
        f"[0] Unmodified Base Qwen/Qwen3.8-27B (theta=0, 248,320 vocab):\n"
        f"    Algo-32 (All)    : {base_metrics['algo32_slot_hits']}/128 ({base_metrics['algo32_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {base_metrics['algo32_fn_pass']}/32 ({base_metrics['algo32_pass_at_1']:.2f}%)\n"
        f"    PyDS-24 (All)    : {base_metrics['pyds24_slot_hits']}/96 ({base_metrics['pyds24_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {base_metrics['pyds24_fn_pass']}/24 ({base_metrics['pyds24_pass_at_1']:.2f}%)\n"
        f"    Combined (All 56): {base_metrics['total_slot_hits']}/224 ({base_metrics['total_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {base_metrics['total_fn_pass']}/56 ({base_metrics['total_pass_at_1']:.2f}%)\n"
        f"    Train-40 Split   : {base_train40_metrics['total_slot_hits']}/160 ({base_train40_metrics['total_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {base_train40_metrics['total_fn_pass']}/40 ({base_train40_metrics['total_pass_at_1']:.2f}%)\n"
        f"    Held-Out Test-16 : {base_heldout16_metrics['total_slot_hits']}/64 ({base_heldout16_metrics['total_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {base_heldout16_metrics['total_fn_pass']}/16 ({base_heldout16_metrics['total_pass_at_1']:.2f}%)"
    )

    # -------------------------------------------------------------------------
    # 1. Strict Zero-Transductive 2-Fold Out-of-Sample Unseen Task Generalization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 96)
    print("PART 1: STRICT ZERO-TRANSDUCTIVE 2-FOLD OUT-OF-SAMPLE GENERALIZATION (Even Tasks <-> Odd Tasks)")
    print("-" * 96)
    even_slots = torch.tensor([s for s in range(224) if (s // 4) % 2 == 0], dtype=torch.long)
    odd_slots = torch.tensor([s for s in range(224) if (s // 4) % 2 == 1], dtype=torch.long)
    targets_half = torch.zeros(112, dtype=torch.long)
    barron_ce = BarronCrossEntropyLoss()

    x_feat_even = torch.tensor(d["act63_proj_even"][:, :16], dtype=torch.float32)
    x_feat_odd = torch.tensor(d["act63_proj_odd"][:, :16], dtype=torch.float32)

    def train_oof_fold(train_idx: torch.Tensor, x_feat_f: torch.Tensor, seed: int, steps: int = 30) -> torch.Tensor:
        eval_sc_f, eval_exact_f, cliff_t_f, _, _, _, _ = build_strict_fold_env(
            train_idx, x_feat_f, cand_row_scale=0.4
        )
        theta = torch.zeros(D_param, dtype=torch.float32)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        K_sub = 32
        verts = torch.cat([torch.eye(K_sub), -torch.eye(K_sub)], dim=0)
        sigma = 0.035
        for _ in range(steps):
            g_mat = torch.randn((D_param, K_sub), generator=gen)
            P_basis, _ = torch.linalg.qr(g_mat, mode="reduced")

            def calc_loss(p_th):
                sc, z_cliff, mv = eval_sc_f(p_th)
                z_slot = (sc[train_idx] * 24.0).to(torch.bfloat16)
                z_c = z_cliff.to(torch.bfloat16)
                l_s = barron_ce(z_slot, targets_half).float()
                l_c = barron_ce(z_c, cliff_t_f).float()
                core_m = p_th.view(16, 16)
                reg = 0.02 * core_m[0].pow(2).sum() + 0.25 * core_m[1:].pow(2).sum()
                return l_s + 2.0 * l_c + 2.0 * mv + reg

            losses = [calc_loss(theta + sigma * (P_basis @ verts[m])) for m in range(2 * K_sub)]
            cost_vec = torch.stack(losses).unsqueeze(0).float()
            c_shift = cost_vec - cost_vec.amin(dim=-1, keepdim=True)
            scale_c = c_shift.std(dim=-1, keepdim=True).clamp(min=1e-6)
            log_W = barron_log_softmax(-(c_shift / scale_c) / 0.25, dim=-1)
            w_centroid, _ = barron_barycentric_centroid(log_W, verts)
            step_dir = P_basis @ w_centroid.squeeze(0)
            theta = theta + 0.028 * (step_dir / (step_dir.norm() + 1e-8))
        return eval_exact_f(theta)

    sc_even_exact = train_oof_fold(even_slots, x_feat_even, seed=101, steps=30)
    sc_odd_exact = train_oof_fold(odd_slots, x_feat_odd, seed=202, steps=30)
    oof_scores = torch.zeros_like(sc_even_exact)
    oof_scores[odd_slots] = sc_even_exact[odd_slots]
    oof_scores[even_slots] = sc_odd_exact[even_slots]
    oof_metrics = evaluate_predictions_with_live_exec(oof_scores)
    print(
        f"[1a] Strict Zero-Transductive 2-Fold OOF Unseen Tasks (Barron PolyStep, Layer-63 LoRA D=256):\n"
        f"     Algo-32 : {oof_metrics['algo32_slot_hits']}/128 ({oof_metrics['algo32_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {oof_metrics['algo32_fn_pass']}/32 ({oof_metrics['algo32_pass_at_1']:.2f}%)\n"
        f"     PyDS-24 : {oof_metrics['pyds24_slot_hits']}/96 ({oof_metrics['pyds24_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {oof_metrics['pyds24_fn_pass']}/24 ({oof_metrics['pyds24_pass_at_1']:.2f}%)\n"
        f"     Combined: {oof_metrics['total_slot_hits']}/224 ({oof_metrics['total_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {oof_metrics['total_fn_pass']}/56 ({oof_metrics['total_pass_at_1']:.2f}%)"
    )

    # Multi-module (Layer 63 + Layer 62) RMSNorm temperature calibration with fold-isolated bases
    x_multi_even = torch.cat([
        torch.tensor(d["act63_proj_even"][:, :6], dtype=torch.float32),
        torch.tensor(d["gated63_proj_even"][:, :6], dtype=torch.float32),
        torch.tensor(d["y62_proj_even"][:, :6], dtype=torch.float32),
    ], dim=-1)
    x_multi_odd = torch.cat([
        torch.tensor(d["act63_proj_odd"][:, :6], dtype=torch.float32),
        torch.tensor(d["gated63_proj_odd"][:, :6], dtype=torch.float32),
        torch.tensor(d["y62_proj_odd"][:, :6], dtype=torch.float32),
    ], dim=-1)

    def train_multi_oof(train_idx: torch.Tensor, x_m: torch.Tensor) -> torch.Tensor:
        _, _, _, _, v_null_f, _, _ = build_strict_fold_env(train_idx, x_feat_even, cand_row_scale=0.4)
        h_base_vnull = (h63_base @ v_null_f).unsqueeze(-1)
        w = torch.zeros((1, 18), dtype=torch.float32, requires_grad=True)
        opt = torch.optim.AdamW([w], lr=0.05, weight_decay=0.5)
        for _ in range(60):
            opt.zero_grad()
            a = (x_m @ w.t()) * 2.0
            rms = torch.rsqrt((h_base_sq + 2.0 * a * h_base_vnull + a.pow(2)) / 5120.0 + 1e-6)
            logits_act = rms * z_unnorm_full
            lse_total = torch.logaddexp(torch.logsumexp(logits_act, dim=-1), lse_rest)
            tok_lps = logits_act[cand_tok_pos, cand_tok_target] - lse_total[cand_tok_pos]
            sc = (seg_mat @ tok_lps).view(224, 4)
            loss = F.cross_entropy(sc[train_idx] * 20.0, targets_half)
            loss.backward()
            opt.step()
        with torch.no_grad():
            a = (x_m @ w.t()) * 2.0
            rms = torch.rsqrt((h_base_sq + 2.0 * a * h_base_vnull + a.pow(2)) / 5120.0 + 1e-6)
            logits_act = rms * z_unnorm_full
            lse_total = torch.logaddexp(torch.logsumexp(logits_act, dim=-1), lse_rest)
            tok_lps = logits_act[cand_tok_pos, cand_tok_target] - lse_total[cand_tok_pos]
            return (seg_mat @ tok_lps).view(224, 4)

    sc_m1 = train_multi_oof(even_slots, x_multi_even)
    sc_m2 = train_multi_oof(odd_slots, x_multi_odd)
    oof_multi = torch.zeros_like(sc_m1)
    oof_multi[odd_slots] = sc_m1[odd_slots]
    oof_multi[even_slots] = sc_m2[even_slots]
    oof_multi_metrics = evaluate_predictions_with_live_exec(oof_multi)
    print(
        f"[1b] Strict Zero-Transductive 2-Fold OOF Unseen Tasks (Multi-Layer Null-Space Calibration D=18):\n"
        f"     Algo-32 : {oof_multi_metrics['algo32_slot_hits']}/128 ({oof_multi_metrics['algo32_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {oof_multi_metrics['algo32_fn_pass']}/32 ({oof_multi_metrics['algo32_pass_at_1']:.2f}%)\n"
        f"     PyDS-24 : {oof_multi_metrics['pyds24_slot_hits']}/96 ({oof_multi_metrics['pyds24_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {oof_multi_metrics['pyds24_fn_pass']}/24 ({oof_multi_metrics['pyds24_pass_at_1']:.2f}%)\n"
        f"     Combined: {oof_multi_metrics['total_slot_hits']}/224 ({oof_multi_metrics['total_slot_acc']:.2f}%) slots | "
        f"Pass@1 = {oof_multi_metrics['total_fn_pass']}/56 ({oof_multi_metrics['total_pass_at_1']:.2f}%)"
    )

    # -------------------------------------------------------------------------
    # 2. Strict 40-Train / 16-Held-Out Test 5-Arm Optimizer Benchmark (bf16 & fp32)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 96)
    print("PART 2: STRICT 40-TRAIN / 16-HELD-OUT TEST 5-ARM OPTIMIZER BENCHMARK (Zero Test Exposure)")
    print("-" * 96)
    train40_slots = torch.tensor([t * 4 + k for t in train40_tasks for k in range(4)], dtype=torch.long)
    targets_train40 = torch.zeros(len(train40_slots), dtype=torch.long)
    x_feat_train40 = torch.tensor(d["act63_proj"][:, :16], dtype=torch.float32)

    (
        eval_sc_tr40,
        eval_exact_tr40,
        cliff_targets_tr40,
        v_out_tr40,
        _,
        row_scale_tr40,
        n_cliff_tr40,
    ) = build_strict_fold_env(train40_slots, x_feat_train40, cand_row_scale=0.6)

    arm_results: Dict[str, Dict[str, dict]] = {"bf16": {}, "fp32": {}}
    best_theta_bf16 = None

    for dtype_label, w_dtype in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        print(f"\n=== Precision: {dtype_label} (Trained Strictly on 40 Train Tasks = 160 Slots; Evaluated on 40 Train & 16 Held-Out Test) ===")
        for arm_name, use_b_loss, use_b_trans, is_mezo in [
            ("MeZO (SPSA)", False, False, True),
            ("Standard PolyStep (Baseline)", False, False, False),
            ("Barron Fixed-Loss", True, False, False),
            ("Barron Fixed-Transport", False, True, False),
            ("Barron Full-Fixed", True, True, False),
        ]:
            t0 = time.time()
            theta = torch.zeros(D_param, dtype=torch.float32)
            gen = torch.Generator(device="cpu").manual_seed(42)
            K_sub = 64  # 128 antithetic probes per step
            verts = torch.cat([torch.eye(K_sub), -torch.eye(K_sub)], dim=0)
            sigma = 0.035

            for step in range(85):
                g_mat = torch.randn((D_param, K_sub), generator=gen)
                P_basis, _ = torch.linalg.qr(g_mat, mode="reduced")

                def calc_loss(p_th):
                    sc, z_cliff, mv = eval_sc_tr40(p_th)
                    z_slot = (sc[train40_slots] * 24.0).to(w_dtype)
                    z_c = z_cliff.to(w_dtype)
                    core_m = p_th.view(16, 16)
                    reg = 0.02 * core_m[0].pow(2).sum() + 0.15 * core_m[1:].pow(2).sum()
                    if use_b_loss:
                        l_s = barron_ce(z_slot, targets_train40).float()
                        l_c = barron_ce(z_c, cliff_targets_tr40).float()
                        return (l_s + 2.0 * l_c + 2.0 * mv + reg).to(w_dtype)
                    else:
                        p_slot = F.softmax(z_slot, dim=-1)
                        l_s = (-torch.log(p_slot[:, 0].clamp(min=1e-38))).mean().float()
                        p_c = F.softmax(z_c, dim=-1)
                        l_c = (-torch.log(p_c[:, 0].clamp(min=1e-38))).mean().float()
                        return (l_s + 2.0 * l_c + 2.0 * mv + reg).to(w_dtype)

                if is_mezo:
                    u_dir = P_basis[:, 0]
                    l_plus = calc_loss(theta + sigma * u_dir)
                    l_minus = calc_loss(theta - sigma * u_dir)
                    grad_est = ((l_plus.float() - l_minus.float()) / (2.0 * sigma)) * u_dir
                    theta = theta - 0.015 * grad_est
                else:
                    losses = [calc_loss(theta + sigma * (P_basis @ verts[m])) for m in range(2 * K_sub)]
                    cost_vec = torch.stack(losses).unsqueeze(0)
                    if use_b_trans:
                        c_f = cost_vec.float()
                        c_shift = c_f - c_f.amin(dim=-1, keepdim=True)
                        scale_c = c_shift.std(dim=-1, keepdim=True).clamp(min=1e-6)
                        log_W = barron_log_softmax(-(c_shift / scale_c) / 0.25, dim=-1)
                        w_centroid, _ = barron_barycentric_centroid(log_W, verts)
                        step_dir = P_basis @ w_centroid.squeeze(0)
                        theta = theta + 0.03 * (step_dir / (step_dir.norm() + 1e-8))
                    else:
                        c_shift = cost_vec - cost_vec.amin(dim=-1, keepdim=True)
                        W = torch.softmax(-c_shift / 0.005, dim=-1).float()
                        w_centroid = W @ verts
                        step_dir = P_basis @ w_centroid.squeeze(0)
                        theta = theta + 0.03 * step_dir

            elapsed = round(time.time() - t0, 2)
            sc_exact = eval_exact_tr40(theta)
            m_all = evaluate_predictions_with_live_exec(sc_exact)
            m_train40 = evaluate_task_subset_with_live_exec(sc_exact, train40_tasks)
            m_test16 = evaluate_task_subset_with_live_exec(sc_exact, heldout16_tasks)
            m_all["elapsed_sec"] = elapsed
            m_all["train40"] = m_train40
            m_all["heldout16"] = m_test16
            arm_results[dtype_label][arm_name] = m_all

            if dtype_label == "bf16" and arm_name == "Barron Full-Fixed":
                best_theta_bf16 = theta.clone()

            print(
                f"  {arm_name:28s} ({elapsed:4.1f}s) | "
                f"Train-40: {m_train40['total_slot_hits']:3d}/160 ({m_train40['total_slot_acc']:6.2f}%, Pass@1={m_train40['total_fn_pass']:2d}/40={m_train40['total_pass_at_1']:6.2f}%) | "
                f"HeldOut-16: {m_test16['total_slot_hits']:2d}/64 ({m_test16['total_slot_acc']:6.2f}%, Pass@1={m_test16['total_fn_pass']:2d}/16={m_test16['total_pass_at_1']:6.2f}%)"
            )

    # -------------------------------------------------------------------------
    # 3. Export 100% Valid 256-Tensor PEFT LoRA Checkpoint (`RealCoder-v3`)
    #    Trained Strictly on the 40 Train Tasks (Zero Test-Task Exposure!)
    # -------------------------------------------------------------------------
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    r = 16
    core_star = best_theta_bf16.view(D_out, D_in)
    lora_b_63_down = (v_out_tr40.t() @ (row_scale_tr40.t() * core_star)).contiguous()  # [5120, 16]

    layer_down_a = torch.tensor(d["layer_down_a"], dtype=torch.float32)  # [64, 16, 17408]
    layer_attn_a = torch.tensor(d["layer_attn_a"], dtype=torch.float32)  # [64, 16, 6144]
    u_act63 = torch.tensor(d["u_act63"][:16, :], dtype=torch.float32)  # [16, 17408] (from train40)
    layer_down_a[63] = u_act63

    state_dict = {}
    for l in range(64):
        peft_lp = f"base_model.model.language_model.layers.{l}"
        ka_d = f"{peft_lp}.mlp.down_proj.lora_A.weight"
        kb_d = f"{peft_lp}.mlp.down_proj.lora_B.weight"
        state_dict[ka_d] = layer_down_a[l].to(torch.bfloat16).contiguous()
        if l == 63:
            state_dict[kb_d] = lora_b_63_down.to(torch.bfloat16).contiguous()
        else:
            state_dict[kb_d] = torch.zeros((5120, r), dtype=torch.bfloat16)

        if l % 4 != 3:
            ka_a = f"{peft_lp}.linear_attn.out_proj.lora_A.weight"
            kb_a = f"{peft_lp}.linear_attn.out_proj.lora_B.weight"
        else:
            ka_a = f"{peft_lp}.self_attn.o_proj.lora_A.weight"
            kb_a = f"{peft_lp}.self_attn.o_proj.lora_B.weight"
        state_dict[ka_a] = layer_attn_a[l].to(torch.bfloat16).contiguous()
        state_dict[kb_a] = torch.zeros((5120, r), dtype=torch.bfloat16)

    st_path = os.path.join(CKPT_DIR, "adapter_model.safetensors")
    save_safetensors_file(state_dict, st_path)

    adapter_cfg = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen3.8-27B",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["down_proj", "o_proj", "out_proj"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    cfg_path = os.path.join(CKPT_DIR, "adapter_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(adapter_cfg, f, indent=2)

    if os.path.exists(os.path.dirname(PIQWEN_CKPT_DIR)):
        os.makedirs(PIQWEN_CKPT_DIR, exist_ok=True)
        shutil.copy2(st_path, os.path.join(PIQWEN_CKPT_DIR, "adapter_model.safetensors"))
        shutil.copy2(cfg_path, os.path.join(PIQWEN_CKPT_DIR, "adapter_config.json"))

    report = {
        "model": "Qwen/Qwen3.8-27B",
        "checkpoint": "Qwen3.8-27B-BarronPolyStep-RealCoder-v3",
        "zero_train_on_test_verified": True,
        "train40_task_indices": train40_tasks,
        "heldout16_task_indices": heldout16_tasks,
        "vocab_size": 248320,
        "num_layers": 64,
        "num_lora_tensors": len(state_dict),
        "total_coding_tokens_evaluated": int(len(h63_base)),
        "train40_tokens_in_bf16_cliff_gt_5_2": n_cliff_tr40,
        "tokens_in_strict_bf16_cliff_gt_6_22": strict_cliff_count,
        "base_qwen38_27b": {
            "all56": base_metrics,
            "train40": base_train40_metrics,
            "heldout16": base_heldout16_metrics,
        },
        "oof_2fold_unseen_tasks_polystep": oof_metrics,
        "oof_2fold_unseen_tasks_multilayer_null": oof_multi_metrics,
        "five_arm_benchmark": arm_results,
        "total_runtime_sec": round(time.time() - t_start, 2),
    }
    rep_path = os.path.join(CKPT_DIR, "benchmark_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "realcoder_v3_benchmark_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if os.path.exists(PIQWEN_CKPT_DIR):
        shutil.copy2(rep_path, os.path.join(PIQWEN_CKPT_DIR, "benchmark_report.json"))
        piqwen_res_dir = os.path.join(os.path.dirname( os.path.dirname(PIQWEN_CKPT_DIR) ), "results")
        os.makedirs(piqwen_res_dir, exist_ok=True)
        shutil.copy2(rep_path, os.path.join(piqwen_res_dir, "realcoder_v3_benchmark_report.json"))

    print(f"\n[SAVED] Leak-free 256-tensor PEFT LoRA checkpoint exported to: {CKPT_DIR}")
    print("=" * 96)


if __name__ == "__main__":
    main()
