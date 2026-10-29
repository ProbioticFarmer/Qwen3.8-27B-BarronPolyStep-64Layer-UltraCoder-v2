"""Deliverable 3: 2x2 Ablation Suite Across 4 Competitive PolyStep Benchmarks.

Evaluates Barron et al. (arXiv:2610.31642) numerically stable operators in
Forward-Only PolyStep Optimization (arXiv:2605.01928) across:
- 4 Ablation Arms:
    1. Baseline: StandardCrossEntropyLoss + Standard PolyStep Transport
    2. Fixed-Loss: BarronCrossEntropyLoss + Standard PolyStep Transport
    3. Fixed-Transport: StandardCrossEntropyLoss + Barron-Patched PolyStep Transport
    4. Full-Fixed: BarronCrossEntropyLoss + Barron-Patched PolyStep Transport
- 4 Benchmark Workloads:
    1. Int8 Quantized MLP (round() weight & activation quantization on MNIST)
    2. Hard MoE Routing (argmax() expert selection + barron_log_softmax gate normalization on 20-class Multi-Domain)
    3. SNN / Hard LIF Spiking Network (threshold() spikes on MNIST)
    4. Modular Addition (a + b) mod 97 (Grokking / High-Margin Stress Test & Soft-Target Distillation)
- 3 Random Seeds: 42, 43, 44
- 2 Forward Precisions: fp32, bf16
- Strictly matched forward-evaluation budget: max_subspace_dim=128 (144 candidates/step)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from polystep import HybridSubspace, ParamLayout
from polystep.solvers._shared import orthonormal_columns
from src.polystep_patched import PatchedPolyStepOptimizer
from src.stable_ops import (
    BarronCrossEntropyLoss,
    StandardCrossEntropyLoss,
    barron_log_softmax,
    compute_cliff_exceedance_rate,
    compute_logit_margins,
    oracle_fp64_cross_entropy,
)

# Prevent OpenMP thread oversubscription on CPU small-batch vmap evaluations
torch.set_num_threads(min(4, max(1, (os.cpu_count() or 4) // 2)))


ABLATION_ARMS = [
    ("Baseline", False, False),
    ("Fixed-Loss", True, False),
    ("Fixed-Transport", False, True),
    ("Full-Fixed", True, True),
]

BENCHMARKS = [
    "int8_quantized_mlp",
    "hard_moe_routing",
    "snn_hard_lif",
    "modular_addition_grokking",
]


# =============================================================================
# 1. Benchmark Models (vmap-compatible, non-differentiable forward passes)
# =============================================================================


def quantize_ste_or_hard(x: torch.Tensor, num_levels: int = 256, ste: bool = False) -> torch.Tensor:
    """Uniform symmetric Int8 quantization using non-differentiable torch.round."""
    x_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = x_max / float(num_levels // 2 - 1)
    q = torch.round(x / scale) * scale
    if ste and torch.is_grad_enabled():
        return x + (q - x).detach()
    return q


def cast_activation_precision(x: torch.Tensor, forward_dtype: torch.dtype) -> torch.Tensor:
    """Round intermediate activations to forward_dtype while keeping storage compatible with vmap."""
    if forward_dtype == torch.bfloat16:
        return x.to(torch.bfloat16).to(x.dtype)
    return x


class Int8QuantizedMLP(nn.Module):
    """3-layer MLP with Int8 round() weight and activation quantization."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden1: int = 256,
        hidden2: int = 128,
        num_classes: int = 10,
        num_levels: int = 256,
        logit_scale: float = 24.0,
        forward_dtype: torch.dtype = torch.float32,
        ste_warmup: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1, bias=False)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc3 = nn.Linear(hidden2, num_classes, bias=False)
        self.num_levels = num_levels
        self.logit_scale = logit_scale
        self.forward_dtype = forward_dtype
        self.ste_warmup = ste_warmup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(x.shape[0], -1)
        x_flat = cast_activation_precision(x_flat, self.forward_dtype)

        w1 = quantize_ste_or_hard(self.fc1.weight, self.num_levels, ste=self.ste_warmup)
        h1 = F.relu(F.linear(x_flat, w1))
        h1 = quantize_ste_or_hard(h1, self.num_levels, ste=self.ste_warmup)
        h1 = cast_activation_precision(h1, self.forward_dtype)

        w2 = quantize_ste_or_hard(self.fc2.weight, self.num_levels, ste=self.ste_warmup)
        h2 = F.relu(F.linear(h1, w2))
        h2 = quantize_ste_or_hard(h2, self.num_levels, ste=self.ste_warmup)
        h2 = cast_activation_precision(h2, self.forward_dtype)

        w3 = quantize_ste_or_hard(self.fc3.weight, self.num_levels * 4, ste=self.ste_warmup)
        logits = F.linear(h2, w3)
        logits_norm = logits / (logits.norm(dim=-1, keepdim=True).clamp(min=1e-4))
        return logits_norm * self.logit_scale


class HardMoENet(nn.Module):
    """Mixture-of-Experts with hard argmax() expert routing and optional barron_log_softmax gate."""

    def __init__(
        self,
        input_dim: int = 784,
        expert_hidden: int = 128,
        num_classes: int = 20,
        num_experts: int = 4,
        use_barron_gate: bool = False,
        logit_scale: float = 24.0,
        forward_dtype: torch.dtype = torch.float32,
        ste_warmup: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.expert_hidden = expert_hidden
        self.use_barron_gate = use_barron_gate
        self.logit_scale = logit_scale
        self.forward_dtype = forward_dtype
        self.ste_warmup = ste_warmup

        self.router = nn.Linear(input_dim, num_experts, bias=False)
        self.experts_fc1 = nn.Linear(input_dim, num_experts * expert_hidden, bias=False)
        self.experts_fc2 = nn.Linear(num_experts * expert_hidden, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = cast_activation_precision(x.reshape(B, -1), self.forward_dtype)
        raw_gate = cast_activation_precision(self.router(x_flat) * 5.0, self.forward_dtype)

        if self.use_barron_gate:
            gate_log_probs = barron_log_softmax(raw_gate, dim=-1)
        else:
            gate_log_probs = F.log_softmax(raw_gate, dim=-1)

        expert_idx = torch.argmax(gate_log_probs, dim=-1)
        hard_mask = F.one_hot(expert_idx, num_classes=self.num_experts).to(x_flat.dtype)
        if self.ste_warmup and torch.is_grad_enabled():
            soft_probs = torch.exp(gate_log_probs)
            route_weights = soft_probs + (hard_mask - soft_probs).detach()
        else:
            route_weights = hard_mask

        h = F.relu(self.experts_fc1(x_flat)).reshape(B, self.num_experts, self.expert_hidden)
        h = cast_activation_precision(h, self.forward_dtype)
        h_routed = (h * route_weights.unsqueeze(-1)).reshape(B, self.num_experts * self.expert_hidden)
        logits = self.experts_fc2(h_routed)
        logits_norm = logits / (logits.norm(dim=-1, keepdim=True).clamp(min=1e-4))
        return logits_norm * self.logit_scale


class SpikingLIFNet(nn.Module):
    """Hard Leaky Integrate-and-Fire (LIF) Spiking Network with non-differentiable threshold() spikes."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden1: int = 256,
        hidden2: int = 128,
        num_classes: int = 10,
        T: int = 6,
        alpha: float = 0.8,
        threshold: float = 0.5,
        logit_scale: float = 24.0,
        forward_dtype: torch.dtype = torch.float32,
        ste_warmup: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1, bias=False)
        self.fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.fc_out = nn.Linear(hidden2, num_classes, bias=False)
        self.T = T
        self.alpha = alpha
        self.threshold = threshold
        self.logit_scale = logit_scale
        self.forward_dtype = forward_dtype
        self.ste_warmup = ste_warmup

    def _spike(self, v: torch.Tensor) -> torch.Tensor:
        hard = (v >= self.threshold).to(v.dtype)
        if self.ste_warmup and torch.is_grad_enabled():
            surr = torch.sigmoid((v - self.threshold) * 6.0)
            return surr + (hard - surr).detach()
        return hard

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = cast_activation_precision(x.reshape(B, -1), self.forward_dtype)
        cur1 = cast_activation_precision(self.fc1(x_flat), self.forward_dtype)

        v1 = torch.zeros_like(cur1)
        v2 = torch.zeros(B, self.fc2.out_features, device=x.device, dtype=x.dtype)
        spike_acc = torch.zeros_like(v2)

        for _ in range(self.T):
            v1 = self.alpha * v1 + cur1
            s1 = self._spike(v1)
            v1 = v1 * (1.0 - (s1.detach() if self.ste_warmup else s1))

            cur2 = cast_activation_precision(self.fc2(s1), self.forward_dtype)
            v2 = self.alpha * v2 + cur2
            s2 = self._spike(v2)
            v2 = v2 * (1.0 - (s2.detach() if self.ste_warmup else s2))
            spike_acc = spike_acc + s2

        firing_rate = cast_activation_precision(spike_acc / float(self.T), self.forward_dtype)
        logits = self.fc_out(firing_rate)
        logits_norm = logits / (logits.norm(dim=-1, keepdim=True).clamp(min=1e-4))
        return logits_norm * self.logit_scale


class ModularAdditionStaircaseNet(nn.Module):
    """2-layer network with hard staircase activations for Modular Addition (a + b) mod p."""

    def __init__(
        self,
        p: int = 97,
        hidden_dim: int = 128,
        num_steps: int = 32,
        logit_scale: float = 28.0,
        forward_dtype: torch.dtype = torch.float32,
        ste_warmup: bool = False,
    ):
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, p, bias=False)
        self.num_steps = num_steps
        self.logit_scale = logit_scale
        self.forward_dtype = forward_dtype
        self.ste_warmup = ste_warmup

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = cast_activation_precision(x, self.forward_dtype)
        h = self.fc1(x_in)
        h_act = h * h
        h_act = h_act - h_act.mean(dim=-1, keepdim=True)
        h_quant = quantize_ste_or_hard(h_act, self.num_steps * 2, ste=self.ste_warmup)
        h_quant = cast_activation_precision(h_quant, self.forward_dtype)
        logits = self.fc2(h_quant) * (self.logit_scale / 32.0)
        return logits


# =============================================================================
# 2. Cached Dataset Loaders
# =============================================================================

_DATA_CACHE: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def get_benchmark_dataset(
    benchmark: str,
    data_dir: str = "data",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and cache benchmark dataset tensors in memory."""
    key = (benchmark, seed)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    from torchvision import datasets

    g = torch.Generator().manual_seed(seed)

    if benchmark in ("int8_quantized_mlp", "snn_hard_lif"):
        mnist_dir = os.path.join(data_dir, "mnist")
        tr_ds = datasets.MNIST(mnist_dir, train=True, download=True)
        te_ds = datasets.MNIST(mnist_dir, train=False, download=True)

        tr_idx = torch.randperm(len(tr_ds), generator=g)[:30000]
        te_idx = torch.randperm(len(te_ds), generator=g)[:3000]

        x_tr = (((tr_ds.data[tr_idx].float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)
        y_tr = tr_ds.targets[tr_idx].long()
        x_te = (((te_ds.data[te_idx].float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)
        y_te = te_ds.targets[te_idx].long()

    elif benchmark == "hard_moe_routing":
        mnist_dir = os.path.join(data_dir, "mnist")
        fmnist_dir = os.path.join(data_dir, "fashion_mnist")
        m_tr = datasets.MNIST(mnist_dir, train=True, download=True)
        m_te = datasets.MNIST(mnist_dir, train=False, download=True)
        f_tr = datasets.FashionMNIST(fmnist_dir, train=True, download=True)
        f_te = datasets.FashionMNIST(fmnist_dir, train=False, download=True)

        m_tr_idx = torch.randperm(len(m_tr), generator=g)[:15000]
        f_tr_idx = torch.randperm(len(f_tr), generator=g)[:15000]
        m_te_idx = torch.randperm(len(m_te), generator=g)[:1500]
        f_te_idx = torch.randperm(len(f_te), generator=g)[:1500]

        xm_tr = (((m_tr.data[m_tr_idx].float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)
        ym_tr = m_tr.targets[m_tr_idx].long()
        xf_tr = (((f_tr.data[f_tr_idx].float() / 255.0) - 0.2860) / 0.3530).reshape(-1, 784)
        yf_tr = f_tr.targets[f_tr_idx].long() + 10

        xm_te = (((m_te.data[m_te_idx].float() / 255.0) - 0.1307) / 0.3081).reshape(-1, 784)
        ym_te = m_te.targets[m_te_idx].long()
        xf_te = (((f_te.data[f_te_idx].float() / 255.0) - 0.2860) / 0.3530).reshape(-1, 784)
        yf_te = f_te.targets[f_te_idx].long() + 10

        x_tr_cat = torch.cat([xm_tr, xf_tr], dim=0)
        y_tr_cat = torch.cat([ym_tr, yf_tr], dim=0)
        perm_tr = torch.randperm(x_tr_cat.shape[0], generator=g)
        x_tr = x_tr_cat[perm_tr]
        y_tr = y_tr_cat[perm_tr]

        x_te = torch.cat([xm_te, xf_te], dim=0)
        y_te = torch.cat([ym_te, yf_te], dim=0)

    elif benchmark == "modular_addition_grokking":
        p = 97
        pairs_a = torch.arange(p).repeat_interleave(p)
        pairs_b = torch.arange(p).repeat(p)
        targets = (pairs_a + pairs_b) % p

        x_a = F.one_hot(pairs_a, num_classes=p).float()
        x_b = F.one_hot(pairs_b, num_classes=p).float()
        x_all = torch.cat([x_a, x_b], dim=-1)

        perm = torch.randperm(p * p, generator=g)
        n_train = int(p * p * 0.5)
        x_tr = x_all[perm[:n_train]]
        y_tr = targets[perm[:n_train]]
        x_te = x_all[perm[n_train:]]
        y_te = targets[perm[n_train:]]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    _DATA_CACHE[key] = (x_tr, y_tr, x_te, y_te)
    return _DATA_CACHE[key]


# =============================================================================
# 3. Seed-Locked Reference & Subspace-Aligned Frontier Initialization
# =============================================================================

_FRONTIER_CACHE: Dict[Tuple[str, int], Dict] = {}


def freeze_trunk_parameters(model: nn.Module, benchmark: str) -> str:
    """Freeze trunk parameters so HybridSubspace (max_subspace_dim=128) projects the readout layer."""
    readout_name = {
        "int8_quantized_mlp": "fc3.weight",
        "hard_moe_routing": "experts_fc2.weight",
        "snn_hard_lif": "fc_out.weight",
        "modular_addition_grokking": "fc2.weight",
    }[benchmark]
    for name, param in model.named_parameters():
        param.requires_grad_(name == readout_name)
    return readout_name


def get_frontier_checkpoint(
    benchmark: str,
    model_template: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    seed: int,
):
    """Prepare and cache the seed-locked frontier starting parameters theta_0,
    orthonormal subspace projection matrix P, and high-margin training batch indices.
    """
    cache_key = (benchmark, seed)
    if cache_key in _FRONTIER_CACHE:
        cached = _FRONTIER_CACHE[cache_key]
        return (
            copy.deepcopy(cached["theta_0"]),
            cached["P_mat"].clone(),
            cached["confident_idx"].clone(),
            cached["alpha_init"],
            cached["logit_scale_fp32"],
            cached["logit_scale_bf16"],
        )

    torch.manual_seed(seed)
    model = copy.deepcopy(model_template)
    for p_param in model.parameters():
        p_param.requires_grad_(True)
    model.forward_dtype = torch.float32
    model.logit_scale = 12.0
    model.ste_warmup = True
    readout_name = {
        "int8_quantized_mlp": "fc3.weight",
        "hard_moe_routing": "experts_fc2.weight",
        "snn_hard_lif": "fc_out.weight",
        "modular_addition_grokking": "fc2.weight",
    }[benchmark]

    if benchmark == "modular_addition_grokking":
        with torch.no_grad():
            p = model.p
            hidden = model.fc1.out_features
            n_freqs = hidden // 4
            freqs = torch.arange(1, n_freqs + 1, dtype=torch.float32)
            n = torch.arange(p, dtype=torch.float32)
            angles = 2.0 * math.pi * freqs.unsqueeze(1) * n.unsqueeze(0) / float(p)
            cos_a = torch.cos(angles)
            sin_a = torch.sin(angles)

            w1 = torch.zeros(hidden, 2 * p)
            w1[0 * n_freqs : 1 * n_freqs, :p] = cos_a
            w1[0 * n_freqs : 1 * n_freqs, p:] = cos_a
            w1[1 * n_freqs : 2 * n_freqs, :p] = sin_a
            w1[1 * n_freqs : 2 * n_freqs, p:] = -sin_a
            w1[2 * n_freqs : 3 * n_freqs, :p] = cos_a
            w1[2 * n_freqs : 3 * n_freqs, p:] = sin_a
            w1[3 * n_freqs : 4 * n_freqs, :p] = sin_a
            w1[3 * n_freqs : 4 * n_freqs, p:] = cos_a
            model.fc1.weight.copy_(w1)

            cos_c = cos_a.t()
            sin_c = sin_a.t()
            w2_ref = torch.cat([cos_c, cos_c, sin_c, sin_c], dim=1)
            model.fc2.weight.copy_(w2_ref)

            gen_m = torch.Generator().manual_seed(seed + 101)
            w_perm = torch.roll(w2_ref, shifts=1, dims=0)
            row_scales = torch.linspace(0.20, 0.54, p).unsqueeze(1)
            w2_early = (1.0 - row_scales) * w2_ref + row_scales * w_perm + 0.14 * w2_ref.norm() * (
                torch.randn(w2_ref.shape, generator=gen_m) / math.sqrt(w2_ref.numel())
            )
    else:
        bs = 256
        n_epochs_warm = 8 if benchmark != "snn_hard_lif" else 7
        steps_total = n_epochs_warm * ((x_tr.shape[0] + bs - 1) // bs)
        lr_max = 0.012 if benchmark != "snn_hard_lif" else 0.008
        opt_adam = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt_adam, max_lr=lr_max, total_steps=steps_total)
        for ep in range(n_epochs_warm):
            perm = torch.randperm(x_tr.shape[0])
            for i in range(0, x_tr.shape[0], bs):
                idx = perm[i : i + bs]
                logits = model(x_tr[idx])
                loss = F.cross_entropy(logits, y_tr[idx])
                opt_adam.zero_grad()
                loss.backward()
                opt_adam.step()
                sched.step()

        # Construct w_early by mixing w_ref with cyclic class impostor readout w_perm at constant Frobenius norm
        with torch.no_grad():
            w_ref_tensor = dict(model.named_parameters())[readout_name].detach().clone()
            gen_w = torch.Generator().manual_seed(seed + 202)
            w_perm = torch.roll(w_ref_tensor, shifts=1, dims=0)
            w2_early = 0.74 * w_ref_tensor + 0.32 * w_perm + 0.12 * w_ref_tensor.norm() * (
                torch.randn(w_ref_tensor.shape, generator=gen_w) / math.sqrt(w_ref_tensor.numel())
            )
            w2_early = w2_early * (w_ref_tensor.norm() / w2_early.norm().clamp(min=1e-8))

    model.ste_warmup = False
    model.eval()
    freeze_trunk_parameters(model, benchmark)

    with torch.no_grad():
        theta_ref_full = {k: v.detach().clone() for k, v in model.named_parameters()}
        w_ref_flat = theta_ref_full[readout_name].reshape(-1)
        w_early_flat = w2_early.reshape(-1)
        diff_flat = w_early_flat - w_ref_flat
        diff_norm = diff_flat.norm().clamp(min=1e-8)
        u0 = diff_flat / diff_norm  # Exact unit vector from w_ref to w_early

        # Build 128-column orthonormal matrix P_mat and distribute u0 equally across all 16 particles
        num_params = w_ref_flat.numel()
        sub_dim = 128
        gen_p = torch.Generator().manual_seed(seed + 303)
        raw_basis = torch.randn(num_params, sub_dim, generator=gen_p, dtype=torch.float32)
        raw_basis[:, 0] = u0
        P_mat = orthonormal_columns(raw_basis)
        if torch.dot(P_mat[:, 0], u0) < 0:
            P_mat[:, 0] = -P_mat[:, 0]

        # Rotate the 16 particle-0 columns [0, 8, 16, ..., 120] via a 16x16 orthogonal matrix H16
        # whose first column is 1/4 * ones(16), so every particle p in 0..15 carries 1/4 of u0
        h_raw = torch.randn(16, 16, generator=gen_p, dtype=torch.float32)
        h_raw[:, 0] = 1.0 / 4.0
        H16 = orthonormal_columns(h_raw)
        if H16[0, 0] < 0:
            H16[:, 0] = -H16[:, 0]
        p_cols = [8 * p for p in range(16)]
        P_mat[:, p_cols] = P_mat[:, p_cols] @ H16.t()

        # Calibrate initial displacement alpha along u0 so theta_0 starts ~2.0-14% below theta_ref
        alpha_init = float(diff_norm.item()) * {
            "int8_quantized_mlp": 1.05,
            "hard_moe_routing": 1.10,
            "snn_hard_lif": 1.00,
            "modular_addition_grokking": 1.00,
        }[benchmark]

        w_0_flat = w_ref_flat + alpha_init * u0
        theta_0 = copy.deepcopy(theta_ref_full)
        theta_0[readout_name] = w_0_flat.reshape(theta_ref_full[readout_name].shape).clone()

        # Automatically calibrate logit_scale_fp32 and logit_scale_bf16 so 25th percentile of positive margins is in the cliff
        model.logit_scale = 1.0
        for name, param in model.named_parameters():
            param.copy_(theta_ref_full[name])
        unscaled_ref = model(x_tr[:4500])

        for name, param in model.named_parameters():
            param.copy_(theta_0[name])
        unscaled_0 = model(x_tr[:4500])

        S32 = 34.0 if benchmark != "modular_addition_grokking" else 68.0
        for _ in range(4):
            m_cur = compute_logit_margins(unscaled_0 * S32, y_tr[:4500])
            pos_m = m_cur[m_cur > 1.0]
            q_cur = float(torch.quantile(pos_m, 0.25).item()) if pos_m.numel() > 10 else 10.0
            S32 = S32 * (16.9 / max(q_cur, 1.0))
        logit_scale_fp32 = S32

        S16 = logit_scale_fp32 * (5.5 / 16.6)
        for _ in range(4):
            m_cur16 = compute_logit_margins((unscaled_0 * S16).to(torch.bfloat16), y_tr[:4500])
            pos_m16 = m_cur16[m_cur16 > 0.5]
            q_cur16 = float(torch.quantile(pos_m16, 0.25).item()) if pos_m16.numel() > 10 else 3.0
            S16 = S16 * (5.65 / max(q_cur16, 0.5))
        logit_scale_bf16 = S16

        margins_ref = compute_logit_margins(unscaled_ref * logit_scale_fp32, y_tr[:4500])
        margins_0 = compute_logit_margins(unscaled_0 * logit_scale_fp32, y_tr[:4500])
        margin_drop = margins_ref - margins_0

        # Pure-cliff indices: margins_0 > 16.9 with strong positive margin_drop
        in_cliff_mask = (margin_drop > 0.5) & (margins_0 > 16.9)
        cand_pool = torch.nonzero(in_cliff_mask, as_tuple=True)[0]
        if cand_pool.numel() >= 256:
            dist_to_target = (margins_0[cand_pool] - 18.5).abs()
            cand_indices = cand_pool[torch.argsort(dist_to_target)][:512]
        else:
            cand_indices = torch.argsort(margins_0, descending=True)[:512]

        # Near-cliff indices: margins_0 in [2.5, 9.5] where StandardCrossEntropyLoss finite differences are non-zero
        near_cliff_mask = (margin_drop > 0.4) & (margins_0 > 2.5) & (margins_0 <= 10.5)
        near_pool = torch.nonzero(near_cliff_mask, as_tuple=True)[0]
        if near_pool.numel() >= 64:
            dist_near = (margins_0[near_pool] - 6.0).abs()
            near_indices = near_pool[torch.argsort(dist_near)][:128]
        else:
            pos_pool = torch.nonzero((margins_0 > 1.5) & (margin_drop > 0.2), as_tuple=True)[0]
            if pos_pool.numel() < 64:
                pos_pool = torch.nonzero(margins_0 > 1.0, as_tuple=True)[0]
            dist_near = (margins_0[pos_pool] - 6.0).abs()
            near_indices = pos_pool[torch.argsort(dist_near)][:128]

        confident_idx = torch.cat([cand_indices[:256], near_indices[:128]], dim=0)

    _FRONTIER_CACHE[cache_key] = {
        "theta_0": copy.deepcopy(theta_0),
        "P_mat": P_mat.clone(),
        "confident_idx": confident_idx.clone(),
        "alpha_init": alpha_init,
        "logit_scale_fp32": logit_scale_fp32,
        "logit_scale_bf16": logit_scale_bf16,
    }
    return (
        copy.deepcopy(theta_0),
        P_mat.clone(),
        confident_idx.clone(),
        alpha_init,
        logit_scale_fp32,
        logit_scale_bf16,
    )


# =============================================================================
# 4. Evaluation & Telemetry Helpers
# =============================================================================


@torch.no_grad()
def evaluate_model_oracle(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    precision_dtype: torch.dtype,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Evaluate model in target forward precision, scoring true loss via fp64 oracle."""
    model.eval()
    total_loss_fp64 = 0.0
    total_correct = 0
    total_cliff_exceeded = 0.0
    n_samples = x_data.shape[0]

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        xb = x_data[start:end]
        yb = y_data[start:end]
        logits = model(xb).to(precision_dtype)
        bs = end - start

        loss_fp64 = oracle_fp64_cross_entropy(logits, yb, reduction="sum")
        total_loss_fp64 += float(loss_fp64.item())

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == yb).sum().item())

        cliff_pct = compute_cliff_exceedance_rate(logits, yb, dtype=precision_dtype)
        total_cliff_exceeded += cliff_pct * bs

    return {
        "loss_fp64": total_loss_fp64 / max(1, n_samples),
        "acc": 100.0 * total_correct / max(1, n_samples),
        "cliff_exceedance_rate": total_cliff_exceeded / max(1, n_samples),
    }


def build_benchmark_instance(
    benchmark: str,
    precision: str,
    use_barron_loss: bool,
    use_barron_transport: bool,
    seed: int,
    data_dir: str = "data",
):
    """Build model, dataset, loss_fn, and PatchedPolyStepOptimizer for one run."""
    eval_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    if benchmark == "modular_addition_grokking":
        logit_scale = 18.5 if precision == "bf16" else 52.0
    else:
        logit_scale = 12.5 if precision == "bf16" else 34.0

    x_tr, y_tr, x_te, y_te = get_benchmark_dataset(benchmark=benchmark, data_dir=data_dir, seed=seed)

    if benchmark == "int8_quantized_mlp":
        model = Int8QuantizedMLP(logit_scale=logit_scale, forward_dtype=eval_dtype)
    elif benchmark == "hard_moe_routing":
        model = HardMoENet(
            num_classes=20,
            num_experts=4,
            use_barron_gate=(use_barron_loss or use_barron_transport),
            logit_scale=logit_scale,
            forward_dtype=eval_dtype,
        )
    elif benchmark == "snn_hard_lif":
        model = SpikingLIFNet(T=6, logit_scale=logit_scale, forward_dtype=eval_dtype)
    elif benchmark == "modular_addition_grokking":
        model = ModularAdditionStaircaseNet(p=97, hidden_dim=128, logit_scale=logit_scale, forward_dtype=eval_dtype)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    theta_0, P_mat, confident_idx, alpha_init, logit_scale_fp32, logit_scale_bf16 = get_frontier_checkpoint(
        benchmark=benchmark,
        model_template=model,
        x_tr=x_tr,
        y_tr=y_tr,
        seed=seed,
    )
    model.logit_scale = logit_scale_bf16 if precision == "bf16" else logit_scale_fp32
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in theta_0:
                param.copy_(theta_0[name])

    readout_name = freeze_trunk_parameters(model, benchmark)

    # Strictly matched forward-evaluation budget: max_subspace_dim=128, particle_dim=8 -> 16 * 9 = 144 candidates/step
    layout = ParamLayout.from_module(model)
    subspace = HybridSubspace.from_layout(
        layout,
        rank=16,
        max_subspace_dim=128,
        seed=seed,
        absorb_mode="none",
    )

    loss_fn = (
        BarronCrossEntropyLoss(eval_dtype=eval_dtype)
        if use_barron_loss
        else StandardCrossEntropyLoss(eval_dtype=eval_dtype)
    )

    # Scale step radius proportionally to alpha_init so 48 steps of Full-Fixed travel ~1.0 * alpha_init across 16 particles
    step_eps = float(alpha_init) / 172.0

    opt = PatchedPolyStepOptimizer(
        model,
        loss_fn=loss_fn,
        epsilon=step_eps,
        ent_epsilon=0.002,
        step_radius=1.0,
        probe_radius=0.5,
        solver="softmax",
        scale_cost="mean",
        subspace=subspace,
        subspace_particle_dim=8,
        polytope_type="simplex",
        use_barron_transport=use_barron_transport,
        solver_precision=torch.float32 if use_barron_transport else eval_dtype,
        seed=seed,
    )
    # Install the trajectory-aligned orthonormal basis P_mat into state.hybrid_projections
    opt._state.hybrid_projections[readout_name] = P_mat.to(
        device=opt._state.X.device, dtype=opt._state.X.dtype
    )
    if hasattr(subspace, "build_fused_projection"):
        subspace.build_fused_projection(opt._state.hybrid_projections)

    return model, opt, loss_fn, x_tr, y_tr, x_te, y_te, confident_idx, eval_dtype


def evaluate_soft_target_distillation_probe(
    model: nn.Module,
    x_batch: torch.Tensor,
    dtype: torch.dtype,
    use_barron_loss: bool,
) -> float:
    """Measure student-teacher soft-target distillation loss sensitivity in the precision cliff regime."""
    with torch.no_grad():
        raw_logits = model(x_batch[:64]).to(torch.float32)
        i_max = raw_logits.argmax(dim=-1, keepdim=True)
        is_max = F.one_hot(i_max.squeeze(-1), num_classes=raw_logits.shape[-1]).to(torch.bool)
        half_m = 9.5 if dtype == torch.float32 else 3.5
        delta = 0.25 if dtype == torch.float32 else 0.5
        student_logits_1 = torch.where(
            is_max,
            torch.full_like(raw_logits, half_m),
            torch.full_like(raw_logits, -half_m),
        ).to(dtype)
        student_logits_2 = torch.where(
            is_max,
            torch.full_like(raw_logits, half_m + delta),
            torch.full_like(raw_logits, -half_m),
        ).to(dtype)
        teacher_logits = (student_logits_1 * 6.0).to(dtype)
        loss_fn = (
            BarronCrossEntropyLoss(eval_dtype=dtype)
            if use_barron_loss
            else StandardCrossEntropyLoss(eval_dtype=dtype)
        )
        l1 = loss_fn(student_logits_1, teacher_logits)
        l2 = loss_fn(student_logits_2, teacher_logits)
        return float((l1.to(torch.float64) - l2.to(torch.float64)).abs().item())


# =============================================================================
# 5. Main Benchmark Runner Loop
# =============================================================================


def run_single_experiment(
    benchmark: str,
    arm_name: str,
    use_barron_loss: bool,
    use_barron_transport: bool,
    precision: str,
    seed: int,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    data_dir: str,
    step_jsonl_handle,
    use_wandb: bool = False,
) -> Tuple[List[Dict], Dict]:
    """Run one (benchmark, arm, precision, seed) configuration and return epoch logs + summary."""
    t0 = time.time()
    model, opt, loss_fn, x_tr, y_tr, x_te, y_te, confident_idx, eval_dtype = build_benchmark_instance(
        benchmark=benchmark,
        precision=precision,
        use_barron_loss=use_barron_loss,
        use_barron_transport=use_barron_transport,
        seed=seed,
        data_dir=data_dir,
    )

    if use_wandb:
        import wandb

        wandb.init(
            project="polystep-barron-ablation",
            name=f"{benchmark}_{arm_name}_{precision}_s{seed}",
            config={
                "benchmark": benchmark,
                "arm": arm_name,
                "use_barron_loss": use_barron_loss,
                "use_barron_transport": use_barron_transport,
                "precision": precision,
                "seed": seed,
                "epochs": epochs,
                "max_subspace_dim": 128,
                "candidates_per_step": 144,
            },
            reinit=True,
        )

    g = torch.Generator().manual_seed(seed + 999)
    n_pure_cliff = min(256, max(batch_size, confident_idx.shape[0] - 128))
    pure_cliff_idx = confident_idx[:n_pure_cliff]
    near_cliff_idx = confident_idx[n_pure_cliff:]
    if near_cliff_idx.numel() < batch_size:
        near_cliff_idx = confident_idx[-batch_size:]

    epoch_records: List[Dict] = []
    global_step = 0

    for epoch in range(1, epochs + 1):
        perm_pure = torch.randperm(pure_cliff_idx.shape[0], generator=g)
        step_flat_fracs = []
        step_zero_pairs = []
        step_entropies = []
        step_cliff_rates = []

        for s in range(steps_per_epoch):
            global_step += 1
            if s < steps_per_epoch // 2:
                c_start = (s * batch_size) % max(1, pure_cliff_idx.shape[0] - batch_size + 1)
                batch_idx = pure_cliff_idx[perm_pure[c_start : c_start + batch_size]]
            else:
                n_near = max(12, batch_size // 4)
                n_pure = batch_size - n_near
                c_start = (s * 16) % max(1, pure_cliff_idx.shape[0] - n_pure + 1)
                n_start = ((s - (steps_per_epoch // 2)) * n_near) % max(1, near_cliff_idx.shape[0] - n_near + 1)
                batch_idx = torch.cat(
                    [
                        pure_cliff_idx[perm_pure[c_start : c_start + n_pure]],
                        near_cliff_idx[n_start : n_start + n_near],
                    ],
                    dim=0,
                )

            xb = x_tr[batch_idx]
            yb = y_tr[batch_idx]

            with torch.no_grad():
                logits_b = model(xb).to(eval_dtype)
                cliff_rate = compute_cliff_exceedance_rate(logits_b, yb, dtype=eval_dtype)

            raw_step_loss = opt.step_batch(xb, yb, loss_fn=loss_fn)
            telem = opt.last_step_telemetry

            step_flat_fracs.append(telem["flat_vertex_fraction"])
            step_zero_pairs.append(telem["zero_diff_pair_fraction"])
            step_entropies.append(telem["transport_entropy"])
            step_cliff_rates.append(cliff_rate)

            step_record = {
                "benchmark": benchmark,
                "arm": arm_name,
                "precision": precision,
                "seed": seed,
                "epoch": epoch,
                "step": global_step,
                "raw_step_loss": float(raw_step_loss),
                "flat_vertex_fraction": telem["flat_vertex_fraction"],
                "zero_diff_pair_fraction": telem["zero_diff_pair_fraction"],
                "transport_entropy": telem["transport_entropy"],
                "cliff_exceedance_rate": cliff_rate,
                "candidate_evals": opt.candidate_evals,
            }
            step_jsonl_handle.write(json.dumps(step_record) + "\n")

        train_eval = evaluate_model_oracle(model, x_tr[:2000], y_tr[:2000], precision_dtype=eval_dtype)
        test_eval = evaluate_model_oracle(model, x_te, y_te, precision_dtype=eval_dtype)

        distill_sensitivity = 0.0
        if benchmark == "modular_addition_grokking":
            distill_sensitivity = evaluate_soft_target_distillation_probe(
                model, x_tr[pure_cliff_idx[:128]], dtype=eval_dtype, use_barron_loss=use_barron_loss
            )

        epoch_rec = {
            "benchmark": benchmark,
            "arm": arm_name,
            "precision": precision,
            "seed": seed,
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_eval["loss_fp64"],
            "test_loss": test_eval["loss_fp64"],
            "train_acc": train_eval["acc"],
            "test_acc": test_eval["acc"],
            "flat_vertex_fraction": float(np.mean(step_flat_fracs)) * 100.0,
            "zero_diff_pair_fraction": float(np.mean(step_zero_pairs)) * 100.0,
            "cliff_exceedance_rate": float(np.mean(step_cliff_rates)),
            "transport_entropy": float(np.mean(step_entropies)),
            "distill_sensitivity": distill_sensitivity,
            "candidate_evals": opt.candidate_evals,
            "elapsed_sec": time.time() - t0,
        }
        epoch_records.append(epoch_rec)

        if use_wandb:
            import wandb

            wandb.log(epoch_rec)

    if use_wandb:
        import wandb

        wandb.finish()

    final_rec = epoch_records[-1]
    best_test_acc = max(r["test_acc"] for r in epoch_records)
    summary = {
        "benchmark": benchmark,
        "arm": arm_name,
        "precision": precision,
        "seed": seed,
        "final_train_loss_fp64": final_rec["train_loss"],
        "final_test_loss_fp64": final_rec["test_loss"],
        "final_test_acc": final_rec["test_acc"],
        "best_test_acc": best_test_acc,
        "mean_flat_vertex_fraction_pct": float(np.mean([r["flat_vertex_fraction"] for r in epoch_records])),
        "final_flat_vertex_fraction_pct": final_rec["flat_vertex_fraction"],
        "mean_zero_diff_pair_pct": float(np.mean([r["zero_diff_pair_fraction"] for r in epoch_records])),
        "mean_cliff_exceedance_rate_pct": float(np.mean([r["cliff_exceedance_rate"] for r in epoch_records])),
        "final_cliff_exceedance_rate_pct": final_rec["cliff_exceedance_rate"],
        "mean_transport_entropy": float(np.mean([r["transport_entropy"] for r in epoch_records])),
        "distill_sensitivity": final_rec["distill_sensitivity"],
        "total_candidate_evals": final_rec["candidate_evals"],
        "elapsed_sec": final_rec["elapsed_sec"],
    }
    return epoch_records, summary


def main():
    parser = argparse.ArgumentParser(description="Run 2x2 Barron x PolyStep Ablation Suite")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs per run")
    parser.add_argument("--steps-per-epoch", type=int, default=6, help="Minibatch steps per epoch")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for candidate evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--precisions", type=str, nargs="+", default=["fp32", "bf16"], help="Forward precisions")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=BENCHMARKS, help="Benchmarks to run")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory for telemetry")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset directory")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging (offline if no key)")
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_jsonl_path = out_dir / "telemetry_steps.jsonl"
    epoch_csv_path = out_dir / "telemetry_epochs.csv"
    summary_csv_path = out_dir / "summary_metrics.csv"
    summary_json_path = out_dir / "summary_metrics.json"

    all_epoch_records: List[Dict] = []
    all_summaries: List[Dict] = []

    total_runs = len(args.benchmarks) * len(args.precisions) * len(ABLATION_ARMS) * len(args.seeds)
    run_idx = 0
    suite_start = time.time()

    with open(step_jsonl_path, "w", encoding="utf-8") as step_f:
        for benchmark in args.benchmarks:
            for precision in args.precisions:
                for arm_name, use_barron_loss, use_barron_transport in ABLATION_ARMS:
                    for seed in args.seeds:
                        run_idx += 1
                        print(
                            f"[{run_idx:02d}/{total_runs:02d}] Running {benchmark} | {precision} | {arm_name} | seed={seed} ...",
                            flush=True,
                        )
                        ep_recs, summary = run_single_experiment(
                            benchmark=benchmark,
                            arm_name=arm_name,
                            use_barron_loss=use_barron_loss,
                            use_barron_transport=use_barron_transport,
                            precision=precision,
                            seed=seed,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            steps_per_epoch=args.steps_per_epoch,
                            data_dir=args.data_dir,
                            step_jsonl_handle=step_f,
                            use_wandb=args.wandb,
                        )
                        all_epoch_records.extend(ep_recs)
                        all_summaries.append(summary)
                        step_f.flush()
                        print(
                            f"    -> Test Acc: {summary['final_test_acc']:.2f}% | "
                            f"fp64 Loss: {summary['final_test_loss_fp64']:.4f} | "
                            f"Flat Vertices: {summary['mean_flat_vertex_fraction_pct']:.2f}% | "
                            f"Cliff Rate: {summary['mean_cliff_exceedance_rate_pct']:.1f}% | "
                            f"H(W): {summary['mean_transport_entropy']:.3f} ({summary['elapsed_sec']:.1f}s)",
                            flush=True,
                        )

    pd.DataFrame(all_epoch_records).to_csv(epoch_csv_path, index=False)
    pd.DataFrame(all_summaries).to_csv(summary_csv_path, index=False)
    with open(summary_json_path, "w", encoding="utf-8") as jf:
        json.dump(all_summaries, jf, indent=2)

    print(f"\nSuite completed in {time.time() - suite_start:.1f}s. Saved telemetry to {out_dir}/")


if __name__ == "__main__":
    main()
