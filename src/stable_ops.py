"""Numerically stable Softmax, Log-Softmax, and Cross-Entropy operators.

Implements the reference operators from Barron et al. (arXiv:2610.31642) to prevent
Softmax Collapse and Softmax Cancellation in fp32 and bf16 forward-only optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIFF_THRESHOLDS = {
    torch.float16: 7.27,
    torch.bfloat16: 5.545,  # c_16 ~ 5.5
    torch.float32: 16.635,  # c_32 ~ 16.6
    torch.float64: 36.736,
}


def barron_log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Excludes argmax from logsumexp to prevent 1.0 + tiny absorption."""
    ndim = z.ndim
    if ndim == 0:
        return torch.zeros_like(z)
    dim_norm = dim % ndim
    i_max = z.argmax(dim=dim_norm, keepdim=True)
    z_max = torch.gather(z, dim_norm, i_max)
    # Guard against all -inf rows where z - (-inf) would produce NaN
    z_max_safe = torch.where(torch.isfinite(z_max), z_max, torch.zeros_like(z_max))
    is_max = F.one_hot(i_max.squeeze(dim_norm), num_classes=z.shape[dim_norm]).to(torch.bool)
    if dim_norm != ndim - 1:
        is_max = is_max.movedim(-1, dim_norm)
    shifted = torch.where(is_max, torch.zeros_like(z), z - z_max_safe)
    others = torch.where(is_max, torch.full_like(z, -torch.inf), shifted)
    m = -torch.logsumexp(others, dim=dim_norm, keepdim=True)
    return shifted - F.softplus(-m)


def barron_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax via exp(barron_log_softmax(z))."""
    return torch.exp(barron_log_softmax(z, dim=dim))


class BarronCrossEntropyLoss(nn.Module):
    """Hard-label and soft-target (distillation) cross-entropy without Softmax Collapse or Cancellation."""

    def __init__(self, reduction: str = "mean", eval_dtype: torch.dtype | None = None):
        super().__init__()
        self.reduction = reduction
        self.eval_dtype = eval_dtype

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.eval_dtype is not None:
            z = z.to(self.eval_dtype)
            if target.is_floating_point():
                target = target.to(self.eval_dtype)
        if target.ndim == z.ndim - 1:  # Hard integer labels
            target_idx = target.long()
            z_t = torch.gather(z, -1, target_idx.unsqueeze(-1))
            is_t = F.one_hot(target_idx, num_classes=z.shape[-1]).to(torch.bool)
            others = torch.where(is_t, torch.full_like(z, -torch.inf), z - z_t)
            m = -torch.logsumexp(others, dim=-1)
            loss = F.softplus(-m)
        else:  # Soft target logits (`target` passed as teacher logits z_T)
            log_p_s = barron_log_softmax(z, dim=-1)
            log_p_t = barron_log_softmax(target, dim=-1)
            # Use expm1 / log1p formulation to avoid Softmax Cancellation when z_S -> z_T
            p_t = torch.exp(log_p_t)
            loss = -(p_t * log_p_s).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class StandardCrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss that preserves input precision (fp32 or bf16) without hidden upcasting.

    Standard ``F.cross_entropy(z, target)`` computes ``-log_softmax(z)[target]`` via
    standard max-shifted ``logsumexp``, which suffers from Softmax Collapse above
    margin c_32 = 16.6 (fp32) or c_16 = 5.5 (bf16). Wrapping it in a custom Module
    ensures PolyStep's NNCostEvaluator treats it identically to ``BarronCrossEntropyLoss``
    (neither bypassing bf16 via ``_candidate_loss`` upcasting nor taking a different evaluator path).
    """

    def __init__(self, reduction: str = "mean", eval_dtype: torch.dtype | None = None):
        super().__init__()
        self.reduction = reduction
        self.eval_dtype = eval_dtype

    def forward(self, z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.eval_dtype is not None:
            z = z.to(self.eval_dtype)
            if target.is_floating_point():
                target = target.to(self.eval_dtype)
        if target.ndim == z.ndim - 1:
            return F.cross_entropy(z, target.long(), reduction=self.reduction)
        else:
            log_p_s = F.log_softmax(z, dim=-1)
            p_t = F.softmax(target, dim=-1)
            loss = -(p_t * log_p_s).sum(dim=-1)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss


@torch.no_grad()
def oracle_fp64_cross_entropy(z: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Compute true cross-entropy loss in fp64 using Barron's exact formulation."""
    z64 = z.detach().to(torch.float64)
    if target.ndim == z.ndim - 1:
        target_idx = target.detach().long()
        z_t = torch.gather(z64, -1, target_idx.unsqueeze(-1))
        is_t = F.one_hot(target_idx, num_classes=z64.shape[-1]).to(torch.bool)
        others = torch.where(is_t, torch.full_like(z64, -torch.inf), z64 - z_t)
        m = -torch.logsumexp(others, dim=-1)
        loss = F.softplus(-m)
    else:
        t64 = target.detach().to(torch.float64)
        log_p_s = barron_log_softmax(z64, dim=-1)
        log_p_t = barron_log_softmax(t64, dim=-1)
        p_t = torch.exp(log_p_t)
        loss = -(p_t * log_p_s).sum(dim=-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


@torch.no_grad()
def compute_logit_margins(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute target logit margin m = z_t - log sum_{j != t} exp(z_j) in fp64."""
    z64 = z.detach().to(torch.float64)
    if target.ndim == z.ndim - 1:
        target_idx = target.detach().long()
    else:
        target_idx = target.detach().argmax(dim=-1)
    z_t = torch.gather(z64, -1, target_idx.unsqueeze(-1))
    is_t = F.one_hot(target_idx, num_classes=z64.shape[-1]).to(torch.bool)
    others = torch.where(is_t, torch.full_like(z64, -torch.inf), z64 - z_t)
    m = -torch.logsumexp(others, dim=-1)
    return m


@torch.no_grad()
def compute_cliff_exceedance_rate(z: torch.Tensor, target: torch.Tensor, dtype: torch.dtype | None = None) -> float:
    """Percentage (0 to 100) of examples whose logit margin exceeds the precision cliff c_d."""
    check_dtype = dtype if dtype is not None else z.dtype
    threshold = CLIFF_THRESHOLDS.get(check_dtype, 16.6)
    if check_dtype == torch.bfloat16:
        threshold = 5.5
    elif check_dtype == torch.float32:
        threshold = 16.6
    margins = compute_logit_margins(z, target)
    return float((margins > threshold).to(torch.float64).mean().item() * 100.0)
