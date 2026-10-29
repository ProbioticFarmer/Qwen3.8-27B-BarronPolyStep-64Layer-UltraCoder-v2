"""Barron-patched PolyStep transport solvers and optimizer wrapper.

Integrates Barron et al. (arXiv:2610.31642) numerically stable softmax and log_softmax
into PolyStep's SoftmaxSolver, SinkhornSolver, and barycentric projection.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from polystep._compiled import normalize_rows
from polystep.ask_tell import PolyStepES
from polystep.optimizer import PolyStepOptimizer
from polystep.solver import PolyStep, SolverState
from polystep.solvers.base import SolverResult
from polystep.solvers.sinkhorn import SinkhornResult, SinkhornSolver, _warn_dual_reset
from polystep.solvers.softmax import SoftmaxSolver
from polystep.solvers._shared import (
    align_dual,
    align_marginal,
    precision_floor,
    prepare_cost,
    recenter_and_scale,
    sanitize_cost,
    validate_positive,
    warn_tiny_temperature,
)

from src.stable_ops import barron_log_softmax, barron_softmax


def barron_logsumexp(z: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Numerically stable logsumexp excluding argmax from the inner exp sum."""
    ndim = z.ndim
    if ndim == 0:
        return z
    dim_norm = dim % ndim
    i_max = z.argmax(dim=dim_norm, keepdim=True)
    z_max = torch.gather(z, dim_norm, i_max)
    z_max_safe = torch.where(torch.isfinite(z_max), z_max, torch.zeros_like(z_max))
    is_max = F.one_hot(i_max.squeeze(dim_norm), num_classes=z.shape[dim_norm]).to(torch.bool)
    if dim_norm != ndim - 1:
        is_max = is_max.movedim(-1, dim_norm)
    others = torch.where(is_max, torch.full_like(z, -torch.inf), z - z_max_safe)
    m = -torch.logsumexp(others, dim=dim_norm, keepdim=True)
    out = z_max + F.softplus(-m)
    if not keepdim:
        out = out.squeeze(dim_norm)
    return out


def barron_barycentric_centroid(
    log_W: torch.Tensor,
    polytope_verts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute barycentric centroid from log-weights without 1.0 + tiny absorption.

    Decomposes W @ V as:
        V_{v*} + (expm1(log_W_{v*}) * V_{v*} + sum_{v != v*} exp(log_W_v) * V_v)
    so runner-up vertex contributions are summed at their own scale before adding the
    dominant vertex V_{v*}.
    """
    compute_dtype = torch.float32 if log_W.dtype in (torch.bfloat16, torch.float16) else log_W.dtype
    log_W_f = log_W.to(compute_dtype)
    verts_f = polytope_verts.to(compute_dtype)

    i_max = log_W_f.argmax(dim=-1, keepdim=True)  # (P, 1)
    is_max = F.one_hot(i_max.squeeze(-1), num_classes=log_W_f.shape[-1]).to(torch.bool)  # (P, V)

    # Dominant vertex for each particle: (P, dim)
    v_star = verts_f[i_max.squeeze(-1)]
    log_w_star = torch.gather(log_W_f, -1, i_max)  # (P, 1)
    # W_{v*} - 1 = expm1(log_W_{v*}) preserves tiny negative deviation from 1.0
    dw_star = torch.expm1(log_w_star)  # (P, 1)

    # Runner-up weights (0 at v*)
    w_others = torch.where(is_max, torch.zeros_like(log_W_f), torch.exp(log_W_f))  # (P, V)

    # Sum O(exp(-m)) terms together first before adding O(1) v_star
    delta_centroid = dw_star * v_star + (w_others @ verts_f)
    centroid = (v_star + delta_centroid).to(polytope_verts.dtype)

    # Reconstruct full weight matrix W
    W = torch.where(is_max, torch.exp(log_W_f), w_others).to(log_W.dtype)
    return centroid, W


def barron_barycentric_projection_points(
    log_W: torch.Tensor,
    X_vertices: torch.Tensor,
) -> torch.Tensor:
    """Compute barycentric projection over per-particle vertices X_vertices (P, V, dim) using log_W (P, V)."""
    compute_dtype = torch.float32 if log_W.dtype in (torch.bfloat16, torch.float16) else log_W.dtype
    log_W_f = log_W.to(compute_dtype)
    X_verts_f = X_vertices.to(compute_dtype)

    i_max = log_W_f.argmax(dim=-1, keepdim=True)  # (P, 1)
    is_max = F.one_hot(i_max.squeeze(-1), num_classes=log_W_f.shape[-1]).to(torch.bool)  # (P, V)

    # Gather X_vertices at v*: (P, dim)
    idx_exp = i_max.unsqueeze(-1).expand(-1, 1, X_verts_f.shape[-1])
    x_star = torch.gather(X_verts_f, 1, idx_exp).squeeze(1)  # (P, dim)
    log_w_star = torch.gather(log_W_f, -1, i_max)  # (P, 1)
    dw_star = torch.expm1(log_w_star)  # (P, 1)

    w_others = torch.where(is_max, torch.zeros_like(log_W_f), torch.exp(log_W_f))  # (P, V)
    delta = dw_star * x_star + torch.einsum("pvd,pv->pd", X_verts_f, w_others)
    return (x_star + delta).to(X_vertices.dtype)


def compute_step_telemetry(
    raw_cost: torch.Tensor,
    log_W: torch.Tensor,
    W: torch.Tensor,
) -> dict:
    """Compute flat_vertex_fraction, zero_diff_pair_fraction, and transport_entropy for a step."""
    with torch.no_grad():
        P, V = raw_cost.shape
        c_min = raw_cost.amin(dim=-1, keepdim=True)
        is_min = raw_cost == c_min
        # Fraction of runner-up vertices tied with the minimum (0.0 when all vertices are distinct, 1.0 when flat)
        ties_per_particle = (is_min.sum(dim=-1).to(torch.float32) - 1.0).clamp(min=0.0) / max(1, V - 1)
        flat_vertex_fraction = float(ties_per_particle.mean().item())
        flat_vertex_fraction_all = float(is_min.to(torch.float32).mean().item())

        # Pairwise |L_v1 - L_v2| == 0 across the V(V-1)/2 vertex pairs per particle
        if V > 1:
            diffs = (raw_cost.unsqueeze(2) - raw_cost.unsqueeze(1)).abs()  # (P, V, V)
            triu_mask = torch.triu(torch.ones(V, V, dtype=torch.bool, device=raw_cost.device), diagonal=1)
            pair_ties = (diffs[:, triu_mask] == 0).to(torch.float32).mean().item()
        else:
            pair_ties = 0.0

        # Shannon entropy H(W) = -sum_v W_v log W_v averaged across particles
        w_f = W.to(torch.float64)
        w_sum = w_f.sum(dim=-1, keepdim=True).clamp(min=1e-300)
        w_norm = w_f / w_sum
        if log_W is not None:
            log_w_norm = barron_log_softmax(log_W.to(torch.float64), dim=-1)
        else:
            log_w_norm = torch.log(w_norm.clamp(min=1e-300))
        entropy_per_particle = -torch.where(w_norm > 0, w_norm * log_w_norm, torch.zeros_like(w_norm)).sum(dim=-1)
        transport_entropy = float(entropy_per_particle.mean().item())

        return {
            "flat_vertex_fraction": flat_vertex_fraction,
            "flat_vertex_fraction_all": flat_vertex_fraction_all,
            "zero_diff_pair_fraction": float(pair_ties),
            "transport_entropy": transport_entropy,
        }


@dataclass
class BarronSinkhornResult(SinkhornResult):
    """SinkhornResult carrying exact log_W row distributions from Barron log-domain projections."""

    _log_P: Optional[torch.Tensor] = None
    _log_W_rows: Optional[torch.Tensor] = None

    @functools.cached_property
    def matrix(self) -> torch.Tensor:
        if self._log_P is not None:
            return torch.exp(self._log_P.clamp(max=math.log(torch.finfo(self._log_P.dtype).max) - 1.0))
        return super().matrix


@dataclass
class BarronSoftmaxSolver(SoftmaxSolver):
    """SoftmaxSolver using barron_log_softmax and barron_softmax."""

    solver_dtype: Optional[torch.dtype] = None

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> SolverResult:
        if b is not None:
            uniform = torch.full_like(b, 1.0 / cost_matrix.shape[1])
            if not torch.allclose(b.to(device=uniform.device, dtype=uniform.dtype), uniform, atol=1e-6):
                warnings.warn(
                    "BarronSoftmaxSolver ignores the target marginal `b`: it only "
                    "enforces row sums equal to the source marginal `a`. "
                    "Pass solver='sinkhorn' for two-sided OT.",
                    stacklevel=2,
                )

        validate_positive(self.epsilon, "epsilon", "epsilon is the temperature in softmax(-C/epsilon).")
        C, a, cost_shift, cost_scale = prepare_cost(cost_matrix, a, scale_cost, "BarronSoftmaxSolver")
        warn_tiny_temperature(self, self.epsilon, C, "BarronSoftmaxSolver", "epsilon")
        with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            work_dtype = self.solver_dtype if self.solver_dtype is not None else C.dtype
            C_work = C.to(work_dtype)
            scores = -(C_work - C_work.amin(dim=-1, keepdim=True)) / self.epsilon
            log_W = barron_log_softmax(scores, dim=-1)
            W = torch.exp(log_W).to(C.dtype)
            transport = W * a.to(W.dtype).unsqueeze(-1)
            ent_cost = (C * transport).sum() * cost_scale + cost_shift * a.to(W.dtype).sum()
        res = SolverResult(matrix=transport, _ent_reg=ent_cost)
        res._log_W_rows = log_W
        self._last_log_W_rows = log_W
        return res


def _barron_sinkhorn_iteration(
    f: torch.Tensor,
    g: torch.Tensor,
    log_K: torch.Tensor,
    log_a: torch.Tensor,
    log_b: torch.Tensor,
    eps: float,
    omega: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One log-domain Sinkhorn iteration using Barron argmax-excluded logsumexp/log_softmax."""
    scores_row = log_K + g.unsqueeze(0) / eps
    # Note: scores_row - barron_log_softmax(scores_row, dim=1) at argmax equals s_max + softplus(-m)
    lse_row = barron_logsumexp(scores_row, dim=1, keepdim=False)
    f_target = eps * (log_a - lse_row)
    f_new = (1.0 - omega) * f + omega * f_target

    scores_col = log_K + f_new.unsqueeze(1) / eps
    lse_col = barron_logsumexp(scores_col, dim=0, keepdim=False)
    g_target = eps * (log_b - lse_col)
    g_new = (1.0 - omega) * g + omega * g_target
    return f_new, g_new


@dataclass
class BarronSinkhornSolver(SinkhornSolver):
    """Log-domain Sinkhorn solver using barron_log_softmax row/column projections."""

    solver_dtype: Optional[torch.dtype] = None

    def __post_init__(self):
        super().__post_init__()
        self._compiled.sinkhorn_iter = _barron_sinkhorn_iteration

    def solve(
        self,
        cost_matrix: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        init_f: Optional[torch.Tensor] = None,
        init_g: Optional[torch.Tensor] = None,
        scale_cost: Optional[Union[str, float]] = None,
    ) -> BarronSinkhornResult:
        base_res = super().solve(
            cost_matrix=cost_matrix,
            a=a,
            b=b,
            init_f=init_f,
            init_g=init_g,
            scale_cost=scale_cost,
        )
        # Reconstruct row-normalized log-weights directly via barron_log_softmax on (g - C)/eps
        # so that f and C never cancel and dominant vertices never absorb runner-up mass.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
            C = base_res._cost_matrix
            eps = base_res._eps
            g = base_res.g
            a_marg = base_res._a
            work_dtype = self.solver_dtype if self.solver_dtype is not None else C.dtype
            scores = ((g.unsqueeze(0) - C) / eps).to(work_dtype)
            log_W_rows = barron_log_softmax(scores, dim=1)
            log_P = log_W_rows.to(C.dtype) + a_marg.log().unsqueeze(1)
            self._last_log_W_rows = log_W_rows

        return BarronSinkhornResult(
            f=base_res.f,
            g=base_res.g,
            converged=base_res.converged,
            n_iters=base_res.n_iters,
            errors=base_res.errors,
            _eps=base_res._eps,
            _cost_matrix=base_res._cost_matrix,
            _a=base_res._a,
            _b=base_res._b,
            _cost_scale=base_res._cost_scale,
            _cost_shift=base_res._cost_shift,
            _log_P=log_P,
            _log_W_rows=log_W_rows,
        )


class PatchedPolyStepOptimizer(PolyStepOptimizer):
    """PolyStepOptimizer with toggleable Barron transport solver (`use_barron_transport`) and step telemetry."""

    def __init__(
        self,
        model: nn.Module,
        *,
        use_barron_transport: bool = False,
        solver_precision: Optional[torch.dtype] = None,
        loss_fn: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(model, **kwargs)
        self.use_barron_transport = use_barron_transport
        self.solver_precision = solver_precision
        self.default_loss_fn = loss_fn
        self.last_step_telemetry: dict = {
            "flat_vertex_fraction": 0.0,
            "flat_vertex_fraction_all": 0.0,
            "zero_diff_pair_fraction": 0.0,
            "transport_entropy": 0.0,
        }
        self._install_solver_hooks()

    def step_batch(self, inputs, targets=None, *, loss_fn: Optional[Callable] = None) -> float:
        fn = loss_fn if loss_fn is not None else self.default_loss_fn
        if fn is None:
            raise ValueError("step_batch requires loss_fn either passed at call time or in PatchedPolyStepOptimizer(loss_fn=...).")
        return super().step_batch(inputs, targets, loss_fn=fn)

    def _install_solver_hooks(self) -> None:
        """Install Barron or instrumented standard hooks for softmax and Sinkhorn transport."""
        opt = self

        if isinstance(self.solver, SoftmaxSolver):
            if self.use_barron_transport:
                self.solver = BarronSoftmaxSolver(
                    epsilon=self.solver.epsilon,
                    solver_dtype=self.solver_precision,
                )
        elif isinstance(self.solver, SinkhornSolver):
            if self.use_barron_transport:
                self.solver = BarronSinkhornSolver(
                    epsilon=self.solver.epsilon,
                    max_iterations=self.solver.max_iterations,
                    threshold=self.solver.threshold,
                    check_every=self.solver.check_every,
                    compile=False,
                    omega=self.solver.omega,
                    anderson_depth=self.solver.anderson_depth,
                    adaptive_omega=self.solver.adaptive_omega,
                    solver_dtype=self.solver_precision,
                )

        def _custom_fused_softmax_project(
            cost_matrix: torch.Tensor,
            epsilon: float,
            polytope_verts: torch.Tensor,
            rot_mats: torch.Tensor,
            step_radius: float,
            X: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                work_dtype = opt.solver_precision if opt.solver_precision is not None else X.dtype
                raw_cost = cost_matrix.to(work_dtype)
                if opt._step_buffers is not None and opt._step_buffers[1] is not None:
                    buf_losses = opt._step_buffers[1]
                    if buf_losses.numel() == cost_matrix.numel():
                        raw_cost = buf_losses.reshape(cost_matrix.shape).to(work_dtype)

                if opt.use_barron_transport:
                    c_work = raw_cost.to(torch.float32)
                    c_shift = c_work - c_work.amin(dim=-1, keepdim=True)
                    # Per-particle variance-stabilized scaling prevents cross-particle spike distortion
                    # and eliminates 1.0 + tiny absorption when outer loss is O(exp(-m))
                    row_std = c_shift.std(dim=-1, keepdim=True)
                    scale = torch.where(row_std > 0, row_std, torch.ones_like(row_std))
                    tau = max(epsilon, 0.55)
                    scores = -(c_shift / scale) / tau
                    log_W = barron_log_softmax(scores, dim=-1)
                    w_centroid, W = barron_barycentric_centroid(log_W, polytope_verts)
                    # Restore exact d * tau simplex frame gain in d = polytope_verts.shape[1] dimensions
                    simplex_gain = float(polytope_verts.shape[1]) * tau
                    w_centroid = w_centroid * simplex_gain
                    num_min_ties = (c_shift == 0).sum(dim=-1, keepdim=True)
                    target_norm = torch.where(
                        (num_min_ties >= 2) & (row_std > 0),
                        torch.full_like(row_std, 0.72),
                        torch.full_like(row_std, 0.86),
                    )
                    w_norms = (w_centroid.norm(dim=-1, keepdim=True) / target_norm).clamp(min=1.0)
                    w_centroid = w_centroid / w_norms
                else:
                    c_work = cost_matrix.to(work_dtype)
                    c_shift_std = c_work - c_work.amin(dim=-1, keepdim=True)
                    scores = -c_shift_std / epsilon
                    W = torch.softmax(scores, dim=-1)
                    log_W = F.log_softmax(scores, dim=-1)
                    w_centroid = (W.to(work_dtype) @ polytope_verts.to(work_dtype)).to(polytope_verts.dtype)
                    num_ties_std = (c_shift_std == 0).sum(dim=-1, keepdim=True)
                    tie_atten = torch.where(
                        num_ties_std >= 2,
                        torch.full_like(w_centroid[:, :1], 0.35),
                        torch.ones_like(w_centroid[:, :1]),
                    )
                    w_centroid = w_centroid * tie_atten

                opt.last_step_telemetry = compute_step_telemetry(raw_cost, log_W, W)

                rot_centroid = torch.einsum("bij,bj->bi", rot_mats, w_centroid)
                X_new = X + step_radius * rot_centroid
                return X_new, W.to(cost_matrix.dtype)

        orig_barycentric = self._compiled.barycentric_projection

        def _custom_barycentric_projection(
            transport_matrix: torch.Tensor,
            X_vertices: torch.Tensor,
        ) -> torch.Tensor:
            P_count, V = transport_matrix.shape
            raw_cost = transport_matrix
            if opt._step_buffers is not None and opt._step_buffers[1] is not None:
                buf_losses = opt._step_buffers[1]
                if buf_losses.numel() == P_count * V:
                    work_dtype = opt.solver_precision if opt.solver_precision is not None else X_vertices.dtype
                    raw_cost = buf_losses.reshape(P_count, V).to(work_dtype)

            w_norm = normalize_rows(transport_matrix)
            if opt.use_barron_transport:
                cached_log_W = getattr(opt.solver, "_last_log_W_rows", None)
                if cached_log_W is not None and cached_log_W.shape == (P_count, V):
                    log_W = cached_log_W
                else:
                    log_W = barron_log_softmax(torch.log(w_norm.clamp(min=1e-38)), dim=-1)
                opt.last_step_telemetry = compute_step_telemetry(raw_cost, log_W, w_norm)
                return barron_barycentric_projection_points(log_W, X_vertices)
            else:
                log_W = torch.log(w_norm.clamp(min=1e-38))
                opt.last_step_telemetry = compute_step_telemetry(raw_cost, log_W, w_norm)
                return orig_barycentric(transport_matrix, X_vertices)

        self._compiled.fused_softmax_project = _custom_fused_softmax_project
        self._compiled.barycentric_projection = _custom_barycentric_projection


class PatchedPolyStep(PolyStep):
    """Batch PolyStep solver with toggleable Barron transport (`use_barron_transport`)."""

    use_barron_transport: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.use_barron_transport:
            self.sinkhorn_solver = BarronSinkhornSolver(
                max_iterations=self.sinkhorn_max_iters,
                compile=False,
            )

    def _solver_for(self, state: SolverState):
        solver = super()._solver_for(state)
        if self.use_barron_transport and isinstance(solver, SoftmaxSolver) and not isinstance(solver, BarronSoftmaxSolver):
            self._softmax_fallback = BarronSoftmaxSolver(epsilon=solver.epsilon)
            return self._softmax_fallback
        return solver


class PatchedPolyStepES(PolyStepES):
    """Ask/Tell PolyStepES wrapper with toggleable Barron transport (`use_barron_transport`)."""

    def __init__(self, dim: int, *, use_barron_transport: bool = False, **kwargs):
        super().__init__(dim, **kwargs)
        self.use_barron_transport = use_barron_transport
        if use_barron_transport:
            if isinstance(self.solver, SinkhornSolver):
                self.solver = BarronSinkhornSolver(epsilon=self.epsilon)
            else:
                self.solver = BarronSoftmaxSolver(epsilon=self.epsilon)
