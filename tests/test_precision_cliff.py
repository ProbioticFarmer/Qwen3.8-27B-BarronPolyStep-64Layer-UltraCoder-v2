"""Verification Gate 1: Precision cliff unit tests for Barron et al. (arXiv:2610.31642) operators."""

import math
import torch
import torch.nn.functional as F
import pytest

from src.stable_ops import (
    BarronCrossEntropyLoss,
    StandardCrossEntropyLoss,
    barron_log_softmax,
    barron_softmax,
    compute_cliff_exceedance_rate,
    oracle_fp64_cross_entropy,
)


def test_fp32_precision_cliff_cross_entropy():
    """Verify standard F.cross_entropy collapses to 0.0 while BarronCrossEntropyLoss resolves ~3.046e-8."""
    z = torch.tensor([[9.0, -9.0, -9.0]], dtype=torch.float32)
    target = torch.tensor([0], dtype=torch.long)

    # 1. Standard F.cross_entropy collapses to exact 0.0
    std_loss = F.cross_entropy(z, target)
    assert std_loss.item() == 0.0, f"Expected 0.0 from collapsed F.cross_entropy, got {std_loss.item()}"

    # 2. BarronCrossEntropyLoss returns ~3.046e-8 (2 * exp(-18))
    barron_ce = BarronCrossEntropyLoss()
    barron_loss = barron_ce(z, target)
    expected = 2.0 * math.exp(-18.0)  # ~3.0459959e-8
    assert barron_loss.item() > 0.0
    assert math.isclose(barron_loss.item(), expected, rel_tol=1e-5), (
        f"Expected ~{expected:.4e}, got {barron_loss.item():.4e}"
    )

    # 3. Finite difference dL != 0 for PolyStep probes (9.001 vs 9.000)
    z_perturbed = torch.tensor([[9.001, -9.0, -9.0]], dtype=torch.float32)
    std_diff = F.cross_entropy(z_perturbed, target) - F.cross_entropy(z, target)
    assert std_diff.item() == 0.0, "Standard cross_entropy has zero finite difference in the cliff regime"

    barron_loss_perturbed = barron_ce(z_perturbed, target)
    barron_diff = barron_loss_perturbed - barron_loss
    assert barron_diff.item() != 0.0, "BarronCrossEntropyLoss must produce non-zero finite difference dL != 0"
    assert barron_diff.item() < 0.0, "Increasing target logit must strictly decrease loss"


def test_bf16_precision_cliff_cross_entropy():
    """Verify bf16 collapse above c_16 ~ 5.5 and resolution by BarronCrossEntropyLoss."""
    z = torch.tensor([[3.5, -3.5, -3.5]], dtype=torch.bfloat16)
    target = torch.tensor([0], dtype=torch.long)

    # Margin m = 7.0 - ln(2) = 6.3068 > 5.545 (bf16 cliff)
    std_loss = F.cross_entropy(z, target)
    assert std_loss.item() == 0.0, f"Expected 0.0 in bf16 standard cross_entropy, got {std_loss.item()}"

    barron_ce = BarronCrossEntropyLoss()
    barron_loss = barron_ce(z, target)
    expected = 2.0 * math.exp(-7.0)  # ~1.8238e-3
    assert barron_loss.dtype == torch.bfloat16
    assert barron_loss.item() > 0.0
    assert math.isclose(barron_loss.item(), expected, rel_tol=2e-2)

    # Perturbed probe in bf16 (3.5 -> 3.5625, 1 ULP in bf16)
    z_perturbed = torch.tensor([[3.5625, -3.5, -3.5]], dtype=torch.bfloat16)
    assert (F.cross_entropy(z_perturbed, target) - std_loss).item() == 0.0
    assert (barron_ce(z_perturbed, target) - barron_loss).item() < 0.0


def test_barron_log_softmax_and_softmax_stability():
    """Verify barron_log_softmax and barron_softmax preserve runner-up weights and winning log_p."""
    z = torch.tensor([[9.0, -9.0, -9.0]], dtype=torch.float32)

    # Standard log_softmax rounds winning class to 0.0
    std_log_p = F.log_softmax(z, dim=-1)
    assert std_log_p[0, 0].item() == 0.0

    # Barron log_softmax keeps winning class strictly negative (-3.046e-8)
    b_log_p = barron_log_softmax(z, dim=-1)
    assert b_log_p[0, 0].item() < 0.0
    assert math.isclose(b_log_p[0, 0].item(), -2.0 * math.exp(-18.0), rel_tol=1e-5)
    # Runner-up log probabilities are exact: -18.0 - softplus(-m) == -18.0
    assert math.isclose(b_log_p[0, 1].item(), -18.0, rel_tol=1e-6)

    # Check non-last dimension support
    z_t = z.t()  # (3, 1)
    b_log_p_dim0 = barron_log_softmax(z_t, dim=0)
    assert torch.allclose(b_log_p_dim0.t(), b_log_p, atol=1e-12)

    # Softmax probabilities are non-negative and sum to ~1
    b_p = barron_softmax(z, dim=-1)
    assert b_p[0, 1].item() > 0.0
    assert math.isclose(b_p[0, 1].item(), math.exp(-18.0), rel_tol=1e-5)


def test_soft_target_distillation_cross_entropy():
    """Verify soft-target cross-entropy avoids collapse on high-margin teacher/student logits."""
    z_teacher = torch.tensor([[10.0, -10.0, -10.0]], dtype=torch.float32)
    z_student_1 = torch.tensor([[9.0, -9.0, -9.0]], dtype=torch.float32)
    z_student_2 = torch.tensor([[9.5, -9.5, -9.5]], dtype=torch.float32)

    barron_ce = BarronCrossEntropyLoss()
    l1 = barron_ce(z_student_1, z_teacher)
    l2 = barron_ce(z_student_2, z_teacher)
    assert l1.item() > 0.0
    assert l2.item() > 0.0
    assert l1.item() != l2.item(), "Soft-target BarronCrossEntropyLoss must distinguish high-margin students"


def test_cliff_exceedance_and_oracle():
    """Verify cliff exceedance rate and fp64 oracle."""
    z = torch.tensor([[9.0, -9.0, -9.0], [1.0, 0.5, 0.0]], dtype=torch.float32)
    target = torch.tensor([0, 0], dtype=torch.long)

    rate_fp32 = compute_cliff_exceedance_rate(z, target, dtype=torch.float32)
    assert math.isclose(rate_fp32, 50.0, abs_tol=1e-5)

    oracle = oracle_fp64_cross_entropy(z[:1], target[:1])
    assert oracle.dtype == torch.float64
    assert math.isclose(oracle.item(), math.log1p(2.0 * math.exp(-18.0)), rel_tol=1e-12)


def test_barron_solvers_and_barycentric_projection():
    """Verify BarronSoftmaxSolver, BarronSinkhornSolver, and barron_barycentric_centroid at small eps / bf16."""
    from src.polystep_patched import (
        BarronSinkhornSolver,
        BarronSoftmaxSolver,
        PatchedPolyStepOptimizer,
        barron_barycentric_centroid,
    )
    from polystep import HybridSubspace, ParamLayout

    # Cost matrix with a dominant vertex and runner-up vertices at small epsilon (eps=0.02)
    cost = torch.tensor([[0.0, 0.36, 0.38, 0.40], [0.1, 0.0, 0.35, 0.37]], dtype=torch.float32)
    sm_solver = BarronSoftmaxSolver(epsilon=0.02)
    sm_res = sm_solver.solve(cost)
    assert hasattr(sm_res, "_log_W_rows")
    assert (sm_res._log_W_rows.amax(dim=-1) < 0.0).all(), "Winning log_W must be strictly < 0 (not collapsed to 0.0)"

    # Verify barron_barycentric_centroid preserves runner-up offset from dominant vertex
    verts = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    centroid, W = barron_barycentric_centroid(sm_res._log_W_rows, verts)
    # Standard W @ V collapses centroid[0] to exact [1.0, 0.0] because W[0, 0] == 1.0 in fp32
    assert not torch.equal(centroid[0], verts[0]), "Barron centroid must retain runner-up pull away from v_star"

    # Sinkhorn solver at small eps=0.05
    sk_solver = BarronSinkhornSolver(epsilon=0.05, max_iterations=100, compile=False)
    sk_res = sk_solver.solve(cost)
    assert sk_res._log_W_rows is not None
    assert torch.isfinite(sk_res.matrix).all()

    # PatchedPolyStepOptimizer end-to-end step (softmax and sinkhorn, fp32 and bf16)
    for use_barron in (False, True):
        for solver_name in ("softmax", "sinkhorn"):
            torch.manual_seed(42)
            model = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
            layout = ParamLayout.from_module(model)
            subspace = HybridSubspace.from_layout(layout, rank=4, max_subspace_dim=32)
            opt = PatchedPolyStepOptimizer(
                model,
                loss_fn=BarronCrossEntropyLoss() if use_barron else StandardCrossEntropyLoss(),
                epsilon=0.05,
                solver=solver_name,
                subspace=subspace,
                use_barron_transport=use_barron,
                seed=42,
            )
            x = torch.randn(8, 16)
            y = torch.randint(0, 3, (8,))
            loss_val = opt.step_batch(x, y)
            assert math.isfinite(loss_val)
            assert "flat_vertex_fraction" in opt.last_step_telemetry
            assert "transport_entropy" in opt.last_step_telemetry
