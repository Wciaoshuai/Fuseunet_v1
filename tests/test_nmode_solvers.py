"""Independent numerical and gradient checks for the four solver ablations."""

import math
from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nnunetv2.nets import nmODEs
from nnunetv2.nets.nmODEs import (
    NMODEHeunDecoderCell,
    NMODERK4DecoderCell,
    NMODEImplicitEulerDecoderCell,
    NMODEImplicitMidpointDecoderCell,
    NMODERosenbrockDecoderCell,
    nmode_heun_step,
    nmode_rk4_step,
    nmode_implicit_step,
)

NEW_CELLS = [
    NMODEHeunDecoderCell, NMODERK4DecoderCell,
    NMODEImplicitEulerDecoderCell, NMODEImplicitMidpointDecoderCell,
]
IMPLICIT_CELLS = [NMODEImplicitEulerDecoderCell, NMODEImplicitMidpointDecoderCell]
SOLVER_STEPS = [
    nmode_heun_step, nmode_rk4_step,
    partial(nmode_implicit_step, midpoint=False),
    partial(nmode_implicit_step, midpoint=True),
]
SOLVER_NAMES = ["heun", "rk4", "implicit_euler", "implicit_midpoint"]


def _scalar_reference(h, v, duration, count=4096):
    """Small-step double-precision trajectory, independent of model helpers."""
    dt = duration / count

    def f(value):
        return -value + math.sin(value + v) ** 2

    for _ in range(count):
        k1 = f(h)
        k2 = f(h + dt * k1 / 2)
        k3 = f(h + dt * k2 / 2)
        k4 = f(h + dt * k3)
        h += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return h


def _bisect_implicit(h, v, delta, midpoint):
    """Monotone implicit equation has a unique root in [0, 1] here."""
    lower, upper = 0.0, 1.0
    for _ in range(80):
        estimate = (lower + upper) / 2
        u = (h + estimate) / 2 if midpoint else estimate
        residual = estimate - h - delta * (-u + math.sin(u + v) ** 2)
        if residual > 0:
            upper = estimate
        else:
            lower = estimate
    return (lower + upper) / 2


@pytest.mark.parametrize("cell_type", NEW_CELLS)
def test_zero_equilibrium_zero_step_and_fp32_state(cell_type):
    cell = cell_type(1, 1, nn.Identity())
    state, history = cell([], [torch.zeros(3)], torch.zeros(3))
    assert history is None
    assert torch.equal(state, torch.zeros(3))
    old = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    result, _ = cell([], [torch.tensor([-0.8, 0.2, 1.4])], old, step=0)
    assert result.dtype == torch.float32
    assert torch.equal(result, old.float())


@pytest.mark.parametrize("solver,order", [(nmode_heun_step, 2), (nmode_rk4_step, 4)])
def test_explicit_accuracy_and_order_against_reference(solver, order):
    initial, drive, duration = 0.1, 0.8, 1.0
    reference = _scalar_reference(initial, drive, duration)
    assert abs(reference - _scalar_reference(initial, drive, duration, count=8192)) < 1e-12
    errors = []
    for count in (4, 8, 16):
        state = torch.tensor(initial, dtype=torch.float64)
        v = torch.tensor(drive, dtype=torch.float64)
        delta = torch.tensor(duration / count, dtype=torch.float64)
        for _ in range(count):
            state = solver(state, v, delta)
        assert state.dtype == torch.float64
        errors.append(abs(state.item() - reference))
    for coarse, fine in zip(errors, errors[1:]):
        assert 0.8 * 2 ** order < coarse / fine < 1.2 * 2 ** order
    assert errors[-1] < (1e-3 if order == 2 else 1e-6)


@pytest.mark.parametrize("midpoint", [False, True])
def test_implicit_result_matches_independent_bisection(midpoint):
    cases = [
        (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, math.pi / 2 - 1, 1.0),
        (0.0, math.pi / 2, 0.5), (0.2, -0.6, 0.7), (0.7, 1.4, 1.0),
        (0.9, -2.8, 0.25), (0.4, 2.2, 1e-6), (1.0, math.pi, 1.0),
    ]
    h, v, delta = torch.tensor(cases, dtype=torch.float64).unbind(dim=1)
    result = nmode_implicit_step(h, v, delta, midpoint=midpoint)
    expected = torch.tensor([_bisect_implicit(*case, midpoint) for case in cases], dtype=torch.float64)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("midpoint", [False, True])
def test_five_newton_fp32_residual_on_state_drive_and_step_grid(midpoint):
    h, v, delta = torch.meshgrid(
        torch.linspace(0, 1, 33), torch.linspace(-math.pi, math.pi, 129),
        torch.tensor([0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 0.99, 1.0]),
        indexing="ij",
    )
    cell_type = NMODEImplicitMidpointDecoderCell if midpoint else NMODEImplicitEulerDecoderCell
    cell = cell_type(1, 1, nn.Identity())
    # Cell steps are scalar; cover each boundary/interior step separately.
    for index in range(delta.shape[-1]):
        old, drive = h[..., index], v[..., index]
        step = delta[0, 0, index]
        result, _ = cell([], [drive], old, step=step)
        u = (old + result) / 2 if midpoint else result
        residual = result - old - step * (-u + torch.sin(u + drive).square())
        assert torch.isfinite(result).all()
        assert residual.abs().max().item() <= 1e-6
        assert result.min() >= -1e-6 and result.max() <= 1 + 1e-6


def test_one_newton_midpoint_equals_rosenbrock_and_five_solves_further():
    torch.manual_seed(42)
    h, v = torch.rand(128), torch.randn(128)
    delta = torch.tensor(0.7)
    one = nmode_implicit_step(h, v, delta, midpoint=True, iterations=1)
    rosenbrock, _ = NMODERosenbrockDecoderCell(1, 1, nn.Identity())([], [v], h, step=delta)
    torch.testing.assert_close(one, rosenbrock, atol=2e-7, rtol=0)
    old, drive, step = torch.tensor(0.0), torch.tensor(math.pi / 2), torch.tensor(0.5)
    result, _ = NMODEImplicitMidpointDecoderCell(1, 1, nn.Identity())([], [drive], old, step=step)
    assert result.item() == pytest.approx(0.3853345476749758, abs=1e-7)
    assert abs(result - nmode_implicit_step(old, drive, step, midpoint=True, iterations=1)) > 1e-2


@pytest.mark.parametrize("cell_type", IMPLICIT_CELLS)
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("equilibrium", [False, True])
def test_exactly_five_full_newton_iterations_in_train_and_inference(cell_type, training, equilibrium):
    cell = cell_type(1, 1, nn.Identity()).train(training)
    old = torch.tensor(0.0, requires_grad=training)
    drive = torch.tensor(0.0 if equilibrium else math.pi / 2, requires_grad=training)
    with patch.object(nmODEs, "nmode_vector_field", wraps=nmODEs.nmode_vector_field) as field:
        with patch.object(nmODEs, "nmode_state_jacobian", wraps=nmODEs.nmode_state_jacobian) as jacobian:
            with torch.set_grad_enabled(training):
                result, _ = cell([], [drive], old, step=0.5)
    assert field.call_count == jacobian.call_count == 5
    assert result.requires_grad == training
    if training:
        result.backward()
        assert torch.isfinite(old.grad) and torch.isfinite(drive.grad)


@pytest.mark.parametrize("solver", SOLVER_STEPS, ids=SOLVER_NAMES)
def test_state_drive_and_step_gradcheck(solver):
    h = torch.tensor([0.2, 0.6, 0.9], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([-0.6, 1.1, -2.5], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(solver, (h, v, delta), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("midpoint", [False, True])
def test_unrolled_implicit_gradients_match_implicit_equation(midpoint):
    h = torch.tensor([0.2, 0.6, 0.9], dtype=torch.float64, requires_grad=True)
    v = torch.tensor([-0.6, 1.1, -2.5], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    result = nmode_implicit_step(h, v, delta, midpoint=midpoint)
    result.sum().backward()
    with torch.no_grad():
        u = (h + result) / 2 if midpoint else result
        sine_derivative = torch.sin(2 * (u + v))
        j = -1 + sine_derivative
        denominator = 1 - (0.5 if midpoint else 1.0) * delta * j
        old_derivative = (1 + delta * j / 2) / denominator if midpoint else 1 / denominator
        drive_derivative = delta * sine_derivative / denominator
        step_derivative = ((-u + torch.sin(u + v).square()) / denominator).sum()
    torch.testing.assert_close(h.grad, old_derivative, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(v.grad, drive_derivative, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(delta.grad, step_derivative, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("cell_type", NEW_CELLS)
def test_projects_selected_scale_once_and_preserves_gradients(cell_type):
    torch.manual_seed(19)
    projection = nn.Linear(3, 3)
    cell = cell_type(3, 2, projection)
    skips = [torch.randn(2, 3, requires_grad=True) for _ in range(3)]
    old = torch.rand(2, 3, requires_grad=True)
    step = torch.tensor(0.4, requires_grad=True)
    calls = []
    handle = projection.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    try:
        state, history = cell([torch.ones(2, 3)], skips, old, step=step)
    finally:
        handle.remove()
    assert len(calls) == 1 and calls[0] is skips[1]
    assert history is None
    state.sum().backward()
    for tensor in [old, step, skips[1], projection.weight]:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert skips[0].grad is None and skips[2].grad is None


@pytest.mark.parametrize("cell_type", NEW_CELLS)
def test_invalid_stage_or_nonscalar_step(cell_type):
    for count, stage in [(0, 1), (2, 0), (2, 3)]:
        with pytest.raises(ValueError, match="current_stage"):
            cell_type(count, stage, nn.Identity())
    cell = cell_type(1, 1, nn.Identity())
    with pytest.raises(ValueError, match="scalar step"):
        cell([], [torch.zeros(2)], torch.zeros(2), step=torch.ones(2))
