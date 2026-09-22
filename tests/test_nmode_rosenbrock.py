"""Equation, Jacobian backpropagation and order checks for Rosenbrock nmODE."""

import math

import pytest
import torch
from torch import nn

from nnunetv2.nets.nmODEs import NMODEEulerDecoderCell, NMODERosenbrockDecoderCell


def test_known_update_differs_from_euler():
    h, v = torch.tensor([0.0]), torch.tensor([math.pi / 2])
    cell = NMODERosenbrockDecoderCell(4, 1, nn.Identity())
    result, history = cell([], [v], h)
    assert history is None
    assert result.item() == pytest.approx(2 / 9)
    euler, _ = NMODEEulerDecoderCell(4, 1, nn.Identity())([], [v], h)
    assert euler.item() == pytest.approx(1 / 4)


def test_zero_equilibrium_zero_step_and_zero_jacobian():
    cell = NMODERosenbrockDecoderCell(1, 1, nn.Identity())
    result, _ = cell([], [torch.zeros(3)], torch.zeros(3))
    assert torch.equal(result, torch.zeros(3))
    h = torch.tensor([0.1, 0.4, 0.8])
    v = math.pi / 4 - h  # J = -1 + sin(pi/2) = 0.
    result, _ = cell([], [v], h, step=0)
    assert torch.equal(result, h)
    result, _ = cell([], [v], h, step=0.6)
    euler, _ = NMODEEulerDecoderCell(1, 1, nn.Identity())([], [v], h, step=0.6)
    torch.testing.assert_close(result, euler)


@pytest.mark.parametrize("delta", [0.0, 1e-6, 0.25, 1.0])
def test_diagonal_system_and_convex_bounds(delta):
    torch.manual_seed(42)
    h = torch.cat([torch.rand(2048), torch.zeros(32), torch.ones(32)])
    v = torch.randn_like(h) * 10
    cell = NMODERosenbrockDecoderCell(1, 1, nn.Identity())
    result, _ = cell([], [v], h, step=delta)
    # Check the defining linear system rather than just duplicating the update.
    j = -1 + torch.sin(2 * (h + v))
    denominator = 1 - 0.5 * delta * j
    beta = delta / denominator
    assert (denominator >= 1).all()
    assert ((beta >= 0) & (beta <= delta)).all()
    assert ((result >= 0) & (result <= 1)).all()
    residual = denominator * (result - h) - delta * (torch.sin(h + v).square() - h)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-7, rtol=0)


def _scalar_update(h, v, delta):
    """Independent double-precision reference for finite differences."""
    f = -h + math.sin(h + v) ** 2
    j = -1 + math.sin(2 * (h + v))
    return h + delta * f / (1 - delta * j / 2)


def test_gradients_include_jacobian_denominator():
    values = [0.2, -0.6, 0.7]
    args = [torch.tensor(value, requires_grad=True) for value in values]
    h, v, delta = args
    result, _ = NMODERosenbrockDecoderCell(1, 1, nn.Identity())([], [v], h, delta)
    result.backward()
    epsilon = 1e-5
    for i, arg in enumerate(args):
        plus, minus = values.copy(), values.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        expected = (_scalar_update(*plus) - _scalar_update(*minus)) / (2 * epsilon)
        assert arg.grad.item() == pytest.approx(expected, rel=2e-5, abs=2e-6)
    # These inputs make the derivative of the denominator materially nonzero.
    z = h.detach() + v.detach()
    beta = delta.detach() / (1 + delta.detach() / 2 * (1 - torch.sin(2 * z)))
    detached_jacobian_v_grad = beta * torch.sin(2 * z)
    assert abs(v.grad - detached_jacobian_v_grad) > 1e-3


def _rk4_reference(h, v, duration, count):
    dt = duration / count
    def f(state):
        return -state + math.sin(state + v) ** 2
    for _ in range(count):
        k1 = f(h)
        k2 = f(h + dt * k1 / 2)
        k3 = f(h + dt * k2 / 2)
        k4 = f(h + dt * k3)
        h += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return h


def test_second_order_for_fixed_external_input():
    # Choose a nontrivial trajectory whose error stays above FP32 roundoff.
    initial, v, duration = 0.1, 0.8, 1.0
    reference = _rk4_reference(initial, v, duration, 4096)
    assert abs(reference - _rk4_reference(initial, v, duration, 8192)) < 1e-12
    errors = []
    cell = NMODERosenbrockDecoderCell(1, 1, nn.Identity())
    for count in (4, 8, 16):
        h = torch.tensor(initial)
        for _ in range(count):
            h, _ = cell([], [torch.tensor(v)], h, step=duration / count)
        errors.append(abs(h.item() - reference))
    for coarse, fine in zip(errors, errors[1:]):
        assert 3.2 < coarse / fine < 4.8


def test_projects_selected_scale_once():
    projection = nn.Linear(3, 3)
    cell = NMODERosenbrockDecoderCell(3, 2, projection)
    skips = [torch.randn(2, 3) for _ in range(3)]
    calls = []
    handle = projection.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    try:
        state, history = cell([torch.ones(2, 3)], skips, torch.zeros(2, 3))
    finally:
        handle.remove()
    assert len(calls) == 1 and calls[0] is skips[1]
    assert history is None
    state.sum().backward()
    assert torch.isfinite(projection.weight.grad).all()
    assert projection.weight.grad.abs().sum() > 0


def test_invalid_stage_or_nonscalar_step():
    for count, stage in [(0, 1), (2, 0), (2, 3)]:
        with pytest.raises(ValueError, match="current_stage"):
            NMODERosenbrockDecoderCell(count, stage, nn.Identity())
    cell = NMODERosenbrockDecoderCell(1, 1, nn.Identity())
    with pytest.raises(ValueError, match="scalar step"):
        cell([], [torch.zeros(2)], torch.zeros(2), step=torch.ones(2))
