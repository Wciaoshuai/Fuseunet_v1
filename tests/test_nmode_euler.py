"""Euler numerical checks and shared Euler/Rosenbrock network checks."""

import io
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from nnunetv2.nets.nmODEs import NMODEEulerDecoderCell
from nnunetv2.nets.unet import PlainConvUNet
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

NMODE_MODES = [
    "nmode_euler_learned", "nmode_euler_fixed",
    "nmode_rosenbrock_learned", "nmode_rosenbrock_fixed",
]


def make_model(dim=2, mode="nmode_euler_learned", deep_supervision=False):
    return PlainConvUNet(
        input_channels=1, n_stages=3, features_per_stage=(4, 8, 12),
        conv_op=nn.Conv2d if dim == 2 else nn.Conv3d,
        kernel_sizes=(3, 3, 3),
        strides=((1,) * dim, (2,) * dim, (2,) * dim),
        n_conv_per_stage=(1, 1, 1), num_classes=2,
        n_conv_per_stage_decoder=(1, 1), conv_bias=True,
        norm_op=None, nonlin=nn.ReLU, nonlin_kwargs={},
        deep_supervision=deep_supervision, decoder_discretization=mode,
    )


def test_original_sine_squared_equation_and_local_derivatives():
    cell = NMODEEulerDecoderCell(4, 1, nn.Identity())
    h = torch.tensor([0.0, 0.2, 0.7], requires_grad=True)
    v = torch.tensor([math.pi / 2, -0.8, 0.4], requires_grad=True)
    delta = torch.tensor(0.25, requires_grad=True)
    result, history = cell([], [v], h, step=delta)
    assert history is None
    assert result[0].item() == pytest.approx(0.25)
    expected = h + delta * (-h + torch.sin(h + v).square())
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    torch.testing.assert_close(h.grad, 1 - delta + delta * torch.sin(2 * (h + v)))
    torch.testing.assert_close(v.grad, delta * torch.sin(2 * (h + v)))
    torch.testing.assert_close(delta.grad, (torch.sin(h + v).square() - h).sum())


def test_zero_input_zero_state_and_single_stage():
    cell = NMODEEulerDecoderCell(1, 1, nn.Identity())
    y, _ = cell([], [torch.zeros(3)], torch.zeros(3))
    assert torch.equal(y, torch.zeros(3))
    y, _ = cell([], [torch.full((3,), math.pi / 2)], torch.zeros(3))
    torch.testing.assert_close(y, torch.ones(3))


@pytest.mark.parametrize("mode", NMODE_MODES)
def test_scale_allocation_and_parameter_overhead(mode):
    model = make_model(mode=mode)
    steps = model.decoder.get_scale_steps()
    torch.testing.assert_close(steps, torch.full((3,), 1 / 3))
    assert steps.dtype == torch.float32
    assert (steps > 0).all()
    torch.testing.assert_close(steps.sum(), torch.tensor(1.0))
    if mode.endswith("learned"):
        assert steps.requires_grad
        with torch.no_grad():
            model.decoder.scale_logits.copy_(torch.tensor([-3.0, 0.0, 4.0]))
        steps = model.decoder.get_scale_steps()
        assert steps[0] < steps[1] < steps[2]
        torch.testing.assert_close(steps.sum(), torch.tensor(1.0))
        fixed = make_model(mode=mode.replace("_learned", "_fixed"))
        assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in fixed.parameters()) == 3
    else:
        assert not steps.requires_grad
        assert not hasattr(model.decoder, "scale_logits")


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("deep_supervision", [False, True])
@pytest.mark.parametrize("solver", ["euler", "rosenbrock"])
def test_equal_initial_weights_give_equal_fixed_and_learned_predictions(dim, deep_supervision, solver):
    learned = make_model(dim, f"nmode_{solver}_learned", deep_supervision).eval()
    fixed = make_model(dim, f"nmode_{solver}_fixed", deep_supervision).eval()
    fixed.load_state_dict({k: v for k, v in learned.state_dict().items() if k != "decoder.scale_logits"})
    x = torch.randn(1, 1, *((12, 16) if dim == 2 else (8, 12, 16)))
    a, b = learned(x), fixed(x)
    if not deep_supervision:
        a, b = [a], [b]
    assert len(a) == len(b)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mode", NMODE_MODES)
@pytest.mark.parametrize("deep_supervision", [False, True])
def test_network_shapes_bounds_backward_and_planning_proxy(dim, mode, deep_supervision):
    torch.manual_seed(18)
    model = make_model(dim, mode, deep_supervision)
    spatial = (12, 16) if dim == 2 else (8, 12, 16)
    x = torch.randn(2, 1, *spatial, requires_grad=True)
    states, conv_sizes = [], []
    handles = [stage.register_forward_hook(lambda module, args, out: states.append(out[0]))
               for stage in model.decoder.stages]
    for projection in model.decoder.upsamples:
        handles.append(projection.conv.register_forward_hook(
            lambda module, args, out: conv_sizes.append(out[0].numel())))
    for head in model.decoder.seg_layers:
        handles.append(head.register_forward_hook(
            lambda module, args, out: conv_sizes.append(out[0].numel())))
    try:
        prediction = model(x)
    finally:
        for handle in handles:
            handle.remove()
    assert len(states) == 3
    for state in states:
        assert state.dtype == torch.float32
        assert torch.isfinite(state).all()
        assert state.min() >= -1e-7 and state.max() <= 1 + 1e-7
    if deep_supervision:
        # Same L-1 target pyramid as nnUNetTrainer._get_deep_supervision_scales.
        assert len(prediction) == 2
        assert prediction[0].shape == (2, 2, *spatial)
        assert prediction[1].shape == (2, 2, *(s // 2 for s in spatial))
        targets = [torch.randn_like(p) for p in prediction]
        loss = DeepSupervisionWrapper(nn.MSELoss(), (1.0, 0.5))(prediction, targets)
        model.decoder.deep_supervision = False
        torch.testing.assert_close(model(x), prediction[0])
        model.decoder.deep_supervision = True
    else:
        assert prediction.shape == (2, 2, *spatial)
        loss = (prediction - torch.randn_like(prediction)).square().mean()
    assert model.decoder.compute_conv_feature_map_size(spatial) == sum(conv_sizes)
    assert model.compute_conv_feature_map_size(spatial) > 0
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for projection in model.decoder.upsamples:
        grad = projection.conv.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    if mode.endswith("learned"):
        grad = model.decoder.scale_logits.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        before = model.decoder.get_scale_steps().detach().clone()
        torch.optim.SGD([model.decoder.scale_logits], lr=0.5).step()
        assert not torch.equal(before, model.decoder.get_scale_steps())


@pytest.mark.parametrize("mode", NMODE_MODES)
def test_checkpoint_round_trip(mode):
    model = make_model(mode=mode).eval()
    if mode.endswith("learned"):
        with torch.no_grad():
            model.decoder.scale_logits.copy_(torch.tensor([0.1, -0.7, 0.8]))
    x = torch.randn(1, 1, 12, 16)
    prediction = model(x)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = make_model(mode=mode).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(x), prediction, rtol=0, atol=0)
    torch.testing.assert_close(restored.decoder.get_scale_steps(), model.decoder.get_scale_steps())


@pytest.mark.parametrize("mode", ["nmode_euler_learned", "nmode_rosenbrock_learned"])
def test_cpu_autocast_keeps_integration_fp32_and_gradients(mode):
    model = make_model(mode=mode)
    x = torch.randn(1, 1, 12, 16, requires_grad=True)
    states = []
    handles = [s.register_forward_hook(lambda module, args, out: states.append(out[0]))
               for s in model.decoder.stages]
    try:
        # Native convolution also works on CPUs without oneDNN BF16 support.
        with torch.backends.mkldnn.flags(enabled=False):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                prediction = model(x)
                loss = prediction.float().square().mean()
            loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert all(h.dtype == torch.float32 for h in states)
    assert torch.isfinite(model.decoder.scale_logits.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("mode", ["nmode_euler_learned", "nmode_rosenbrock_learned"])
def test_cuda_autocast_backward(mode):
    model = make_model(mode=mode).cuda()
    x = torch.randn(1, 1, 12, 16, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x).float().square().mean()
    loss.backward()
    assert torch.isfinite(model.decoder.scale_logits.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mode", [*NMODE_MODES, None])
def test_environment_mode_entrypoint(mode):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    if mode is None:
        env.pop("FUSEUNET_DECODER_MODE", None)
    else:
        env["FUSEUNET_DECODER_MODE"] = mode
    script = (
        "import inspect; from nnunetv2.nets.unet import PlainConvUNet; "
        "print(inspect.signature(PlainConvUNet).parameters['decoder_discretization'].default)"
    )
    result = subprocess.run([sys.executable, "-c", script], env=env, check=True,
                            capture_output=True, text=True)
    assert result.stdout.strip() == (mode or "multistep")


def test_existing_nmzoh_mode_keeps_its_parameter_structure():
    model = make_model(mode="nmzoh")
    assert not hasattr(model.decoder, "scale_logits")
    assert all(hasattr(s, "log_A") for s in model.decoder.stages)
    assert model(torch.randn(1, 1, 12, 16)).shape == (1, 2, 12, 16)
    with pytest.raises(ValueError, match="only for nmODE Euler"):
        model.decoder.get_scale_steps()
