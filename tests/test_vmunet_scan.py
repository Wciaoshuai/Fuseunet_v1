import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.nets.nmODEs import (
    _vmunet_cross_merge_2d,
    _vmunet_cross_scan_2d,
    VMUNetSS2D,
    VMUNetScanDecoderCell,
)
from nnunetv2.nets.unet import PlainConvUNet


def _vm_unet_scan_reference(x: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _, _ = x.shape
    seq_row = x.flatten(2, 3)
    seq_col = x.transpose(2, 3).contiguous().flatten(2, 3)
    return torch.stack(
        [
            seq_row,
            seq_col,
            torch.flip(seq_row, dims=[-1]),
            torch.flip(seq_col, dims=[-1]),
        ],
        dim=1,
    ).view(batch_size, 4, channels, -1)


def _vm_unet_merge_reference(y_dir: torch.Tensor, h: int, w: int) -> torch.Tensor:
    batch_size, _, channels, seq_len = y_dir.shape
    inv_y = torch.flip(y_dir[:, 2:4], dims=[-1]).reshape(batch_size, 2, channels, seq_len)
    wh_y = y_dir[:, 1].reshape(batch_size, channels, w, h).transpose(2, 3).contiguous().reshape(
        batch_size, channels, seq_len
    )
    invwh_y = inv_y[:, 1].reshape(batch_size, channels, w, h).transpose(2, 3).contiguous().reshape(
        batch_size, channels, seq_len
    )
    return (y_dir[:, 0] + inv_y[:, 0] + wh_y + invwh_y).reshape(batch_size, channels, h, w)


def test_vmunet_cross_scan_matches_known_order():
    x = torch.arange(1, 7, dtype=torch.float32).view(1, 1, 2, 3)
    scanned = _vmunet_cross_scan_2d(x)
    expected = torch.tensor(
        [
            [
                [[1, 2, 3, 4, 5, 6]],
                [[1, 4, 2, 5, 3, 6]],
                [[6, 5, 4, 3, 2, 1]],
                [[6, 3, 5, 2, 4, 1]],
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.equal(scanned, expected)


def test_vmunet_cross_merge_restores_vm_unet_layout():
    x = torch.arange(1, 13, dtype=torch.float32).view(1, 2, 2, 3)
    scanned = _vmunet_cross_scan_2d(x)
    merged = _vmunet_cross_merge_2d(scanned, 2, 3)
    assert torch.equal(merged, 4 * x)


def test_vmunet_scan_helpers_match_reference_formulas():
    x = torch.randn(2, 3, 5, 4)
    scanned = _vmunet_cross_scan_2d(x)
    scanned_ref = _vm_unet_scan_reference(x)
    assert torch.allclose(scanned, scanned_ref)

    y_dir = torch.randn_like(scanned)
    merged = _vmunet_cross_merge_2d(y_dir, 5, 4)
    merged_ref = _vm_unet_merge_reference(y_dir, 5, 4)
    assert torch.allclose(merged, merged_ref)


def test_vmunet_scan_plainconvunet_2d_smoke():
    model = PlainConvUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 32),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3),
        strides=((1, 1), (2, 2), (2, 2)),
        n_conv_per_stage=(1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(1, 1),
        conv_bias=True,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={},
        deep_supervision=False,
        decoder_discretization="vmunet_scan",
    )
    x = torch.randn(2, 1, 32, 32, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 2, 32, 32)
    assert torch.isfinite(y).all()
    y.mean().backward()
    assert x.grad is not None


def test_vmunet_ss2d_output_stays_finite():
    block = VMUNetSS2D(d_model=6, d_state=16, d_conv=3)
    x = torch.randn(2, 8, 8, 6)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_vmunet_scan_decoder_cell_output_stays_finite():
    cell = VMUNetScanDecoderCell(
        num_stage=2,
        current_stage=1,
        opx=nn.Identity(),
        conv_op=nn.Conv2d,
        hidden_channels=6,
    )
    skip = torch.randn(2, 6, 32, 32)
    state = torch.randn(2, 6, 32, 32)
    y_out, _ = cell([], [skip], state)
    assert y_out.shape == state.shape
    assert torch.isfinite(y_out).all()


def test_vmunet_scan_decoder_cell_multi_step_accumulation_stays_finite():
    cell = VMUNetScanDecoderCell(
        num_stage=8,
        current_stage=1,
        opx=nn.Identity(),
        conv_op=nn.Conv2d,
        hidden_channels=6,
    )
    skip = torch.randn(2, 6, 32, 32)
    state = torch.zeros(2, 6, 32, 32)
    for _ in range(8):
        state, _ = cell([], [skip], state)
        assert torch.isfinite(state).all()


def test_vmunet_scan_patchify_and_restore_shapes():
    cell = VMUNetScanDecoderCell(
        num_stage=2,
        current_stage=1,
        opx=nn.Identity(),
        conv_op=nn.Conv2d,
        hidden_channels=6,
    )
    fused = torch.randn(1, 6, 512, 512)
    fused_low = cell._patchify_for_scan(fused)
    assert fused_low.shape == (1, 6, 128, 128)

    restored = cell._restore_scan_output(fused_low, fused.shape[-2:])
    assert restored.shape == fused.shape


def test_vmunet_scan_skips_patchify_for_small_inputs():
    cell = VMUNetScanDecoderCell(
        num_stage=2,
        current_stage=1,
        opx=nn.Identity(),
        conv_op=nn.Conv2d,
        hidden_channels=6,
    )
    fused = torch.randn(1, 6, 3, 3)
    fused_low = cell._patchify_for_scan(fused)
    assert fused_low.shape == fused.shape


def test_vmunet_ss2d_raises_on_non_finite_inputs():
    block = VMUNetSS2D(d_model=6, d_state=16, d_conv=3)
    x = torch.randn(1, 8, 8, 6)
    x[0, 0, 0, 0] = float("nan")
    try:
        block(x)
    except FloatingPointError as exc:
        assert "VMUNetSS2D.input" in str(exc)
        return
    raise AssertionError("Expected VMUNetSS2D to reject non-finite inputs")


def test_vmunet_scan_autocast_cuda_smoke():
    if not torch.cuda.is_available():
        return

    block = VMUNetSS2D(d_model=6, d_state=16, d_conv=3).cuda()
    x = torch.randn(2, 16, 16, 6, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = block(x)
        loss = y.square().mean()

    assert torch.isfinite(y).all()
    assert torch.isfinite(loss)

    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_vmunet_scan_plainconvunet_autocast_cuda_smoke():
    if not torch.cuda.is_available():
        return

    model = PlainConvUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 32),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3),
        strides=((1, 1), (2, 2), (2, 2)),
        n_conv_per_stage=(1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(1, 1),
        conv_bias=True,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={},
        deep_supervision=False,
        decoder_discretization="vmunet_scan",
    ).cuda()

    x = torch.randn(2, 1, 32, 32, device="cuda", requires_grad=True)
    target = torch.randint(0, 2, (2, 32, 32), device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = F.cross_entropy(logits, target)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_vmunet_scan_decoder_raises_on_non_finite_stage_output():
    model = PlainConvUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 32),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3),
        strides=((1, 1), (2, 2), (2, 2)),
        n_conv_per_stage=(1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(1, 1),
        conv_bias=True,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={},
        deep_supervision=False,
        decoder_discretization="vmunet_scan",
    )

    class NaNStage(nn.Module):
        def forward(self, f, x, y):
            del f, x
            return torch.full_like(y, float("nan")), None

    model.decoder.stages[0] = NaNStage()
    x = torch.randn(2, 1, 32, 32)
    try:
        model(x)
    except FloatingPointError as exc:
        assert "MyDecoder.stage_1_output" in str(exc)
        return
    raise AssertionError("Expected vmunet_scan decoder to reject non-finite stage outputs")


def test_vmunet_scan_decoder_raises_on_non_finite_logits():
    model = PlainConvUNet(
        input_channels=1,
        n_stages=3,
        features_per_stage=(8, 16, 32),
        conv_op=nn.Conv2d,
        kernel_sizes=(3, 3, 3),
        strides=((1, 1), (2, 2), (2, 2)),
        n_conv_per_stage=(1, 1, 1),
        num_classes=2,
        n_conv_per_stage_decoder=(1, 1),
        conv_bias=True,
        norm_op=None,
        norm_op_kwargs=None,
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.ReLU,
        nonlin_kwargs={},
        deep_supervision=False,
        decoder_discretization="vmunet_scan",
    )

    class NaNSegLayer(nn.Module):
        def forward(self, y):
            return torch.full((y.shape[0], 2, *y.shape[2:]), float("nan"), device=y.device, dtype=y.dtype)

    model.decoder.seg_layers[-1] = NaNSegLayer()
    x = torch.randn(2, 1, 32, 32)
    try:
        model(x)
    except FloatingPointError as exc:
        assert "MyDecoder.seg_head_3_logits" in str(exc)
        return
    raise AssertionError("Expected vmunet_scan decoder to reject non-finite logits")


def test_vmunet_scan_rejects_3d_models():
    try:
        PlainConvUNet(
            input_channels=1,
            n_stages=3,
            features_per_stage=(8, 16, 32),
            conv_op=nn.Conv3d,
            kernel_sizes=(3, 3, 3),
            strides=((1, 1, 1), (2, 2, 2), (2, 2, 2)),
            n_conv_per_stage=(1, 1, 1),
            num_classes=2,
            n_conv_per_stage_decoder=(1, 1),
            conv_bias=True,
            norm_op=None,
            norm_op_kwargs=None,
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.ReLU,
            nonlin_kwargs={},
            deep_supervision=False,
            decoder_discretization="vmunet_scan",
        )
    except ValueError as exc:
        assert "2D" in str(exc)
        return
    raise AssertionError("Expected vmunet_scan to reject 3D models")
