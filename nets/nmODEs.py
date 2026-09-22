import math
import os
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd,_ConvTransposeNd
from typing import Type, Optional
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba_selective_scan_fn
except Exception:
    mamba_selective_scan_fn = None

up_conv = 1

def do_nothing(data):
    return data

def convert_conv_op_to_dim(conv_op: Type[_ConvNd]) -> int:
    """
    :param conv_op: conv class
    :return: dimension: 1, 2 or 3
    """
    if conv_op == nn.Conv1d:
        return 1
    elif conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    else:
        raise ValueError("Unknown dimension. Only 1d 2d and 3d conv are supported. got %s" % str(conv_op))

def get_matching_upsample(conv_op: Type[_ConvNd] = None, dimension: int = None) -> Type[_ConvTransposeNd]:
    """
    You MUST set EITHER conv_op OR dimension. Do not set both!

    :param conv_op:
    :param dimension:
    :return:
    """
    assert not ((conv_op is not None) and (dimension is not None)), \
        "You MUST set EITHER conv_op OR dimension. Do not set both!"
    if conv_op is not None:
        dimension = convert_conv_op_to_dim(conv_op)
    assert dimension in [1, 2, 3], 'Dimension must be 1, 2 or 3'
    if dimension == 1:
        return upsample1d
    elif dimension == 2:
        return upsample2d
    elif dimension == 3:
        return upsample3d

class upsample1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,scale_factor,bias=False):
        super(upsample1d, self).__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        if up_conv:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        return x
    
class upsample2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,scale_factor,bias=False):
        super(upsample2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        if up_conv:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x
    
class upsample3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,scale_factor,bias=False):
        super(upsample3d, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2,bias=bias)
        self.scale_factor = scale_factor

    def forward(self, x):
        if up_conv:
            x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return x
    
class add_op(torch.nn.Module):
    def __init__(self,op):
        super(add_op,self).__init__()
        self.op = op
    def forward(self,x,y):
        # pdb.set_trace()
        y = x + y
        y = self.op(y)
        return y

class single_decoder(torch.nn.Module):
    def __init__(self,num_stage,current_stage,opx,opx_upper_layer,opy,opxy):
        super(single_decoder,self).__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.opx_upper_layer = opx_upper_layer
        self.opy = opy
        self.opxy = opxy
        self.step = 1/self.num_stage

    def ODE_eq(self,x,y):
        return -self.opy(y)+self.opxy(self.opx(x),self.opy(y))

    def ODE_eq_pre(self,x,y):
        return -self.opy(y)+self.opxy(self.opx_upper_layer(x),self.opy(y))

    def step1_order1_explicit(self,x1,y1):
        f1 = self.ODE_eq(x1,y1)
        y2 = y1 + self.step*f1
        return y2,f1

    def step1_order2_implicit(self,x1,y1,x2):
        y2_pre,f1 = self.step1_order1_explicit(x1,y1)
        f2_pre = self.ODE_eq_pre(x2,y2_pre)
        y2 = y1 + (self.step/2)*(f1 + f2_pre)
        return y2,f1

    def steps2_order2_explicit(self,f1,x2,y2):
        f2 = self.ODE_eq(x2,y2)
        y3 = y2 + (self.step/2)*(3*f2 - f1)
        return y3,f2

    def steps2_order3_implicit(self,f1,x2,y2,x3):
        y3_pre,f2 = self.steps2_order2_explicit(f1,x2,y2)
        f3_pre = self.ODE_eq_pre(x3,y3_pre)
        y3 = y2 + (self.step/12)*(5*f3_pre + 8*f2 - f1)
        return y3,f2

    def steps3_order3_explicit(self,f1,f2,x3,y3):
        f3 = self.ODE_eq(x3,y3)
        y4 = y3 + (self.step/12)*(23*f3-16*f2+5*f1)
        return y4,f3

    def steps3_order4_implicit(self,f1,f2,x3,y3,x4):
        y4_pre,f3 = self.steps3_order3_explicit(f1,f2,x3,y3)
        f4_pre = self.ODE_eq_pre(x4,y4_pre)
        y4 = y3 + (self.step/24)*(9*f4_pre + 19*f3 - 5*f2 +f1)
        return y4,f3

    def steps4_order4_explicit(self,f1,f2,f3,x4,y4):
        f4 = self.ODE_eq(x4,y4)
        y5 = y4 + (self.step/24)*(55*f4 - 59*f3 + 37*f2 - 9*f1)
        return y5,f4

    def steps4_order4_implicit(self,f1,f2,f3,x4,y4,x5):
        y5_pre,f4 = self.steps4_order4_explicit(f1,f2,f3,x4,y4)
        f5_pre = self.ODE_eq_pre(x5,y5_pre)
        y5 = y4 + (self.step/24)*(9*f5_pre + 19*f4 - 5*f3 +f2)
        return y5,f4
    
    def forward(self,f,x,y):
        if self.current_stage == self.num_stage:
            if self.current_stage == 2:
                return self.steps2_order2_explicit(f[0],x[-self.current_stage],y)
            elif self.current_stage == 3:
                return self.steps3_order3_explicit(f[0],f[1],x[-self.current_stage],y)
            else:
                return self.steps4_order4_explicit(f[0],f[1],f[2],x[-self.current_stage],y)
        else:
            if self.current_stage == 1:
                return self.step1_order2_implicit(x[-self.current_stage],y,x[-self.current_stage-1])
            elif self.current_stage == 2:
                return self.steps2_order3_implicit(f[0],x[-self.current_stage],y,x[-self.current_stage-1])
            elif self.current_stage == 3:
                return self.steps3_order4_implicit(f[0],f[1],x[-self.current_stage],y,x[-self.current_stage-1])
            else:
                return self.steps4_order4_implicit(f[-self.current_stage+4],f[-self.current_stage+3],f[-self.current_stage+2],
                                                   x[-self.current_stage],y,x[-self.current_stage-1])

    # def forward(self,f,x,y):  #1
    #     return self.step1_order1_explicit(x[-self.current_stage],y)
    
    # def forward(self,f,x,y):  #2
    #     if self.current_stage == self.num_stage:
    #         return self.steps2_order2_explicit(f[-1],x[-self.current_stage],y)
    #     else:
    #         return self.step1_order2_implicit(x[-self.current_stage],y,x[-self.current_stage-1])

    # def forward(self,f,x,y): #3
    #     if self.current_stage == self.num_stage:
    #         if self.current_stage == 2:
    #             return self.steps2_order2_explicit(f[-1],x[-self.current_stage],y)
    #         else:
    #             return self.steps3_order3_explicit(f[-2],f[-1],x[-self.current_stage],y)
            
    #     else:
    #         if self.current_stage == 1:
    #             return self.step1_order2_implicit(x[-self.current_stage],y,x[-self.current_stage-1])
    #         else:
    #             return self.steps2_order3_implicit(f[-1],x[-self.current_stage],y,x[-self.current_stage-1])


def get_hippo_diagonal(size: int, scale: float = 1.0) -> torch.Tensor:
    """
    Returns the diagonal approximation of the HiPPO-LegS matrix as used in S4D.
    Values are strictly negative and spaced to capture different decay speeds.
    """
    idx = torch.arange(size, dtype=torch.float32)
    diag = -(idx + 0.5) * scale
    return diag


def _vmunet_cross_scan_2d(x: torch.Tensor) -> torch.Tensor:
    """
    VM-UNet scan order:
    1. row-major
    2. column-major
    3. reversed row-major
    4. reversed column-major
    """
    b, c, _, _ = x.shape
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
    ).view(b, 4, c, -1)


def _vmunet_cross_merge_2d(y_dir: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Inverse of the VM-UNet 4-direction scan.
    """
    b, _, c, l = y_dir.shape
    inv_y = torch.flip(y_dir[:, 2:4], dims=[-1]).reshape(b, 2, c, l)
    col_y = y_dir[:, 1].reshape(b, c, w, h).transpose(2, 3).contiguous().reshape(b, c, l)
    inv_col_y = inv_y[:, 1].reshape(b, c, w, h).transpose(2, 3).contiguous().reshape(b, c, l)
    merged = y_dir[:, 0] + inv_y[:, 0] + col_y + inv_col_y
    return merged.reshape(b, c, h, w)


def _selective_scan_recurrence(
    u_seq: torch.Tensor,
    delta: torch.Tensor,
    B_param: torch.Tensor,
    C_param: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Memory-efficient selective scan recurrence.

    u_seq:   (B, C, L)
    delta:   (B, L, C)
    B_param: (B, L, N)
    C_param: (B, L, N)
    A:       (C, N)
    D:       (C,)

    Returns:
        y_seq: (B, C, L)
    """
    batch_size, channels, seq_len = u_seq.shape
    d_state = A.shape[1]
    state = u_seq.new_zeros((batch_size, channels, d_state))
    u_steps = u_seq.transpose(1, 2)
    outputs = []

    A = A.to(dtype=u_seq.dtype, device=u_seq.device)
    D = D.to(dtype=u_seq.dtype, device=u_seq.device)

    for t in range(seq_len):
        delta_t = delta[:, t, :]
        B_t = B_param[:, t, :]
        C_t = C_param[:, t, :]
        u_t = u_steps[:, t, :]

        deltaA_t = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))
        deltaB_u_t = delta_t.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
        state = deltaA_t * state + deltaB_u_t
        y_t = torch.einsum('b d n, b n -> b d', state, C_t) + D.unsqueeze(0) * u_t
        outputs.append(y_t.unsqueeze(-1))

    return torch.cat(outputs, dim=-1)


def _run_selective_scan(
    u_seq: torch.Tensor,
    delta: torch.Tensor,
    B_param: torch.Tensor,
    C_param: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    use_checkpointing: bool,
) -> torch.Tensor:
    if mamba_selective_scan_fn is not None and u_seq.is_cuda:
        delta_kernel = delta.transpose(1, 2).contiguous()
        B_kernel = B_param.transpose(1, 2).contiguous()
        C_kernel = C_param.transpose(1, 2).contiguous()
        return mamba_selective_scan_fn(
            u_seq.contiguous(),
            delta_kernel,
            A.to(torch.float32),
            B_kernel,
            C_kernel,
            D.to(torch.float32),
            z=None,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=False,
        )
    if use_checkpointing and torch.is_grad_enabled():
        return checkpoint(
            _selective_scan_recurrence,
            u_seq,
            delta,
            B_param,
            C_param,
            A,
            D,
            use_reentrant=False,
        )
    return _selective_scan_recurrence(u_seq, delta, B_param, C_param, A, D)


class MambaSelectiveDecoderCell(nn.Module):
    """
    Mamba-like selective SSM cell with ZOH discretization.
    Treats spatial dims as sequence, applies input-dependent Δ/B/C and output gate.
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 hidden_channels: int,
                 d_state: int = 16,
                 dt_rank: int = None,
                 conv_kernel: int = 3,
                 activation: Optional[nn.Module] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.step = 1 / max(1, num_stage)
        self.opx = opx
        self.hidden_channels = hidden_channels
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, hidden_channels // 16)
        self.activation = activation if activation is not None else nn.Identity()
        self.use_scan_checkpoint = True

        d_inner = hidden_channels
        # 输入投影，拆成 x 与 gate 分支
        self.in_proj = nn.Conv1d(hidden_channels, d_inner * 2, kernel_size=1, bias=False)
        # depthwise 卷积捕获局部上下文
        self.dw_conv = nn.Conv1d(d_inner, d_inner, kernel_size=conv_kernel, padding=conv_kernel - 1,
                                 groups=d_inner, bias=False)
        # 生成 Δ/B/C
        self.x_proj = nn.Conv1d(d_inner, self.dt_rank + d_state * 2, kernel_size=1, bias=False)
        self.dt_proj = nn.Conv1d(self.dt_rank, d_inner, kernel_size=1, bias=True)

        # A 对角，初始化为 1..d_state 归一化后取负对数
        A = torch.arange(1, d_state + 1, dtype=torch.float32) / max(1, d_state)
        A = repeat(A, 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        # 额外输出投影回 hidden_channels
        self.out_proj = nn.Conv1d(d_inner, hidden_channels, kernel_size=1, bias=False)

    def forward(self, f, x, y):
        # x: list of skips, y: latent state (unused directly here)
        u = self.opx(x[-self.current_stage])  # (B,C, H,W[,D]) -> upsampled
        b, c = u.shape[:2]
        spatial_shape = u.shape[2:]
        seq_len = int(torch.prod(torch.tensor(spatial_shape)))
        # flatten to (B, C, L)
        u_flat = u.reshape(b, c, seq_len)

        # 输入投影 + gate
        x_and_res = self.in_proj(u_flat)  # (B, 2*d_inner, L)
        x_proj, res = torch.split(x_and_res, self.hidden_channels, dim=1)
        # depthwise + SiLU
        x_proj = self.dw_conv(x_proj)[..., :seq_len]
        x_proj = F.silu(x_proj)

        # 生成 Δ, B, C
        x_dbl = self.x_proj(x_proj)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        delta = F.softplus(self.dt_proj(delta))  # (B, d_inner, L)

        A = -torch.exp(self.A_log.to(dtype=u.dtype, device=u.device))  # (d_inner, d_state)
        D = self.D.to(dtype=u.dtype, device=u.device)

        # 转置为 (B, L, *)
        delta = delta.transpose(1, 2)  # (B, L, d_inner)
        B = B.transpose(1, 2)  # (B, L, d_state)
        C = C.transpose(1, 2)  # (B, L, d_state)

        # 流式递推，避免构造 (B, L, C, N) 的大中间张量
        y_seq = _run_selective_scan(u_flat, delta, B, C, A, D, self.use_scan_checkpoint)

        # 输出门
        y_seq = y_seq * F.silu(res[..., :seq_len])
        y_seq = self.out_proj(y_seq)
        # reshape back
        y_out = y_seq.reshape(b, self.hidden_channels, *spatial_shape)
        y_out = self.activation(y_out)
        return y_out, None


def _vmunet_selective_scan_recurrence(
    xs: torch.Tensor,
    dts: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Fallback selective scan for VM-UNet style grouped directions.

    xs:   (B, K*C, L)
    dts:  (B, K*C, L)
    As:   (K*C, N)
    Bs:   (B, K, N, L)
    Cs:   (B, K, N, L)
    Ds:   (K*C,)
    """
    input_dtype = xs.dtype
    xs = xs.to(torch.float32)
    dts = dts.to(torch.float32)
    Bs = Bs.to(torch.float32)
    Cs = Cs.to(torch.float32)

    batch_size, kc, seq_len = xs.shape
    groups = Bs.shape[1]
    d_state = As.shape[1]
    channels = kc // groups

    xs = xs.view(batch_size, groups, channels, seq_len)
    dts = dts.view(batch_size, groups, channels, seq_len)
    As = As.view(groups, channels, d_state).to(dtype=torch.float32, device=xs.device)
    Ds = Ds.view(groups, channels).to(dtype=torch.float32, device=xs.device)

    if delta_bias is not None:
        dts = dts + delta_bias.view(1, groups, channels, 1).to(dtype=torch.float32, device=xs.device)
    if delta_softplus:
        dts = F.softplus(dts)

    state = xs.new_zeros((batch_size, groups, channels, d_state))
    outputs = []

    for t in range(seq_len):
        dt_t = dts[:, :, :, t]
        B_t = Bs[:, :, :, t]
        C_t = Cs[:, :, :, t]
        x_t = xs[:, :, :, t]

        deltaA_t = torch.exp(dt_t.unsqueeze(-1) * As.unsqueeze(0))
        deltaB_u_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(2) * x_t.unsqueeze(-1)
        state = deltaA_t * state + deltaB_u_t
        y_t = torch.einsum("b k c n, b k n -> b k c", state, C_t) + Ds.unsqueeze(0) * x_t
        outputs.append(y_t.unsqueeze(-1))

    return torch.cat(outputs, dim=-1).view(batch_size, kc, seq_len).to(input_dtype)


def _raise_if_not_finite(name: str, tensor: torch.Tensor) -> None:
    if bool(torch.isfinite(tensor).all().item()):
        return
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    raise FloatingPointError(
        f"Encountered non-finite values in {name}: nan_count={nan_count}, inf_count={inf_count}, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}."
    )


def _run_vmunet_selective_scan(
    xs: torch.Tensor,
    dts: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    Cs: torch.Tensor,
    Ds: torch.Tensor,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
    use_checkpointing: bool = True,
) -> torch.Tensor:
    if mamba_selective_scan_fn is not None and xs.is_cuda:
        return mamba_selective_scan_fn(
            xs.contiguous().to(torch.float32),
            dts.contiguous().to(torch.float32),
            As.to(torch.float32),
            Bs.contiguous().to(torch.float32),
            Cs.contiguous().to(torch.float32),
            Ds.to(torch.float32),
            z=None,
            delta_bias=None if delta_bias is None else delta_bias.to(torch.float32),
            delta_softplus=delta_softplus,
            return_last_state=False,
        )
    if use_checkpointing and torch.is_grad_enabled():
        return checkpoint(
            _vmunet_selective_scan_recurrence,
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias,
            delta_softplus,
            use_reentrant=False,
        )
    return _vmunet_selective_scan_recurrence(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus)


class VMUNetSS2D(nn.Module):
    """
    VM-UNet-style 2D selective scan block adapted for FuseUNet decoder stages.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank is None else dt_rank
        self.k_groups = 4
        self.use_scan_checkpoint = True

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )
        self.act = nn.SiLU()

        x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            for _ in range(self.k_groups)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj], dim=0))

        dt_projs = [
            self._dt_init(self.dt_rank, self.d_inner)
            for _ in range(self.k_groups)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_projs], dim=0))

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=self.k_groups, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=self.k_groups, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _dt_init(dt_rank: int, d_inner: int) -> nn.Linear:
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def _A_log_init(d_state: int, d_inner: int, copies: int = 1, merge: bool = True) -> nn.Parameter:
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        return nn.Parameter(A_log)

    @staticmethod
    def _D_init(d_inner: int, copies: int = 1, merge: bool = True) -> nn.Parameter:
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        return nn.Parameter(D)

    def _forward_core(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        seq_len = height * width

        xs = _vmunet_cross_scan_2d(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
        _raise_if_not_finite("VMUNetSS2D.dts", dts)

        xs = xs.contiguous().to(torch.float32).view(batch_size, -1, seq_len)
        dts = dts.contiguous().to(torch.float32).view(batch_size, -1, seq_len)
        Bs = Bs.contiguous().to(torch.float32).view(batch_size, self.k_groups, -1, seq_len)
        Cs = Cs.contiguous().to(torch.float32).view(batch_size, self.k_groups, -1, seq_len)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        Ds = self.Ds.float().view(-1)
        delta_bias = self.dt_projs_bias.float().view(-1)

        out_y = _run_vmunet_selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias=delta_bias,
            delta_softplus=True,
            use_checkpointing=self.use_scan_checkpoint,
        ).view(batch_size, self.k_groups, -1, seq_len)
        _raise_if_not_finite("VMUNetSS2D.scan_output", out_y)

        merged = _vmunet_cross_merge_2d(out_y, height, width)
        _raise_if_not_finite("VMUNetSS2D.merged_output", merged)
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        _, height, width, _ = x.shape
        _raise_if_not_finite("VMUNetSS2D.input", x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.to(torch.float32)
            xz = self.in_proj(x_fp32)
            x_proj, z = xz.chunk(2, dim=-1)

            x_proj = x_proj.permute(0, 3, 1, 2).contiguous()
            x_proj = self.act(self.conv2d(x_proj))
            _raise_if_not_finite("VMUNetSS2D.conv_output", x_proj)

            y = self._forward_core(x_proj)
            y = y.permute(0, 2, 3, 1).contiguous().view(x.shape[0], height, width, -1)
            y = self.out_norm(y)
            y = y * F.silu(z)
            y = self.out_proj(y)
            _raise_if_not_finite("VMUNetSS2D.output", y)

        return y.to(input_dtype)


class VMUNetScanDecoderCell(nn.Module):
    """
    FuseUNet decoder cell using the 2D VM-UNet scan formulation.
    """

    def __init__(
        self,
        num_stage: int,
        current_stage: int,
        opx: nn.Module,
        conv_op: Type[_ConvNd],
        hidden_channels: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        conv_kernel: int = 3,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if conv_op != nn.Conv2d:
            raise ValueError("vmunet_scan only supports 2D models (nn.Conv2d).")

        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.hidden_channels = hidden_channels
        self.step = 1 / max(1, num_stage)
        self.activation = activation if activation is not None else nn.Identity()
        self.scan_patch_stride = 4
        self.scan_upsample_mode = "bilinear"

        self.state_fuse = conv_op(hidden_channels * 2, hidden_channels, 1, 1, 0, bias=False)
        self.pre_scan_norm = nn.GroupNorm(1, hidden_channels)
        self.post_scan_norm = nn.GroupNorm(1, hidden_channels)
        self.scan_patch_embed = conv_op(
            hidden_channels,
            hidden_channels,
            kernel_size=self.scan_patch_stride,
            stride=self.scan_patch_stride,
            padding=0,
            bias=False,
        )
        self.ss2d = VMUNetSS2D(
            d_model=hidden_channels,
            d_state=d_state,
            d_conv=conv_kernel,
            expand=2,
            dt_rank=dt_rank,
        )

    def _patchify_for_scan(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.shape[-2] < self.scan_patch_stride or fused.shape[-1] < self.scan_patch_stride:
            return fused
        return self.scan_patch_embed(fused)

    def _restore_scan_output(self, scanned: torch.Tensor, target_hw) -> torch.Tensor:
        if scanned.shape[-2:] == target_hw:
            return scanned
        return F.interpolate(
            scanned,
            size=target_hw,
            mode=self.scan_upsample_mode,
            align_corners=False,
        )

    def forward(self, f, x, y):
        del f
        with torch.autocast(device_type=y.device.type, enabled=False):
            skip = x[-self.current_stage].to(torch.float32)
            y_state = y.to(torch.float32)
            u = self.opx(skip)
            fused = self.state_fuse(torch.cat([u, y_state], dim=1))
            _raise_if_not_finite("VMUNetScanDecoderCell.fused", fused)
            fused_scan = self.pre_scan_norm(fused)
            _raise_if_not_finite("VMUNetScanDecoderCell.fused_scan", fused_scan)

            fused_low = self._patchify_for_scan(fused_scan)
            _raise_if_not_finite("VMUNetScanDecoderCell.fused_low", fused_low)

            ss2d_out = self.ss2d(fused_low.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            _raise_if_not_finite("VMUNetScanDecoderCell.ss2d_low", ss2d_out)

            ss2d_out = self._restore_scan_output(ss2d_out, fused.shape[-2:])
            ss2d_out = self.post_scan_norm(ss2d_out)
            _raise_if_not_finite("VMUNetScanDecoderCell.ss2d_restored", ss2d_out)

            y_out = self.activation(fused + self.step * ss2d_out)
            _raise_if_not_finite("VMUNetScanDecoderCell.output", y_out)
        return y_out, None

    def compute_conv_feature_map_size(self, input_size):
        spatial = int(torch.prod(torch.tensor(input_size, dtype=torch.int64)).item())
        return spatial * self.hidden_channels


class NMODEEulerDecoderCell(nn.Module):
    """Explicit Euler for the original nmODE: h' = -h + sin(h + v)^2.

    The parent decoder supplies a scalar step from its scale allocation. The
    aligned external input is held constant during this one Euler step; no
    derivative history, learned decay, or output activation is used.
    """

    def __init__(self, num_stage: int, current_stage: int, opx: nn.Module):
        super().__init__()
        if num_stage < 1 or not 1 <= current_stage <= num_stage:
            raise ValueError("Expected 1 <= current_stage <= num_stage.")
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.step = 1.0 / num_stage

    def forward(self, f, x, y, step=None):
        del f
        # Projection follows the surrounding AMP policy; integration stays FP32.
        v = self.opx(x[-self.current_stage])
        with torch.autocast(device_type=y.device.type, enabled=False):
            state = y.float()
            delta = torch.as_tensor(
                self.step if step is None else step,
                dtype=torch.float32, device=y.device,
            )
            if delta.numel() != 1:
                raise ValueError("An nmODE stage requires a scalar step.")
            drive = torch.sin(state + v.float()).square()
            state = (1.0 - delta) * state + delta * drive
        return state, None


class NMODERosenbrockDecoderCell(nn.Module):
    """One-stage Rosenbrock (gamma=1/2) for h' = -h + sin(h + v)^2.

    Holding v fixed within a stage gives the diagonal Jacobian
    J = -1 + sin(2 * (h + v)). Thus (I - step/2 * J) d = step * F
    needs only elementwise division. Gradients flow through both F and J.
    For decoder-supplied steps in [0, 1], the update preserves h in [0, 1].
    """

    def __init__(self, num_stage: int, current_stage: int, opx: nn.Module):
        super().__init__()
        if num_stage < 1 or not 1 <= current_stage <= num_stage:
            raise ValueError("Expected 1 <= current_stage <= num_stage.")
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.step = 1.0 / num_stage

    def forward(self, f, x, y, step=None):
        del f
        # Evaluate the aligned external input once, following the AMP policy.
        v = self.opx(x[-self.current_stage])
        with torch.autocast(device_type=y.device.type, enabled=False):
            state = y.float()
            delta = torch.as_tensor(
                self.step if step is None else step,
                dtype=torch.float32, device=y.device,
            )
            if delta.numel() != 1:
                raise ValueError("An nmODE stage requires a scalar step.")
            z = state + v.float()
            drive = torch.sin(z).square()
            # The denominator is >= 1 for nonnegative steps: no epsilon or clamp.
            denominator = 1.0 + 0.5 * delta * (1.0 - torch.sin(2.0 * z))
            beta = delta / denominator
            state = (1.0 - beta) * state + beta * drive
        return state, None


class NMZOHDecoderCell(nn.Module):
    """
    ZOH discretization of the original nmODE-style decoder:
        dy/dt = -opy(y) + opxy(opx(x), opy(y))
    Treats the skip as piece-wise constant within the step; diagonal A enforces stability.
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 opy: nn.Module,
                 opxy: nn.Module,
                 hidden_channels: int,
                 activation: Optional[nn.Module] = None,
                 init_logA: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.step = 1 / max(1, num_stage)
        self.opx = opx
        self.opy = opy
        self.opxy = opxy
        self.activation = activation if activation is not None else nn.Identity()

        if init_logA is None:
            self.log_A = nn.Parameter(torch.zeros(hidden_channels))
        else:
            self.log_A = nn.Parameter(init_logA.to(dtype=torch.float32))

    def _expand_diag(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.log_A.numel()] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype, device=y.device)).view(*shape)

    def forward(self, f, x, y):
        # Use previous y when computing the driving term, mirroring the semi-implicit nmODE form
        u = self.opxy(self.opx(x[-self.current_stage]), self.opy(y))
        A = self._expand_diag(y)
        dtA = self.step * A
        Ad = torch.exp(dtA)
        Bd = torch.where(torch.abs(A) > 1e-4,
                         torch.expm1(dtA) / A,
                         torch.full_like(A, self.step, dtype=y.dtype))
        y = Ad * y + Bd * u
        y = self.activation(y)
        return y, None


class ZOHDecoderCell(nn.Module):
    """
    Zero-Order Hold discretization cell inspired by state-space models.
    Treats each skip as piece-wise constant input over a time step and
    updates the latent state analytically using exp(A * dt).
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 activation: Optional[nn.Module] = None,
                 init_logA: Optional[torch.Tensor] = None,
                 simple_B: bool = False):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.step = 1 / max(1, num_stage)
        self.opx = opx
        self.hidden_channels = hidden_channels
        self.activation = activation if activation is not None else nn.Identity()
        env_simple = os.environ.get("FUSEUNET_ZOH_SIMPLE_B", "false").lower() == "true"
        self.simple_B = simple_B or env_simple

        # learnable continuous-time diagonal A, optional custom init (e.g., HiPPO diag)
        if init_logA is None:
            a_init = torch.arange(1, hidden_channels + 1, dtype=torch.float32) / max(1, hidden_channels)
            self.log_A = nn.Parameter(torch.log(a_init))
        else:
            self.log_A = nn.Parameter(init_logA.to(dtype=torch.float32))

        # simple input projection (no gating)
        self.input_proj = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        # optional B scale for simplified Bd_u
        if self.simple_B:
            self.b_scale_proj = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        else:
            self.b_scale_proj = None

    def _expand_diag(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype)).view(*shape)

    def forward(self, f, x, y):
        # piece-wise constant input over the current interval
        u_raw = self.opx(x[-self.current_stage])
        u = self.input_proj(u_raw)
        A = self._expand_diag(y)
        dtA = self.step * A
        A_discrete = torch.exp(dtA)

        if self.simple_B:
            b_scale = torch.tanh(self.b_scale_proj(u_raw)) if self.b_scale_proj is not None else 1.0
            y = A_discrete * y + self.step * b_scale * u
        else:
            stable_mask = torch.abs(A) > 1e-4
            denom = torch.where(stable_mask, A, torch.ones_like(A))
            Bd = torch.expm1(dtA) / denom
            Bd = torch.where(stable_mask, Bd, torch.full_like(A, self.step, dtype=y.dtype))
            y = A_discrete * y + Bd * u
        y = self.activation(y)
        return y, None


class SelectiveZOHDecoderCell(nn.Module):
    """
    ZOH cell with input-dependent Δ and B (selective), no extra gate.
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 activation: Optional[nn.Module] = None,
                 init_logA: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.hidden_channels = hidden_channels
        self.activation = activation if activation is not None else nn.Identity()

        # A init mamba-minimal style
        if init_logA is None:
            a_init = torch.arange(1, hidden_channels + 1, dtype=torch.float32) / max(1, hidden_channels)
            self.log_A = nn.Parameter(torch.log(a_init))
        else:
            self.log_A = nn.Parameter(init_logA.to(dtype=torch.float32))

        # project to delta and B scale
        self.proj = conv_op(hidden_channels, hidden_channels * 2, 1, 1, 0, bias=True)

    def _expand_diag(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype, device=y.device)).view(*shape)

    def forward(self, f, x, y):
        u_raw = self.opx(x[-self.current_stage])
        delta_raw, b_raw = torch.split(self.proj(u_raw), self.hidden_channels, dim=1)
        # positive step, avoid zero
        delta = F.softplus(delta_raw) + 1e-4
        b_scale = torch.tanh(b_raw)  # keep B stable

        A = self._expand_diag(y)
        dtA = delta * A
        Ad = torch.exp(dtA)

        stable_mask = torch.abs(A) > 1e-4
        denom = torch.where(stable_mask, A, torch.ones_like(A))
        Bd = torch.expm1(dtA) / denom
        Bd = torch.where(stable_mask, Bd, delta)
        Bd = Bd * b_scale

        # apply input as driving term
        y = Ad * y + Bd * u_raw
        y = self.activation(y)
        return y, None


class FuseSSMZOHDecoderCell(nn.Module):
    """
    FuseSSM decoder cell with dual gating and ZOH-style state update.

    Continuous dynamics:
        dh/dt = A(r ⊙ h) + (W_b u) ⊙ z

    Exact ZOH update:
        h_t = exp(step * (A ⊙ r)) ⊙ h_{t-1} + beta ⊙ ((W_b u) ⊙ z)
        beta = (exp(step * (A ⊙ r)) - 1) / (A ⊙ r), with beta = step when |A ⊙ r| is small

    Optional approximate update:
        h_t ≈ exp(step * (A ⊙ r)) ⊙ h_{t-1} + step * ((W_b u) ⊙ z)
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 activation: Optional[nn.Module] = None,
                 approximate: bool = False,
                 eps: float = 1e-4):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.hidden_channels = hidden_channels
        self.step = 1 / max(1, num_stage)
        self.activation = activation if activation is not None else nn.Identity()
        self.approximate = approximate
        self.eps = eps

        a_init = torch.arange(1, hidden_channels + 1, dtype=torch.float32) / max(1, hidden_channels)
        self.log_A = nn.Parameter(torch.log(a_init))

        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.Wr_u = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.Wr_h = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.Wz_u = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.Wz_h = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)

    def _expand_A(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype, device=y.device)).view(*shape)

    def forward(self, f, x, y):
        del f  # unused, kept for decoder interface compatibility
        u = self.opx(x[-self.current_stage])
        r = torch.sigmoid(self.Wr_u(u) + self.Wr_h(y))
        z = torch.sigmoid(self.Wz_u(u) + self.Wz_h(y))

        A = self._expand_A(y)
        A_eff = A * r
        input_drive = self.B_base(u) * z

        alpha = torch.exp(self.step * A_eff)
        if self.approximate:
            beta = torch.full_like(A_eff, self.step, dtype=y.dtype, device=y.device)
        else:
            stable_mask = torch.abs(A_eff) > self.eps
            denom = torch.where(stable_mask, A_eff, torch.ones_like(A_eff))
            beta = torch.expm1(self.step * A_eff) / denom
            beta = torch.where(stable_mask, beta, torch.full_like(A_eff, self.step, dtype=y.dtype))

        y = alpha * y + beta * input_drive
        y = self.activation(y)
        return y, None


def _global_avg_keepdim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim <= 2:
        return x
    return x.mean(dim=tuple(range(2, x.ndim)), keepdim=True)


def _largest_divisor_at_most(value: int, upper_bound: int) -> int:
    for candidate in range(min(value, upper_bound), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class BaseSSMGate(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.hidden_channels = hidden_channels

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        raise NotImplementedError


class TanhInputGate(BaseSSMGate):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int):
        super().__init__(hidden_channels)
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        Ay = A * y
        Bu = self.B_base(ux) * torch.tanh(self.B_gate(ux))
        return Ay, Bu


class StateSigmoidGate(BaseSSMGate):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int):
        super().__init__(hidden_channels)
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate_y = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        Ay = A * y
        Bu = self.B_base(ux) * torch.sigmoid(self.B_gate_y(y))
        return Ay, Bu


class JointSigmoidGateBase(BaseSSMGate):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int, gate_y_bias: float = 0.0):
        super().__init__(hidden_channels)
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate_u = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate_y = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        if self.B_gate_y.bias is not None:
            nn.init.constant_(self.B_gate_y.bias, gate_y_bias)

    def _gate(self, ux: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.B_gate_u(ux) + self.B_gate_y(y))

    def _drive(self, ux: torch.Tensor) -> torch.Tensor:
        return self.B_base(ux)


class JointSigmoidGate(JointSigmoidGateBase):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int):
        super().__init__(conv_op, hidden_channels)

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        Ay = A * y
        z = self._gate(ux, y)
        Bu = self._drive(ux) * z
        return Ay, Bu


class JointSigmoidOpenBiasGate(JointSigmoidGateBase):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int):
        super().__init__(conv_op, hidden_channels, gate_y_bias=2.0)

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        Ay = A * y
        z = self._gate(ux, y)
        Bu = self._drive(ux) * z
        return Ay, Bu


class JointSigmoidFloorGate(JointSigmoidGateBase):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int, floor: float = 0.3):
        super().__init__(conv_op, hidden_channels)
        self.floor = floor

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        Ay = A * y
        z = self._gate(ux, y)
        Bu = self._drive(ux) * (self.floor + (1.0 - self.floor) * z)
        return Ay, Bu


class JointSigmoidAmplifiedGate(JointSigmoidGateBase):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int, scale: float = 2.0):
        super().__init__(conv_op, hidden_channels)
        self.scale = scale

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        Ay = A * y
        z = self._gate(ux, y)
        Bu = self._drive(ux) * (self.scale * z)
        return Ay, Bu


class GRULikeDualGate(BaseSSMGate):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int):
        super().__init__(hidden_channels)
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.reset_gate_u = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.reset_gate_y = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.update_gate_u = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.update_gate_y = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        r = torch.sigmoid(self.reset_gate_u(ux) + self.reset_gate_y(y))
        z = torch.sigmoid(self.update_gate_u(ux) + self.update_gate_y(y))
        Ay = A * (r * y)
        Bu = self.B_base(ux) * z
        return Ay, Bu


class GroupWiseJointGAPGate(BaseSSMGate):
    def __init__(self, conv_op: Type[_ConvNd], hidden_channels: int, max_groups: int = 8):
        super().__init__(hidden_channels)
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.num_groups = _largest_divisor_at_most(hidden_channels, max_groups)
        self.channels_per_group = hidden_channels // self.num_groups
        self.group_gate = conv_op(2 * hidden_channels, self.num_groups, 1, 1, 0, bias=True)

    def _expand_group_gate(self, group_gate: torch.Tensor) -> torch.Tensor:
        return group_gate.repeat_interleave(self.channels_per_group, dim=1)

    def forward(self, ux: torch.Tensor, y: torch.Tensor, A: torch.Tensor):
        pooled_ux = _global_avg_keepdim(ux)
        pooled_y = _global_avg_keepdim(y)
        z_group = torch.sigmoid(self.group_gate(torch.cat([pooled_ux, pooled_y], dim=1)))
        z = self._expand_group_gate(z_group)
        Ay = A * y
        Bu = self.B_base(ux) * z
        return Ay, Bu


SSM_GATE_REGISTRY = {
    "tanh_input": TanhInputGate,
    "state_sigmoid": StateSigmoidGate,
    "state_only_sigmoid": StateSigmoidGate,
    "joint_sigmoid": JointSigmoidGate,
    "joint_sigmoid_openbias": JointSigmoidOpenBiasGate,
    "joint_sigmoid_floor": JointSigmoidFloorGate,
    "joint_sigmoid_amplified": JointSigmoidAmplifiedGate,
    "gru_dual": GRULikeDualGate,
    "groupwise_joint_gap": GroupWiseJointGAPGate,
}


def build_ssm_gate(gate_variant: str, conv_op: Type[_ConvNd], hidden_channels: int) -> BaseSSMGate:
    gate_key = gate_variant.lower()
    if gate_key not in SSM_GATE_REGISTRY:
        available = ", ".join(sorted(SSM_GATE_REGISTRY.keys()))
        raise ValueError(f"Unsupported SSM gate variant '{gate_variant}'. Available: {available}")
    return SSM_GATE_REGISTRY[gate_key](conv_op, hidden_channels)


class LinearSSMMultiStepDecoderCell(nn.Module):
    """
    Linear SSM with stable diagonal A and pluggable input gating.
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 opx_upper_layer: Optional[nn.Module],
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 activation: Optional[nn.Module] = None,
                 gate_variant: Optional[str] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.opx_upper_layer = opx_upper_layer
        self.hidden_channels = hidden_channels
        self.step = 1 / max(1, num_stage)
        self.gate_variant = (gate_variant or os.environ.get("FUSEUNET_SSM_GATE_MODE", "joint_sigmoid")).lower()
        self.activation = activation if activation is not None else nn.Identity()

        a_init = torch.arange(1, hidden_channels + 1, dtype=torch.float32) / max(1, hidden_channels)
        self.log_A = nn.Parameter(torch.log(a_init))
        self.gate_module = build_ssm_gate(self.gate_variant, conv_op, hidden_channels)

    def _expand_A(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype, device=y.device)).view(*shape)

    def ODE_eq(self, x, y):
        A = self._expand_A(y)
        ux = self.opx(x)
        Ay, Bu = self.gate_module(ux, y, A)
        return Ay + Bu

    def ODE_eq_pre(self, x, y):
        if self.opx_upper_layer is not None:
            ux = self.opx_upper_layer(x)
        else:
            ux = self.opx(x)
        A = self._expand_A(y)
        Ay, Bu = self.gate_module(ux, y, A)
        return Ay + Bu

    def step1_order1_explicit(self, x1, y1):
        f1 = self.ODE_eq(x1, y1)
        y2 = y1 + self.step * f1
        return y2, f1

    def step1_order2_implicit(self, x1, y1, x2):
        y2_pre, f1 = self.step1_order1_explicit(x1, y1)
        f2_pre = self.ODE_eq_pre(x2, y2_pre)
        y2 = y1 + (self.step / 2) * (f1 + f2_pre)
        return y2, f1

    def steps2_order2_explicit(self, f1, x2, y2):
        f2 = self.ODE_eq(x2, y2)
        y3 = y2 + (self.step / 2) * (3 * f2 - f1)
        return y3, f2

    def steps2_order3_implicit(self, f1, x2, y2, x3):
        y3_pre, f2 = self.steps2_order2_explicit(f1, x2, y2)
        f3_pre = self.ODE_eq_pre(x3, y3_pre)
        y3 = y2 + (self.step / 12) * (5 * f3_pre + 8 * f2 - f1)
        return y3, f2

    def steps3_order3_explicit(self, f1, f2, x3, y3):
        f3 = self.ODE_eq(x3, y3)
        y4 = y3 + (self.step / 12) * (23 * f3 - 16 * f2 + 5 * f1)
        return y4, f3

    def steps3_order4_implicit(self, f1, f2, x3, y3, x4):
        y4_pre, f3 = self.steps3_order3_explicit(f1, f2, x3, y3)
        f4_pre = self.ODE_eq_pre(x4, y4_pre)
        y4 = y3 + (self.step / 24) * (9 * f4_pre + 19 * f3 - 5 * f2 + f1)
        return y4, f3

    def steps4_order4_explicit(self, f1, f2, f3, x4, y4):
        f4 = self.ODE_eq(x4, y4)
        y5 = y4 + (self.step / 24) * (55 * f4 - 59 * f3 + 37 * f2 - 9 * f1)
        return y5, f4

    def steps4_order4_implicit(self, f1, f2, f3, x4, y4, x5):
        y5_pre, f4 = self.steps4_order4_explicit(f1, f2, f3, x4, y4)
        f5_pre = self.ODE_eq_pre(x5, y5_pre)
        y5 = y4 + (self.step / 24) * (9 * f5_pre + 19 * f4 - 5 * f3 + f2)
        return y5, f4

    def forward(self, f, x, y):
        if self.current_stage == self.num_stage:
            if self.current_stage == 2:
                return self.steps2_order2_explicit(f[0], x[-self.current_stage], y)
            if self.current_stage == 3:
                return self.steps3_order3_explicit(f[0], f[1], x[-self.current_stage], y)
            return self.steps4_order4_explicit(f[0], f[1], f[2], x[-self.current_stage], y)
        if self.current_stage == 1:
            return self.step1_order2_implicit(x[-self.current_stage], y, x[-self.current_stage - 1])
        if self.current_stage == 2:
            return self.steps2_order3_implicit(f[0], x[-self.current_stage], y, x[-self.current_stage - 1])
        if self.current_stage == 3:
            return self.steps3_order4_implicit(f[0], f[1], x[-self.current_stage], y, x[-self.current_stage - 1])
        return self.steps4_order4_implicit(
            f[-self.current_stage + 4],
            f[-self.current_stage + 3],
            f[-self.current_stage + 2],
            x[-self.current_stage],
            y,
            x[-self.current_stage - 1]
        )

class SpatialSSMMultiStepDecoderCell(nn.Module):
    """
    Spatial SSM: dy/dt = A(y) + B(x), integrated with the existing multistep scheme.
    A is a depthwise spatial operator with per-channel decay; B is a 1x1 conv (optional gate).
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 opx_upper_layer: Optional[nn.Module],
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 spatial_kernel: int = 3,
                 use_gate: bool = True,
                 activation: Optional[nn.Module] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.opx_upper_layer = opx_upper_layer
        self.hidden_channels = hidden_channels
        self.step = 1 / max(1, num_stage)
        self.activation = activation if activation is not None else nn.Identity()

        padding = spatial_kernel // 2
        self.A_spatial = conv_op(hidden_channels,
                                 hidden_channels,
                                 spatial_kernel,
                                 1,
                                 padding,
                                 bias=False,
                                 groups=hidden_channels)
        # Initialize with small decay to keep updates stable at start.
        self.A_decay = nn.Parameter(torch.full((hidden_channels,), -2.0))

        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True) if use_gate else None

    def _expand_decay(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return F.softplus(self.A_decay).to(dtype=y.dtype, device=y.device).view(*shape)

    def _apply_A(self, y: torch.Tensor) -> torch.Tensor:
        decay = self._expand_decay(y)
        return self.A_spatial(y) - decay * y

    def _apply_B(self, ux: torch.Tensor) -> torch.Tensor:
        if self.B_gate is None:
            return self.B_base(ux)
        return self.B_base(ux) * torch.tanh(self.B_gate(ux))

    def ODE_eq(self, x, y):
        Ay = self._apply_A(y)
        ux = self.opx(x)
        Bu = self._apply_B(ux)
        return Ay + Bu

    def ODE_eq_pre(self, x, y):
        if self.opx_upper_layer is not None:
            ux = self.opx_upper_layer(x)
        else:
            ux = self.opx(x)
        Bu = self._apply_B(ux)
        Ay = self._apply_A(y)
        return Ay + Bu

    def step1_order1_explicit(self, x1, y1):
        f1 = self.ODE_eq(x1, y1)
        y2 = y1 + self.step * f1
        return y2, f1

    def step1_order2_implicit(self, x1, y1, x2):
        y2_pre, f1 = self.step1_order1_explicit(x1, y1)
        f2_pre = self.ODE_eq_pre(x2, y2_pre)
        y2 = y1 + (self.step / 2) * (f1 + f2_pre)
        return y2, f1

    def steps2_order2_explicit(self, f1, x2, y2):
        f2 = self.ODE_eq(x2, y2)
        y3 = y2 + (self.step / 2) * (3 * f2 - f1)
        return y3, f2

    def steps2_order3_implicit(self, f1, x2, y2, x3):
        y3_pre, f2 = self.steps2_order2_explicit(f1, x2, y2)
        f3_pre = self.ODE_eq_pre(x3, y3_pre)
        y3 = y2 + (self.step / 12) * (5 * f3_pre + 8 * f2 - f1)
        return y3, f2

    def steps3_order3_explicit(self, f1, f2, x3, y3):
        f3 = self.ODE_eq(x3, y3)
        y4 = y3 + (self.step / 12) * (23 * f3 - 16 * f2 + 5 * f1)
        return y4, f3

    def steps3_order4_implicit(self, f1, f2, x3, y3, x4):
        y4_pre, f3 = self.steps3_order3_explicit(f1, f2, x3, y3)
        f4_pre = self.ODE_eq_pre(x4, y4_pre)
        y4 = y3 + (self.step / 24) * (9 * f4_pre + 19 * f3 - 5 * f2 + f1)
        return y4, f3

    def steps4_order4_explicit(self, f1, f2, f3, x4, y4):
        f4 = self.ODE_eq(x4, y4)
        y5 = y4 + (self.step / 24) * (55 * f4 - 59 * f3 + 37 * f2 - 9 * f1)
        return y5, f4

    def steps4_order4_implicit(self, f1, f2, f3, x4, y4, x5):
        y5_pre, f4 = self.steps4_order4_explicit(f1, f2, f3, x4, y4)
        f5_pre = self.ODE_eq_pre(x5, y5_pre)
        y5 = y4 + (self.step / 24) * (9 * f5_pre + 19 * f4 - 5 * f3 + f2)
        return y5, f4

    def forward(self, f, x, y):
        if self.current_stage == self.num_stage:
            if self.current_stage == 2:
                return self.steps2_order2_explicit(f[0], x[-self.current_stage], y)
            elif self.current_stage == 3:
                return self.steps3_order3_explicit(f[0], f[1], x[-self.current_stage], y)
            else:
                return self.steps4_order4_explicit(f[0], f[1], f[2], x[-self.current_stage], y)
        else:
            if self.current_stage == 1:
                return self.step1_order2_implicit(x[-self.current_stage], y, x[-self.current_stage - 1])
            elif self.current_stage == 2:
                return self.steps2_order3_implicit(f[0], x[-self.current_stage], y, x[-self.current_stage - 1])
            elif self.current_stage == 3:
                return self.steps3_order4_implicit(f[0], f[1], x[-self.current_stage], y, x[-self.current_stage - 1])
            else:
                return self.steps4_order4_implicit(f[-self.current_stage + 4],
                                                   f[-self.current_stage + 3],
                                                   f[-self.current_stage + 2],
                                                   x[-self.current_stage],
                                                   y,
                                                   x[-self.current_stage - 1])


class LinearSSMMultiStepDecoderCellDelta(nn.Module):
    """
    Linear SSM with input-adaptive step size (delta) and the same multistep scheme.
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 opx_upper_layer: Optional[nn.Module],
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 activation: Optional[nn.Module] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.opx_upper_layer = opx_upper_layer
        self.hidden_channels = hidden_channels
        self.step = 1 / max(1, num_stage)
        self.activation = activation if activation is not None else nn.Identity()

        a_init = torch.arange(1, hidden_channels + 1, dtype=torch.float32) / max(1, hidden_channels)
        self.log_A = nn.Parameter(torch.log(a_init))
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.dt_proj = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)

    def _expand_A(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype, device=y.device)).view(*shape)

    def _compute_delta(self, u: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.dt_proj(u)) + 1e-4

    def ODE_eq(self, x, y):
        Ay = self._expand_A(y) * y
        ux = self.opx(x)
        Bu = self.B_base(ux) * torch.tanh(self.B_gate(ux))
        delta = self._compute_delta(ux)
        return delta * (Ay + Bu)

    def ODE_eq_pre(self, x, y):
        if self.opx_upper_layer is not None:
            ux = self.opx_upper_layer(x)
        else:
            ux = self.opx(x)
        Bu = self.B_base(ux) * torch.tanh(self.B_gate(ux))
        Ay = self._expand_A(y) * y
        delta = self._compute_delta(ux)
        return delta * (Ay + Bu)

    def step1_order1_explicit(self, x1, y1):
        f1 = self.ODE_eq(x1, y1)
        y2 = y1 + self.step * f1
        return y2, f1

    def step1_order2_implicit(self, x1, y1, x2):
        y2_pre, f1 = self.step1_order1_explicit(x1, y1)
        f2_pre = self.ODE_eq_pre(x2, y2_pre)
        y2 = y1 + (self.step / 2) * (f1 + f2_pre)
        return y2, f1

    def steps2_order2_explicit(self, f1, x2, y2):
        f2 = self.ODE_eq(x2, y2)
        y3 = y2 + (self.step / 2) * (3 * f2 - f1)
        return y3, f2

    def steps2_order3_implicit(self, f1, x2, y2, x3):
        y3_pre, f2 = self.steps2_order2_explicit(f1, x2, y2)
        f3_pre = self.ODE_eq_pre(x3, y3_pre)
        y3 = y2 + (self.step / 12) * (5 * f3_pre + 8 * f2 - f1)
        return y3, f2

    def steps3_order3_explicit(self, f1, f2, x3, y3):
        f3 = self.ODE_eq(x3, y3)
        y4 = y3 + (self.step / 12) * (23 * f3 - 16 * f2 + 5 * f1)
        return y4, f3

    def steps3_order4_implicit(self, f1, f2, x3, y3, x4):
        y4_pre, f3 = self.steps3_order3_explicit(f1, f2, x3, y3)
        f4_pre = self.ODE_eq_pre(x4, y4_pre)
        y4 = y3 + (self.step / 24) * (9 * f4_pre + 19 * f3 - 5 * f2 + f1)
        return y4, f3

    def steps4_order4_explicit(self, f1, f2, f3, x4, y4):
        f4 = self.ODE_eq(x4, y4)
        y5 = y4 + (self.step / 24) * (55 * f4 - 59 * f3 + 37 * f2 - 9 * f1)
        return y5, f4

    def steps4_order4_implicit(self, f1, f2, f3, x4, y4, x5):
        y5_pre, f4 = self.steps4_order4_explicit(f1, f2, f3, x4, y4)
        f5_pre = self.ODE_eq_pre(x5, y5_pre)
        y5 = y4 + (self.step / 24) * (9 * f5_pre + 19 * f4 - 5 * f3 + f2)
        return y5, f4

    def forward(self, f, x, y):
        if self.current_stage == self.num_stage:
            if self.current_stage == 2:
                return self.steps2_order2_explicit(f[0], x[-self.current_stage], y)
            elif self.current_stage == 3:
                return self.steps3_order3_explicit(f[0], f[1], x[-self.current_stage], y)
            else:
                return self.steps4_order4_explicit(f[0], f[1], f[2], x[-self.current_stage], y)
        else:
            if self.current_stage == 1:
                return self.step1_order2_implicit(x[-self.current_stage], y, x[-self.current_stage - 1])
            elif self.current_stage == 2:
                return self.steps2_order3_implicit(f[0], x[-self.current_stage], y, x[-self.current_stage - 1])
            elif self.current_stage == 3:
                return self.steps3_order4_implicit(f[0], f[1], x[-self.current_stage], y, x[-self.current_stage - 1])
            else:
                return self.steps4_order4_implicit(f[-self.current_stage + 4],
                                                   f[-self.current_stage + 3],
                                                   f[-self.current_stage + 2],
                                                   x[-self.current_stage],
                                                   y,
                                                   x[-self.current_stage - 1])


class SelectiveSSMMultiStepDecoderCell(nn.Module):
    """
    Linear multistep decoder with Mamba-style selective parameters.
    - RHS: f = delta * (A * y + B(u)), where delta/B are input依赖
    - A is learnable稳定对角，B 用 1x1 conv + tanh 门
    - 保留 Adams-Bashforth/Moulton 多步调度，接口与其他 decoder cell 兼容
    """
    def __init__(self,
                 num_stage: int,
                 current_stage: int,
                 opx: nn.Module,
                 opx_upper_layer: Optional[nn.Module],
                 conv_op: Type[_ConvNd],
                 hidden_channels: int,
                 dt_rank: Optional[int] = None,
                 conv_kernel: int = 3,
                 activation: Optional[nn.Module] = None):
        super().__init__()
        self.num_stage = num_stage
        self.current_stage = current_stage
        self.opx = opx
        self.opx_upper_layer = opx_upper_layer
        self.hidden_channels = hidden_channels
        self.step = 1 / max(1, num_stage)
        self.dt_rank = dt_rank or max(1, hidden_channels // 16)
        self.activation = activation if activation is not None else nn.Identity()

        # selective parameter projections
        self.in_proj = conv_op(hidden_channels, hidden_channels * 2, 1, 1, 0, bias=False)
        pad = conv_kernel // 2
        self.dw_conv = conv_op(hidden_channels, hidden_channels, conv_kernel, 1, pad,
                               groups=hidden_channels, bias=False)
        self.x_proj = conv_op(hidden_channels, self.dt_rank + hidden_channels, 1, 1, 0, bias=False)
        self.dt_proj = conv_op(self.dt_rank, hidden_channels, 1, 1, 0, bias=True)

        a_init = torch.arange(1, hidden_channels + 1, dtype=torch.float32) / max(1, hidden_channels)
        self.log_A = nn.Parameter(torch.log(a_init))
        self.B_base = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.B_gate = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=True)
        self.D = nn.Parameter(torch.ones(hidden_channels))
        self.out_proj = conv_op(hidden_channels, hidden_channels, 1, 1, 0, bias=False)

    def _expand_A(self, y: torch.Tensor) -> torch.Tensor:
        dims = y.ndim - 2
        shape = [1, self.hidden_channels] + [1] * max(0, dims)
        return (-torch.exp(self.log_A).to(dtype=y.dtype, device=y.device)).view(*shape)

    def _prepare_u(self, x_input: torch.Tensor, use_upper: bool) -> torch.Tensor:
        if use_upper and self.opx_upper_layer is not None:
            return self.opx_upper_layer(x_input)
        return self.opx(x_input)

    def _compute_params(self, u: torch.Tensor):
        x_and_res = self.in_proj(u)
        x_proj, res_gate = torch.split(x_and_res, self.hidden_channels, dim=1)
        x_proj = self.dw_conv(x_proj)
        x_proj = F.silu(x_proj)

        proj = self.x_proj(x_proj)
        delta_raw, b_raw = torch.split(proj, [self.dt_rank, self.hidden_channels], dim=1)
        delta = F.softplus(self.dt_proj(delta_raw)) + 1e-4
        b_scale = torch.tanh(b_raw)

        B = self.B_base(u) * torch.tanh(self.B_gate(u))
        return delta, b_scale, B, res_gate, u

    def ODE_eq(self, x_input, y):
        u = self._prepare_u(x_input, use_upper=False)
        delta, b_scale, B, _, _ = self._compute_params(u)
        A = self._expand_A(y)
        f = delta * (A * y + b_scale * B)
        return f

    def ODE_eq_pre(self, x_input, y):
        u = self._prepare_u(x_input, use_upper=True)
        delta, b_scale, B, _, _ = self._compute_params(u)
        A = self._expand_A(y)
        f = delta * (A * y + b_scale * B)
        return f

    def step1_order1_explicit(self, x1, y1):
        f1 = self.ODE_eq(x1, y1)
        y2 = y1 + self.step * f1
        return y2, f1

    def step1_order2_implicit(self, x1, y1, x2):
        y2_pre, f1 = self.step1_order1_explicit(x1, y1)
        f2_pre = self.ODE_eq_pre(x2, y2_pre)
        y2 = y1 + (self.step / 2) * (f1 + f2_pre)
        return y2, f1

    def steps2_order2_explicit(self, f1, x2, y2):
        f2 = self.ODE_eq(x2, y2)
        y3 = y2 + (self.step / 2) * (3 * f2 - f1)
        return y3, f2

    def steps2_order3_implicit(self, f1, x2, y2, x3):
        y3_pre, f2 = self.steps2_order2_explicit(f1, x2, y2)
        f3_pre = self.ODE_eq_pre(x3, y3_pre)
        y3 = y2 + (self.step / 12) * (5 * f3_pre + 8 * f2 - f1)
        return y3, f2

    def steps3_order3_explicit(self, f1, f2, x3, y3):
        f3 = self.ODE_eq(x3, y3)
        y4 = y3 + (self.step / 12) * (23 * f3 - 16 * f2 + 5 * f1)
        return y4, f3

    def steps3_order4_implicit(self, f1, f2, x3, y3, x4):
        y4_pre, f3 = self.steps3_order3_explicit(f1, f2, x3, y3)
        f4_pre = self.ODE_eq_pre(x4, y4_pre)
        y4 = y3 + (self.step / 24) * (9 * f4_pre + 19 * f3 - 5 * f2 + f1)
        return y4, f3

    def steps4_order4_explicit(self, f1, f2, f3, x4, y4):
        f4 = self.ODE_eq(x4, y4)
        y5 = y4 + (self.step / 24) * (55 * f4 - 59 * f3 + 37 * f2 - 9 * f1)
        return y5, f4

    def steps4_order4_implicit(self, f1, f2, f3, x4, y4, x5):
        y5_pre, f4 = self.steps4_order4_explicit(f1, f2, f3, x4, y4)
        f5_pre = self.ODE_eq_pre(x5, y5_pre)
        y5 = y4 + (self.step / 24) * (9 * f5_pre + 19 * f4 - 5 * f3 + f2)
        return y5, f4

    def _finalize_output(self, y_state: torch.Tensor, x_input: torch.Tensor):
        # 用当前跳连重新算 gate/D，作为输出调制
        u = self._prepare_u(x_input, use_upper=False)
        _, _, _, res_gate, u_raw = self._compute_params(u)
        y_out = self.out_proj(y_state)
        y_out = y_out * F.silu(res_gate)
        d_shape = [1, self.hidden_channels] + [1] * (y_out.ndim - 2)
        y_out = y_out + self.D.view(*d_shape) * u_raw
        y_out = self.activation(y_out)
        return y_out

    def forward(self, f, x, y):
        # 多步积分，同 LinearSSMMultiStepDecoderCell
        if self.current_stage == self.num_stage:
            if self.current_stage == 2:
                y_new, f_curr = self.steps2_order2_explicit(f[0], x[-self.current_stage], y)
            elif self.current_stage == 3:
                y_new, f_curr = self.steps3_order3_explicit(f[0], f[1], x[-self.current_stage], y)
            else:
                y_new, f_curr = self.steps4_order4_explicit(f[0], f[1], f[2], x[-self.current_stage], y)
        else:
            if self.current_stage == 1:
                y_new, f_curr = self.step1_order2_implicit(x[-self.current_stage], y, x[-self.current_stage - 1])
            elif self.current_stage == 2:
                y_new, f_curr = self.steps2_order3_implicit(f[0], x[-self.current_stage], y, x[-self.current_stage - 1])
            elif self.current_stage == 3:
                y_new, f_curr = self.steps3_order4_implicit(f[0], f[1], x[-self.current_stage], y, x[-self.current_stage - 1])
            else:
                y_new, f_curr = self.steps4_order4_implicit(f[-self.current_stage + 4],
                                                            f[-self.current_stage + 3],
                                                            f[-self.current_stage + 2],
                                                            x[-self.current_stage],
                                                            y,
                                                            x[-self.current_stage - 1])
        y_out = self._finalize_output(y_new, x[-self.current_stage])
        return y_out, f_curr
