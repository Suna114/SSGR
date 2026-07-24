#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint, choice
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.sh_utils import SH2RGB
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from typing import List, Optional, Tuple
import numpy as np
import re

_SOBEL_KERNEL_X: Optional[torch.Tensor] = None
_SOBEL_KERNEL_Y: Optional[torch.Tensor] = None
_SAR_NAN_DIAG_ONCE = False
_SAR_PARAM_FIX_LOGGED = False
_SAR_DENSIFY_MISMATCH_LOGGED = False
_SAR_CUDA_ABORT = False


def _safe_cuda_isfinite(x) -> bool:
    """CUDA 上下文损坏时 isfinite 也会抛错；安全返回 False。"""
    if not isinstance(x, torch.Tensor):
        return bool(x == x)
    try:
        return bool(torch.isfinite(x).all().item()) if x.numel() > 1 else bool(torch.isfinite(x).item())
    except RuntimeError:
        return False


def _loss_scalar_isfinite(loss) -> bool:
    if not isinstance(loss, torch.Tensor):
        import math
        return math.isfinite(float(loss))
    try:
        return bool(torch.isfinite(loss).item())
    except RuntimeError:
        return False


def _loss_term_finite(term) -> bool:
    if isinstance(term, torch.Tensor):
        return term.numel() > 0 and bool(torch.isfinite(term).all().item())
    if isinstance(term, (int, float)):
        import math
        return math.isfinite(float(term))
    return False


def _coerce_visibility_index_tensor(idx) -> torch.Tensor:
    """兼容 nonzero 返回 tuple / 0-dim / [M,1] 等多种格式。"""
    if idx is None:
        return torch.empty(0, dtype=torch.long, device="cuda")
    if isinstance(idx, (tuple, list)):
        if not idx:
            return torch.empty(0, dtype=torch.long, device="cuda")
        idx = idx[0]
    if not isinstance(idx, torch.Tensor):
        return torch.empty(0, dtype=torch.long, device="cuda")
    return idx.reshape(-1).long()


def _normalize_gaussian_index(idx, n: int) -> torch.Tensor:
    """将 visibility / densify 索引规范为 [0, n) 内的 1D long 向量。"""
    flat = _coerce_visibility_index_tensor(idx)
    if n <= 0 or flat.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=flat.device)
    valid = (flat >= 0) & (flat < n)
    return flat[valid]


def _align_radii_to_count(radii: torch.Tensor, n: int) -> torch.Tensor:
    if n <= 0:
        return radii.new_zeros(0)
    flat = radii.reshape(-1)
    if flat.numel() == n:
        return flat
    if flat.numel() > n:
        return flat[:n]
    return torch.cat([flat, flat.new_zeros(n - flat.numel())], dim=0)


def _apply_densify_visibility_stats(
    gaussians: GaussianModel,
    viewspace_point_tensor: torch.Tensor,
    visibility_filter,
    radii: torch.Tensor,
) -> None:
    """安全更新 max_radii2D 与 densify 梯度统计（SDGR 路径下 buffer 长度可能短暂不一致）。"""
    gaussians.ensure_densification_buffers()
    n = int(gaussians.get_xyz.shape[0])
    n_vs = int(viewspace_point_tensor.shape[0])
    n_eff = min(n, n_vs, int(gaussians.xyz_gradient_accum.shape[0]))
    if n_eff <= 0:
        return
    global _SAR_DENSIFY_MISMATCH_LOGGED
    if (n_vs != n or int(gaussians.max_radii2D.shape[0]) != n) and not _SAR_DENSIFY_MISMATCH_LOGGED:
        _SAR_DENSIFY_MISMATCH_LOGGED = True
        print(
            f"[SAR densify] 对齐 buffer: gaussians={n}, viewspace={n_vs}, "
            f"max_radii2D={gaussians.max_radii2D.shape[0]}, accum={gaussians.xyz_gradient_accum.shape[0]}"
        )
    vf = _normalize_gaussian_index(visibility_filter, n_eff)
    if vf.numel() == 0:
        return
    vf = torch.unique(vf)
    r = _align_radii_to_count(radii, n_eff)
    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], r[vf])
    if viewspace_point_tensor.grad is not None:
        gaussians.add_densification_stats(viewspace_point_tensor, vf)


def _zero_nonfinite_gradients(gaussians: GaussianModel) -> int:
    """将 NaN/Inf 梯度置零，避免 clip_grad_norm_ 把 NaN 扩散到整组参数。"""
    cleared = 0
    for group in gaussians.optimizer.param_groups:
        p = group["params"][0]
        if p.grad is None:
            continue
        bad = ~torch.isfinite(p.grad)
        if bad.any():
            cleared += int(bad.sum().item())
            p.grad.data[bad] = 0.0
    for p in gaussians.exposure_optimizer.param_groups[0]["params"]:
        if p.grad is None:
            continue
        bad = ~torch.isfinite(p.grad)
        if bad.any():
            cleared += int(bad.sum().item())
            p.grad.data[bad] = 0.0
    return cleared


def _clip_gaussian_gradients(gaussians: GaussianModel, max_norm: float) -> None:
    if max_norm <= 0:
        return
    _zero_nonfinite_gradients(gaussians)
    params = [p for g in gaussians.optimizer.param_groups for p in g["params"] if p.grad is not None]
    if params:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if isinstance(total_norm, torch.Tensor) and not torch.isfinite(total_norm):
            for p in params:
                if p.grad is not None:
                    p.grad.detach_().zero_()


def _sar_gray_sobel_magnitude(rgb_chw: torch.Tensor) -> torch.Tensor:
    """灰度 Sobel 梯度幅值 (H,W)，输入 (3,H,W)，可微。"""
    global _SOBEL_KERNEL_X, _SOBEL_KERNEL_Y
    g = rgb_chw.mean(dim=0, keepdim=True).unsqueeze(0)  # 1,1,H,W
    if _SOBEL_KERNEL_X is None:
        _SOBEL_KERNEL_X = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        _SOBEL_KERNEL_Y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
    kx = _SOBEL_KERNEL_X.to(device=rgb_chw.device, dtype=rgb_chw.dtype)
    ky = _SOBEL_KERNEL_Y.to(device=rgb_chw.device, dtype=rgb_chw.dtype)
    gx = torch.nn.functional.conv2d(g, kx, padding=1).squeeze(0).squeeze(0)
    gy = torch.nn.functional.conv2d(g, ky, padding=1).squeeze(0).squeeze(0)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def _sar_depth_order_aggregate(flat_depth: torch.Tensor, reduce_mode: str) -> torch.Tensor:
    """
    SAR 深度顺序损失：对一个区域内的渲染深度做单标量汇总。
    mean：与旧行为一致；median：对掩码边缘拉丝、少量异类像素混进区域内的离群深度更稳健。
    """
    if flat_depth.numel() == 0:
        raise RuntimeError("SAR depth order aggregate on empty mask")
    m = (reduce_mode or "mean").lower().strip()
    if m == "median":
        return flat_depth.median()
    return flat_depth.mean()


# ============================================================================
# SAR-GS默认配置（MSTAR数据集推荐参数）
# ============================================================================
SAR_DEFAULT_CONFIG = {
    # SAR模式开关
    'use_sar_mode': True,              # 是否启用SAR模式（设为False使用标准3D-GS）
    'use_sar_rendering': True,         # 是否使用SAR渲染器（Python版SDGR）
    
    # MSTAR数据集标准参数
    'sar_camera_height': 12000.0,      # SAR平台高度（米）- MSTAR典型值
    'sar_platform_velocity': 250.0,    # 平台速度（m/s）- MSTAR典型值
    'sar_prf': 2500.0,                 # 脉冲重复频率（Hz）- MSTAR典型值
    'sar_bandwidth': 500e6,            # 带宽（Hz）- MSTAR典型值：500 MHz
    'sar_image_size_azimuth': 512,     # 方位向图像尺寸
    'sar_image_size_range': 512,       # 距离向图像尺寸
    
    # 相机分布和几何
    'sar_camera_distribution_mode': 'ring',  # 水平圆环（360° 方位/SAR 卫星轨迹）；多俯角可用 sphere
    'sar_depression_angle': 31.57,     # 中心俯视角（度）- MSTAR常用15°或45°
    'sar_radius_scale': 1.0,           # 相机半径缩放因子
    
    # 形状约束（防止点云塌陷成2D圆饼）
    'sar_shape_constraint_weight': 0.1,       # 形状约束权重（增强10倍，约束Z方向）
    'sar_shape_constraint_start_iter': 500,   # 提前开始（从500迭代）
    'sar_shape_constraint_end_iter': 25000,   # 延长约束时间
    
    # 训练参数
    'iterations': 30000,               # 总训练迭代次数（SAR图像建议30000）
    'position_lr_init': 0.00016,       # 位置学习率初始值
    'lambda_dssim': 0.2,               # SSIM损失权重
    'densify_until_iter': 15000,       # 密集化停止迭代
}

# 快速配置预设（用于测试和调试）
SAR_FAST_CONFIG = {
    **SAR_DEFAULT_CONFIG,
    'iterations': 5000,
    'densify_until_iter': 2500,
    'sar_shape_constraint_end_iter': 3000,
}

# 高质量配置预设（用于最终结果）
SAR_HIGH_QUALITY_CONFIG = {
    **SAR_DEFAULT_CONFIG,
    'iterations': 50000,
    'densify_until_iter': 25000,
    'sar_shape_constraint_end_iter': 35000,
    'position_lr_init': 0.00012,  # 降低学习率以获得更稳定的结果
}

# 选择使用的配置（修改这里来切换配置）
# ACTIVE_SAR_CONFIG = SAR_FAST_CONFIG   
ACTIVE_SAR_CONFIG = SAR_DEFAULT_CONFIG  # 可改为 SAR_FAST_CONFIG 或 SAR_HIGH_QUALITY_CONFIG
# ACTIVE_SAR_CONFIG = SAR_HIGH_QUALITY_CONFIG
def print_sar_config():
    """打印当前SAR配置"""
    print("\n" + "="*70)
    print("当前SAR配置")
    print("="*70)
    
    if not ACTIVE_SAR_CONFIG['use_sar_mode']:
        print("⚠️  SAR模式已禁用，使用标准3D Gaussian Splatting")
        return
    
    print(f"✅ SAR模式: 启用")
    print(f"✅ SAR渲染器: {'启用 (Python版SDGR)' if ACTIVE_SAR_CONFIG['use_sar_rendering'] else '禁用'}")
    print(f"\n【MSTAR数据集参数】")
    print(f"  平台高度: {ACTIVE_SAR_CONFIG['sar_camera_height']:.1f} m")
    print(f"  平台速度: {ACTIVE_SAR_CONFIG['sar_platform_velocity']:.1f} m/s")
    print(f"  脉冲重复频率: {ACTIVE_SAR_CONFIG['sar_prf']:.1f} Hz")
    print(f"  带宽: {ACTIVE_SAR_CONFIG['sar_bandwidth']/1e6:.0f} MHz")
    print(f"  图像尺寸: {ACTIVE_SAR_CONFIG['sar_image_size_azimuth']}×{ACTIVE_SAR_CONFIG['sar_image_size_range']}")
    print(f"\n【几何参数】")
    print(f"  相机分布: {ACTIVE_SAR_CONFIG['sar_camera_distribution_mode']}")
    print(f"  中心俯视角: {ACTIVE_SAR_CONFIG['sar_depression_angle']:.2f}°")
    print(f"\n【形状约束】")
    print(f"  约束权重: {ACTIVE_SAR_CONFIG['sar_shape_constraint_weight']}")
    print(f"  约束迭代: {ACTIVE_SAR_CONFIG['sar_shape_constraint_start_iter']} - {ACTIVE_SAR_CONFIG['sar_shape_constraint_end_iter']}")
    print(f"\n【训练参数】")
    print(f"  总迭代次数: {ACTIVE_SAR_CONFIG['iterations']}")
    print(f"  位置学习率: {ACTIVE_SAR_CONFIG['position_lr_init']}")
    print(f"  密集化截止: {ACTIVE_SAR_CONFIG['densify_until_iter']}")
    print("="*70 + "\n")

def _cli_specifies_dest(argv: Optional[List[str]], dest: str) -> bool:
    """检测 argv 中是否出现 argparse 长选项 --<dest> 或 --<dest>=…（dest 与 Namespace 字段名一致，含下划线）。"""
    if not argv:
        return False
    flag = "--" + dest
    pre = flag + "="
    for a in argv:
        if a == flag or a.startswith(pre):
            return True
    return False


def apply_sar_config_to_args(args, argv: Optional[List[str]] = None):
    """将 ACTIVE_SAR_CONFIG 合并到 args。对在命令行中显式出现的选项（--name 或 --name=）不覆盖。"""
    if argv is None:
        argv = sys.argv

    # SAR模式参数
    if hasattr(args, 'sar_mode') and args.sar_mode is None:
        args.sar_mode = ACTIVE_SAR_CONFIG['use_sar_mode']

    sar_param_mapping = {
        'sar_camera_height': 'sar_camera_height',
        'sar_platform_velocity': 'sar_platform_velocity',
        'sar_prf': 'sar_prf',
        'sar_bandwidth': 'sar_bandwidth',
        'sar_image_size_azimuth': 'sar_image_size_azimuth',
        'sar_image_size_range': 'sar_image_size_range',
        'sar_camera_distribution_mode': 'sar_camera_distribution_mode',
        'sar_depression_angle': 'sar_depression_angle',
        'sar_radius_scale': 'sar_radius_scale',
        'sar_shape_constraint_weight': 'sar_shape_constraint_weight',
        'sar_shape_constraint_start_iter': 'sar_shape_constraint_start_iter',
        'sar_shape_constraint_end_iter': 'sar_shape_constraint_end_iter',
    }

    for config_key, arg_key in sar_param_mapping.items():
        if _cli_specifies_dest(argv, arg_key):
            continue
        if hasattr(args, arg_key) and config_key in ACTIVE_SAR_CONFIG:
            setattr(args, arg_key, ACTIVE_SAR_CONFIG[config_key])

    if hasattr(args, 'iterations') and 'iterations' in ACTIVE_SAR_CONFIG:
        if not _cli_specifies_dest(argv, "iterations"):
            if args.iterations == 30000:
                args.iterations = ACTIVE_SAR_CONFIG['iterations']

    if hasattr(args, 'position_lr_init') and 'position_lr_init' in ACTIVE_SAR_CONFIG:
        if not _cli_specifies_dest(argv, "position_lr_init"):
            if args.position_lr_init == 0.00016:
                args.position_lr_init = ACTIVE_SAR_CONFIG['position_lr_init']

    if hasattr(args, 'densify_until_iter') and 'densify_until_iter' in ACTIVE_SAR_CONFIG:
        if not _cli_specifies_dest(argv, "densify_until_iter"):
            if args.densify_until_iter == 15000:
                args.densify_until_iter = ACTIVE_SAR_CONFIG['densify_until_iter']

    return args


def apply_sar_detail_focus_overrides(args, argv: Optional[List[str]] = None):
    """
    「细节优先」预设：弱化长时间形状盒约束（两阶段早期关闭）、去掉易压成盒子的尺寸/尺度损失，
    加强 densify、略放宽目标高斯尺度上限、加快球谐升阶；默认打开方位分层采样。
    若命令行已显式传入某参数（--name 或 --name=），则该项不会被本预设覆盖。
    宜配合 <数据根>/mstar_nosfm_init/points3D.ply（CAD 密采样含炮塔）。
    方位分桶：--sar_azimuth_bins N；N≤0 时桶数取训练相机数量（与视角数对齐）。
    """
    if argv is None:
        argv = sys.argv
    o = {
        "sar_shape_constraint_weight": 0.04,
        "sar_shape_constraint_end_iter": 7500,
        "sar_target_dimension_constraint_weight": 0.0,
        "sar_scale_constraint_weight": 0.0,
        "sar_target_height_constraint_weight": 0.0,
        "sar_height_constraint_weight": 0.0,
        "sar_ground_constraint_weight": 0.0,
        "sar_spread_constraint_weight": 0.0,
        "sar_shadow_constraint_weight": 0.0,
        "sar_multiview_consistency_weight": 0.0,
        "densify_until_iter": 22000,
        "densify_grad_threshold": 0.0002,
        "densification_interval": 80,
        "percent_dense": 0.015,
        "sar_target_max_scaling": 0.22,
        "sh_degree_interval": 600,
        "sar_stratify_azimuth": True,
    }
    skipped = [k for k in o.keys() if _cli_specifies_dest(argv, k)]
    for k, v in o.items():
        if _cli_specifies_dest(argv, k):
            continue
        if hasattr(args, k):
            setattr(args, k, v)
    if hasattr(args, "iterations"):
        if not _cli_specifies_dest(argv, "iterations") and int(getattr(args, "iterations", 0)) < 35000:
            args.iterations = 35000
    print("\n" + "=" * 62)
    print("sar_detail_focus：已应用细节优先参数（命令行已指定的项未覆盖）")
    print("=" * 62)
    for k, v in o.items():
        if hasattr(args, k):
            note = "  [保留命令行]" if k in skipped else ""
            print(f"  {k}: {getattr(args, k)}{note}")
    iter_cli = _cli_specifies_dest(argv, "iterations")
    print(f"  iterations: {getattr(args, 'iterations', '?')}{'  [保留命令行]' if iter_cli else ''}")
    unapplied = list(skipped)
    if iter_cli:
        unapplied.append("iterations")
    if unapplied:
        print(f"  （预设未覆盖: {', '.join(unapplied)}）")
    print("=" * 62 + "\n")


# ============================================================================
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

_SAR_INTERP_SOFT_LOGGED = False


def _parse_azimuth_from_image_name(image_name: str):
    """从 MSTAR 风格文件名解析方位角，失败返回 None（不打印日志，避免训练刷屏）。"""
    if not image_name:
        return None
    stem = os.path.splitext(os.path.basename(image_name))[0]
    m = re.search(r"([A-Za-z0-9]+)-(\d+)-(\d+)", stem)
    if m:
        return float(m.group(3))
    return None


def _get_camera_azimuth_deg(cam) -> Optional[float]:
    """优先使用相机属性，否则解析 image_name。"""
    if hasattr(cam, "azimuth_angle") and cam.azimuth_angle is not None:
        return float(cam.azimuth_angle) % 360.0
    name = getattr(cam, "image_name", None)
    if name:
        a = _parse_azimuth_from_image_name(str(name))
        if a is not None:
            return a % 360.0
    return None


def _build_azimuth_strata(train_cameras: list, n_bins: int) -> Tuple[Optional[List[List[int]]], str]:
    """
    将训练相机索引按方位角 [0,360) 均分入 n_bins 个桶。
    返回 (每个非空桶的索引列表组成的列表, 诊断信息)；若无法为任何相机解析方位则返回 (None, reason)。
    """
    if n_bins < 1:
        return None, "sar_azimuth_bins < 1"
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    n_ok = 0
    for idx, cam in enumerate(train_cameras):
        azi = _get_camera_azimuth_deg(cam)
        if azi is None:
            continue
        b = int(azi % 360.0 / 360.0 * n_bins)
        b = min(b, n_bins - 1)
        bins[b].append(idx)
        n_ok += 1
    if n_ok == 0:
        return None, "无法解析任何训练相机的方位角"
    nonempty = sum(1 for b in bins if len(b) > 0)
    diag = f"方位分层: {n_bins} 桶, 已解析 {n_ok}/{len(train_cameras)} 视角, 非空桶 {nonempty}"
    return bins, diag


def _sar_dc_luminance_densify_grad_mult(
    gaussians: GaussianModel,
    opt: Namespace,
    dataset,
    logged_once: List[bool],
) -> Optional[torch.Tensor]:
    """按 DC 粗分「阴影 / 中暗背景」两档缩放 densify 梯度阈值倍数（更小 mult → 越容易 clone/split）。"""
    if not getattr(dataset, "sar_mode", False):
        return None
    if not bool(getattr(opt, "sar_dense_by_dc_luminance", False)):
        return None
    if gaussians.get_xyz.shape[0] < 1:
        return None
    if not hasattr(gaussians, "get_dc_luminosity"):
        return None

    lo_sh = float(getattr(opt, "sar_dense_lumi_shadow_upper", 0.09))
    hi_bg = float(getattr(opt, "sar_dense_lumi_background_upper", 0.28))
    m_sh = float(getattr(opt, "sar_dense_shadow_grad_thresh_mult", 1.7))
    m_bg = float(getattr(opt, "sar_dense_background_grad_thresh_mult", 0.55))
    if hi_bg <= lo_sh + 1e-6:
        hi_bg = lo_sh + 0.06

    with torch.no_grad():
        lum = gaussians.get_dc_luminosity().detach()
        m = torch.ones_like(lum)
        sh_ix = lum <= lo_sh
        bg_ix = (lum > lo_sh) & (lum <= hi_bg)
        m[sh_ix] = m_sh
        m[bg_ix] = m_bg

    if not logged_once[0]:
        logged_once[0] = True
        print(
            f"[SAR densify tiers] lum≤{lo_sh:.3f}→阴影 mult={m_sh:.3f}（增生更难），"
            f"({lo_sh:.3f},{hi_bg:.3f}]→中暗背景 mult={m_bg:.3f}（增生更易），"
            "更亮视作目标一档 mult=1.0；可依数据调 thresholds / mult。"
        )

    return m


def _sar_apply_dense_shadow_scaling_log_cap(gaussians: GaussianModel, opt: Namespace, dataset) -> None:
    """densify 后对最暗一档压 log-scale 上限，减小阴影椭球（需与 sar_dense_by_dc_luminance 同开）。"""
    if not getattr(dataset, "sar_mode", False):
        return
    if not bool(getattr(opt, "sar_dense_by_dc_luminance", False)):
        return
    cap = float(getattr(opt, "sar_dense_shadow_scaling_log_cap", -999.0))
    if cap < -900.0:
        return
    lo_sh = float(getattr(opt, "sar_dense_lumi_shadow_upper", 0.09))
    with torch.no_grad():
        lum = gaussians.get_dc_luminosity().detach()
        mask = lum <= lo_sh
        if not torch.any(mask):
            return
        idx = torch.where(mask)[0]
        capped = gaussians._scaling.data[idx].clamp(max=cap)
        gaussians._scaling.data[idx] = capped


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    global _SAR_CUDA_ABORT
    _SAR_CUDA_ABORT = False
    _last_iter = 0

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    else:
        try:
            from sar.sar_seed_refine_bg_shadow import maybe_seed_sar_refine_bg_shadow_at_init

            maybe_seed_sar_refine_bg_shadow_at_init(
                gaussians,
                opt,
                dataset_sar_mode=bool(getattr(dataset, "sar_mode", False)),
                checkpoint_was_loaded=False,
                scene=scene,
            )
        except ImportError as e:
            print(f"[SAR refine seed] 跳过（无法导入 sar.sar_seed_refine_bg_shadow）：{e}")

    gaussians.ensure_densification_buffers()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # SAR refine 亮度帽：仅当未从 checkpoint 恢复时，在首迭代对「初始」点云快照（见迭代内）
    refine_brightness_cap_fresh_start = checkpoint is None

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    train_cameras_list = scene.getTrainCameras().copy()
    try:
        from sar.render_backend import print_sar_render_backend
        print_sar_render_backend(
            dataset,
            pipe,
            train_cameras_list,
            use_sar_rendering_config=ACTIVE_SAR_CONFIG.get("use_sar_rendering", True),
        )
    except ImportError:
        pass
    if getattr(dataset, "sar_mode", False) and getattr(dataset, "scattering_masks", None):
        try:
            from sar.semantic_mask_config import audit_scattering_masks_on_cameras
            audit_scattering_masks_on_cameras(train_cameras_list)
        except ImportError:
            pass
    sar_stratify_bins: Optional[List[List[int]]] = None
    if getattr(opt, "sar_stratify_azimuth", False) and getattr(dataset, "sar_mode", False):
        nb_raw = int(getattr(opt, "sar_azimuth_bins", 24))
        n_cam = len(train_cameras_list)
        if nb_raw <= 0:
            nb = max(1, n_cam)
            auto_bins = True
        else:
            nb = max(1, nb_raw)
            auto_bins = False
        sar_stratify_bins, strat_diag = _build_azimuth_strata(train_cameras_list, nb)
        if sar_stratify_bins is not None:
            extra = (
                f"；桶数={'训练相机数=' + str(n_cam) + '（sar_azimuth_bins≤0 自动）' if auto_bins else '手动指定 nb=' + str(nb)}"
            )
            print(f"   ✅ SAR {strat_diag}{extra}")
        else:
            print(f"   ⚠️  SAR 方位分层未启用: {strat_diag}，退回随机视角抽样")
            sar_stratify_bins = None
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    sar_dense_tier_log_holder = [False]

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True, leave=True)
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        _last_iter = iteration
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    use_sar_mode = getattr(dataset, 'sar_mode', False) and getattr(pipe, 'sar_rendering', False)
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, sar_mode=use_sar_mode)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # SAR refine：首次迭代记录「训练开始前」点数与 DC 亮度上沿，用于约束 densify 新增点不得亮过该上沿（见 sar_refine_target_bg 预设）
        refine_bg = bool(getattr(opt, "sar_refine_target_bg", False))
        cap_disabled = bool(getattr(opt, "sar_refine_brightness_cap_disable", False))
        w_cap = float(getattr(opt, "sar_refine_brightness_cap_weight", 0.0) or 0.0)
        new_luma_frac = float(getattr(opt, "sar_refine_new_point_max_luma_fraction", 0.0) or 0.0)
        if refine_bg and not cap_disabled and w_cap <= 1e-12 and new_luma_frac <= 1e-6:
            w_cap = 0.12  # 旧 exp15-5 回落；BG/SH 预设已显式关闭
        try:
            from sar.sar_refine_bg_shadow_train import maybe_snapshot_refine_target_luma

            maybe_snapshot_refine_target_luma(
                gaussians, opt, fresh_start=refine_brightness_cap_fresh_start,
            )
        except ImportError:
            pass
        if (
            refine_brightness_cap_fresh_start
            and w_cap > 0.0
            and not cap_disabled
            and not getattr(gaussians, "_refine_brightness_snapshotted", False)
        ):
            with torch.no_grad():
                n0_snap = int(gaussians.get_xyz.shape[0])
                luma_s = gaussians.get_dc_luminosity()
                q_snap = float(getattr(opt, "sar_refine_brightness_cap_quantile", 0.99))
                q_snap = min(0.9999, max(0.5, q_snap))
                cap_snap = torch.quantile(luma_s, q_snap)
                med_snap = torch.median(luma_s)
                gaussians._refine_brightness_snapshotted = True
                gaussians._refine_brightness_n0 = n0_snap
                gaussians._refine_brightness_cap_max = cap_snap.detach()
                gaussians._refine_brightness_cap_median = med_snap.detach()
            med_extra = float(getattr(opt, "sar_refine_brightness_cap_median_margin", 0.0) or 0.0)
            print(
                f"[SAR refine] DC 亮度帽快照：点数={n0_snap}，p{q_snap:.3f}≈{float(cap_snap.detach()):.4f}，"
                f"median≈{float(med_snap.detach()):.4f}；"
                f"新增点亮度超过 min(上沿−margin, median−median_margin) 将被惩罚"
                + ("" if med_extra <= 1e-12 else f"（median_margin={med_extra:.4f}）")
            )

        # Increase SH degree periodically (sh_degree_interval 越小越早用满表示能力)
        sh_int = int(getattr(opt, "sh_degree_interval", 1000))
        if sh_int > 0 and iteration % sh_int == 0:
            gaussians.oneupSHdegree()

        # Pick a training camera: optional SAR 方位分层（均匀选桶再随机）
        if sar_stratify_bins is not None:
            nonempty_bins = [i for i, b in enumerate(sar_stratify_bins) if len(b) > 0]
            if nonempty_bins:
                bi = choice(nonempty_bins)
                ci = choice(sar_stratify_bins[bi])
                viewpoint_cam = train_cameras_list[ci]
            else:
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_indices = list(range(len(viewpoint_stack)))
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_idx)
                viewpoint_indices.pop(rand_idx)
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ========== SAR渲染模式 ==========
        # 检查是否使用SAR渲染模式（使用SDGR - SAR Differentiable Gaussian Splatting Rasterizer）
        use_sar_mode = (getattr(dataset, 'sar_mode', False) and 
                       getattr(viewpoint_cam, 'use_sar_rendering', False) and
                       ACTIVE_SAR_CONFIG.get('use_sar_rendering', False))

        if use_sar_mode:
            nfix = gaussians.sanitize_nonfinite_parameters(reset_optimizer=True)
            if nfix > 0:
                global _SAR_PARAM_FIX_LOGGED
                if not _SAR_PARAM_FIX_LOGGED or iteration <= 20:
                    print(
                        f"\n[SAR iter {iteration}] 已修复非有限高斯参数 {nfix} 个元素"
                        f"（含 Adam 动量重置）"
                    )
                _SAR_PARAM_FIX_LOGGED = True
                if iteration > 1:
                    from sar.rasterizer import sar_mark_cuda_unstable
                    sar_mark_cuda_unstable(
                        f"iter {iteration} 出现 {nfix} 个非有限参数",
                        auto_disable_after=1,
                    )
        
        # 渲染（使用SDGR或标准渲染器）
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                          use_trained_exp=dataset.train_test_exp, 
                          separate_sh=SPARSE_ADAM_AVAILABLE, 
                          sar_mode=use_sar_mode)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        n_gs = gaussians.get_xyz.shape[0]
        visibility_filter = _normalize_gaussian_index(visibility_filter, n_gs)
        radii = _align_radii_to_count(radii, n_gs)
        image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        if use_sar_mode:
            from sar.rasterizer import sar_cuda_sync_check
            if not sar_cuda_sync_check(f"iter {iteration} render", fatal=False):
                _SAR_CUDA_ABORT = True
                print(
                    f"\n[SAR] iter {iteration} CUDA 渲染异常，训练中止。"
                    f"请新开 output 目录并用 SAR_SDGR_FORCE_PYTHON=1 重跑。"
                )
                break

        # 应用alpha掩码（如果有）
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image = image * alpha_mask

        # ========== 计算损失函数 ==========
        gt_image = viewpoint_cam.original_image.cuda()
        
        # SAR模式：可选地应用散射强度掩码来只在目标区域计算损失
        L_depth_order = 0.0
        lambda_dssim_eff = float(opt.lambda_dssim)
        if use_sar_mode and hasattr(viewpoint_cam, 'scattering_mask') and viewpoint_cam.scattering_mask is not None:
            import torch.nn.functional as F
            # 将散射强度掩码转换为损失权重
            # 掩码值：0=背景, 1=目标(含原边缘), 3=阴影；旧缓存或简易三类仍可能含 2=边缘
            scattering_mask = viewpoint_cam.scattering_mask
            if isinstance(scattering_mask, np.ndarray):
                scattering_mask = torch.from_numpy(scattering_mask).float().cuda()
            else:
                scattering_mask = scattering_mask.float().cuda()
            
            # 调整掩码尺寸以匹配图像
            if scattering_mask.shape != image.shape[1:]:
                scattering_mask = F.interpolate(
                    scattering_mask.unsqueeze(0).unsqueeze(0), 
                    size=image.shape[1:], 
                    mode='nearest'
                ).squeeze()

            t_only_until = int(getattr(opt, "sar_target_only_loss_until_iter", 0) or 0)
            in_target_only = t_only_until > 0 and iteration <= t_only_until

            if in_target_only:
                inc_edge = bool(getattr(opt, "sar_target_only_loss_include_edge", True))
                if inc_edge:
                    fit_m = (scattering_mask == 1) | (scattering_mask == 2)
                else:
                    fit_m = scattering_mask == 1
                denom = fit_m.float().sum() + 1e-8
                if denom > 64:
                    Ll1 = (torch.abs(image - gt_image) * fit_m.float().unsqueeze(0)).sum() / denom
                else:
                    Ll1 = l1_loss(image, gt_image)
                L_depth_order = 0.0
                if bool(getattr(opt, "sar_target_only_phase_disable_ssim", True)):
                    lambda_dssim_eff = 0.0
            else:
                if bool(getattr(opt, "sar_refine_bg_shadow_only_loss", False)):
                    try:
                        from sar.sar_refine_bg_shadow_train import apply_bg_shadow_only_loss_weights

                        loss_weight = apply_bg_shadow_only_loss_weights(
                            scattering_mask,
                            opt,
                            bg_w=float(getattr(opt, "background_loss_weight", 0.5)),
                            sh_w=float(getattr(opt, "shadow_loss_weight", 0.55)),
                        )
                        Ll1 = (torch.abs(image - gt_image) * loss_weight.unsqueeze(0)).mean()
                    except ImportError:
                        bg_m = scattering_mask == 0
                        sh_m = scattering_mask == 3
                        fit_m = bg_m | sh_m
                        denom = fit_m.float().sum() + 1e-8
                        if denom > 64:
                            Ll1 = (torch.abs(image - gt_image) * fit_m.float().unsqueeze(0)).sum() / denom
                        else:
                            Ll1 = l1_loss(image, gt_image)
                elif bool(getattr(opt, "sar_uniform_region_loss", False)):
                    loss_weight = torch.ones_like(scattering_mask)
                    loss_weight = loss_weight / loss_weight.mean()
                    Ll1 = (torch.abs(image - gt_image) * loss_weight.unsqueeze(0)).mean()
                else:
                    bg_w = float(getattr(opt, "background_loss_weight", 0.5))
                    tgt_w = float(getattr(opt, "target_loss_weight", 1.0))
                    edge_w = float(getattr(opt, "edge_loss_weight", tgt_w))
                    sh_w = float(getattr(opt, "shadow_loss_weight", 0.55))
                    loss_weight = torch.ones_like(scattering_mask)
                    loss_weight[scattering_mask == 0] = bg_w
                    # 目标内部(1) 用 target_loss_weight；轮廓(2) 用 edge_loss_weight（叠掩致 GT 略糊时略抬 edge 可保边缘更利）
                    loss_weight[scattering_mask == 1] = tgt_w
                    loss_weight[scattering_mask == 2] = edge_w
                    loss_weight[scattering_mask == 3] = sh_w

                    # 归一化权重
                    loss_weight = loss_weight / loss_weight.mean()

                    # 应用权重到损失计算
                    Ll1 = (torch.abs(image - gt_image) * loss_weight.unsqueeze(0)).mean()

                w_do = float(getattr(opt, "sar_target_depth_order_weight", 0.0))
                margin = float(getattr(opt, "sar_target_depth_order_margin", 1e-3))
                # 阴影 vs 背景的深度顺序可分权：<0 时该子项复用 sar_target_depth_order_weight
                _w_sh = float(getattr(opt, "sar_shadow_depth_order_weight", -1.0))
                _w_bg = float(getattr(opt, "sar_background_depth_order_weight", -1.0))
                w_sh_depth = w_do if _w_sh < 0.0 else _w_sh
                w_bg_depth = w_do if _w_bg < 0.0 else _w_bg
                if (w_sh_depth > 0.0 or w_bg_depth > 0.0) and render_pkg.get("depth") is not None:
                    dep = render_pkg["depth"]
                    # SDGR 返回 per-Gaussian 深度 (N,)；深度顺序损失需要 (H,W) 渲染深度图
                    if dep.dim() >= 2:
                        if dep.dim() == 3:
                            dep = dep.mean(dim=0)
                        if dep.shape != scattering_mask.shape:
                            dep = F.interpolate(
                                dep.unsqueeze(0).unsqueeze(0),
                                size=scattering_mask.shape,
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        tgt_m = (scattering_mask == 1) | (scattering_mask == 2)
                        sh_m = scattering_mask == 3
                        bg_m = scattering_mask == 0
                        if tgt_m.any():
                            depth_red = str(
                                getattr(opt, "sar_depth_order_reduce", "mean")
                            ).lower().strip()
                            dt = _sar_depth_order_aggregate(
                                dep[tgt_m].reshape(-1), depth_red
                            )
                            if sh_m.any() and w_sh_depth > 0.0:
                                ds = _sar_depth_order_aggregate(
                                    dep[sh_m].reshape(-1), depth_red
                                )
                                if bool(getattr(opt, "sar_shadow_depth_before_target", False)):
                                    L_depth_order = (
                                        L_depth_order
                                        + w_sh_depth * torch.relu(ds - dt + margin)
                                    )
                                else:
                                    L_depth_order = (
                                        L_depth_order
                                        + w_sh_depth * torch.relu(dt - ds + margin)
                                    )
                            if bg_m.any() and w_bg_depth > 0.0:
                                db = _sar_depth_order_aggregate(
                                    dep[bg_m].reshape(-1), depth_red
                                )
                                L_depth_order = (
                                    L_depth_order + w_bg_depth * torch.relu(dt - db + margin)
                                )
        else:
            # 标准L1损失
            Ll1 = l1_loss(image, gt_image)

        lambda_dssim_eff = float(lambda_dssim_eff)
        if lambda_dssim_eff > 0.0:
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            if not torch.isfinite(ssim_value):
                ssim_value = image.new_tensor(0.0)
        else:
            ssim_value = image.new_tensor(0.0)

        L_grad_target = image.new_zeros(())
        wg = float(getattr(opt, "sar_target_gradient_l1_weight", 0.0))
        g_a = int(getattr(opt, "sar_target_gradient_start_iter", 0))
        g_b = int(getattr(opt, "sar_target_gradient_end_iter", 10**9))
        if (
            wg > 0.0
            and g_a <= iteration <= g_b
            and use_sar_mode
            and hasattr(viewpoint_cam, "scattering_mask")
            and viewpoint_cam.scattering_mask is not None
        ):
            import torch.nn.functional as Fn
            sm_raw = viewpoint_cam.scattering_mask
            if isinstance(sm_raw, np.ndarray):
                sm_t = torch.from_numpy(sm_raw).float().cuda()
            else:
                sm_t = sm_raw.float().cuda()
            if sm_t.shape != image.shape[1:]:
                sm_t = Fn.interpolate(
                    sm_t.unsqueeze(0).unsqueeze(0),
                    size=image.shape[1:],
                    mode="nearest",
                ).squeeze(0).squeeze(0)
            t_only_u = int(getattr(opt, "sar_target_only_loss_until_iter", 0) or 0)
            in_t_only = t_only_u > 0 and iteration <= t_only_u
            if in_t_only:
                if bool(getattr(opt, "sar_target_only_loss_include_edge", True)):
                    m = ((sm_t == 1) | (sm_t == 2)).float()
                else:
                    m = (sm_t == 1).float()
            else:
                if bool(getattr(opt, "sar_target_gradient_include_shadow", False)):
                    m = ((sm_t == 1) | (sm_t == 2) | (sm_t == 3)).float()
                else:
                    m = ((sm_t == 1) | (sm_t == 2)).float()
            if m.sum() < 64.0:
                m = (sm_t != 0).float()
            mag_i = _sar_gray_sobel_magnitude(image)
            mag_g = _sar_gray_sobel_magnitude(gt_image.detach())
            eb = float(getattr(opt, "sar_target_gradient_edge_boost", 1.0))
            if eb < 1.0:
                eb = 1.0
            wmap = m.clone()
            if eb > 1.001:
                if (sm_t == 2).any():
                    ring = (sm_t == 2).float()
                else:
                    bf = (m > 0.01).float()
                    ring = bf.new_zeros(bf.shape)
                    if bf.sum() > 64.0:
                        bf3 = bf.unsqueeze(0).expand(3, -1, -1)
                        mb = _sar_gray_sobel_magnitude(bf3)
                        sel = bf > 0.5
                        if int(sel.sum().item()) > 16:
                            qv = torch.quantile(
                                mb[sel],
                                float(
                                    getattr(
                                        opt,
                                        "sar_target_gradient_edge_quantile",
                                        0.88,
                                    )
                                ),
                            )
                            ring = ((mb >= qv) & (m > 0.01)).float()
                wmap = m * (1.0 + (eb - 1.0) * ring)
            denom = wmap.sum() + 1e-8
            L_grad_target = (torch.abs(mag_i - mag_g) * wmap).sum() / denom

        if not torch.isfinite(Ll1):
            Ll1 = torch.nan_to_num(Ll1, nan=0.0, posinf=1e4, neginf=0.0)
        gscale = float(getattr(opt, "sar_target_gradient_loss_scale", 1.0))
        loss = (
            (1.0 - lambda_dssim_eff) * Ll1
            + (lambda_dssim_eff * (1.0 - ssim_value) if lambda_dssim_eff > 0.0 else Ll1.new_zeros(()))
            + L_depth_order
            + wg * gscale * L_grad_target
        )
        if (
            use_sar_mode
            and hasattr(viewpoint_cam, "scattering_mask")
            and viewpoint_cam.scattering_mask is not None
        ):
            try:
                from sar.sar_refine_bg_shadow_train import (
                    compute_shadow_over_occlusion_loss,
                    compute_bg_ground_plane_loss,
                    compute_new_point_luma_cap_loss,
                )

                sm_oc = scattering_mask
                L_sh_occ = compute_shadow_over_occlusion_loss(image, gt_image, sm_oc, opt)
                if _loss_term_finite(L_sh_occ):
                    loss = loss + L_sh_occ
                L_bg_plane = compute_bg_ground_plane_loss(gaussians, opt)
                if _loss_term_finite(L_bg_plane):
                    loss = loss + L_bg_plane
                L_new_luma = compute_new_point_luma_cap_loss(gaussians, opt)
                if _loss_term_finite(L_new_luma):
                    loss = loss + L_new_luma
            except ImportError:
                pass
        if not torch.isfinite(loss):
            loss = Ll1 + L_depth_order + wg * gscale * L_grad_target

        t_ou = int(getattr(opt, "sar_target_only_loss_until_iter", 0) or 0)
        if (
            t_ou > 0
            and iteration == t_ou + 1
            and use_sar_mode
            and hasattr(viewpoint_cam, "scattering_mask")
            and viewpoint_cam.scattering_mask is not None
        ):
            print(
                f"[SAR] 分阶段损失：第 {t_ou + 1} 次迭代起切换为加权全图（背景/阴影+SSIM 恢复）"
            )

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            if _loss_term_finite(Ll1depth):
                loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        
        # SAR形状约束（防止点云塌陷成圆饼状）
        Lshape = 0.0
        if (hasattr(opt, 'sar_shape_constraint_weight') and 
            opt.sar_shape_constraint_weight > 0 and
            opt.sar_shape_constraint_start_iter <= iteration <= opt.sar_shape_constraint_end_iter):
            # 计算点云的高度方向方差，防止塌陷
            xyz = gaussians.get_xyz  # (N, 3)
            if xyz.shape[0] > 0:
                # ⚠️ 重要：对SAR坐标系，Z坐标才是高度方向！
                # SAR几何：X-方位向，Y-距离向，Z-高度（垂直）
                z_coords = xyz[:, 2]  # Z坐标（高度方向，SAR几何）
                z_mean = z_coords.mean()
                z_var = ((z_coords - z_mean) ** 2).mean()
                
                # 形状约束：鼓励高度方向的方差（防止塌陷成2D圆饼）
                # 如果方差太小，说明点云在Z方向塌陷了，需要惩罚
                min_z_var = 0.1  # 最小方差阈值（增加以更强约束）
                if z_var < min_z_var:
                    Lshape = opt.sar_shape_constraint_weight * (min_z_var - z_var)
                    if _loss_term_finite(Lshape):
                        loss += Lshape

        # SAR：将「暗 + 较低」的高斯在高度方向压成近似平面（Z 方差惩罚，与 sar_shape 互补）
        L_ground_plane = 0.0
        w_ground = float(getattr(opt, "sar_ground_constraint_weight", 0.0))
        if (
            w_ground > 0
            and getattr(dataset, "sar_mode", False)
            and getattr(opt, "sar_ground_constraint_start_iter", 0) <= iteration
            <= getattr(opt, "sar_ground_constraint_end_iter", 10**9)
        ):
            xyz = gaussians.get_xyz
            n_pts = xyz.shape[0]
            if n_pts > 0:
                z = xyz[:, 2]
                f_dc = gaussians._features_dc[:, 0, :]
                rgb = SH2RGB(f_dc)
                lum = (
                    0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
                ).clamp(0.0, 1.0)
                lum_thr = float(getattr(opt, "sar_background_brightness_threshold", 0.2))
                z_q = float(getattr(opt, "sar_ground_plane_z_quantile", 0.45))
                z_q = min(0.999, max(0.01, z_q))
                z_cut = torch.quantile(z.detach(), z_q)
                bg_mask = (lum < lum_thr) & (z <= z_cut)
                min_bg = int(getattr(opt, "sar_ground_plane_min_points", 32))
                if int(bg_mask.sum().item()) >= min_bg:
                    z_bg = z[bg_mask]
                    L_ground_plane = w_ground * z_bg.var()
                    if _loss_term_finite(L_ground_plane):
                        loss += L_ground_plane

        # SAR：鼓励「暗」高斯在水平面 XY 上占有足够对角跨度（与本仓库 XYZ：Z 为高度一致）；不拉大 Z，缓解部分视角背景塌缩留白/炸点
        L_spread_horizontal = gaussians.get_xyz.new_zeros(())
        w_spread = float(getattr(opt, "sar_spread_constraint_weight", 0.0) or 0.0)
        if (
            w_spread > 0.0
            and getattr(dataset, "sar_mode", False)
            and getattr(opt, "sar_spread_constraint_start_iter", 500) <= iteration
            <= getattr(opt, "sar_spread_constraint_end_iter", 25_000)
        ):
            xyz_s = gaussians.get_xyz
            if xyz_s.shape[0] > 31:
                f_dcs = gaussians._features_dc[:, 0, :]
                rgbs = SH2RGB(f_dcs)
                lumv = (
                    0.299 * rgbs[:, 0]
                    + 0.587 * rgbs[:, 1]
                    + 0.114 * rgbs[:, 2]
                ).clamp(0.0, 1.0)
                lum_thr_s = float(getattr(opt, "sar_background_brightness_threshold", 0.2))
                dim_m = lumv < lum_thr_s
                if int(dim_m.sum().item()) >= 16:
                    xy_d = xyz_s[dim_m, :2]
                    x_span_d = xy_d[:, 0].max() - xy_d[:, 0].min()
                    y_span_d = xy_d[:, 1].max() - xy_d[:, 1].min()
                    diag_d = torch.sqrt(x_span_d * x_span_d + y_span_d * y_span_d + 1e-12)
                    xy_all = xyz_s[:, :2]
                    sx = xy_all[:, 0].max() - xy_all[:, 0].min()
                    sy = xy_all[:, 1].max() - xy_all[:, 1].min()
                    scene_diag = torch.sqrt(sx * sx + sy * sy + 1e-12)
                    min_r = float(getattr(opt, "sar_spread_min_radius", 10.0))
                    frac = float(getattr(opt, "sar_spread_scene_fraction", 0.5))
                    frac = min(0.95, max(0.15, frac))
                    min_diag = torch.maximum(
                        scene_diag * frac,
                        xyz_s.new_tensor(min_r * 1.41421356),
                    )
                    gap = min_diag - diag_d
                    if gap > 0:
                        L_spread_horizontal = w_spread * (gap / (min_diag + 1e-6)) ** 2
                        if _loss_term_finite(L_spread_horizontal):
                            loss += L_spread_horizontal

        # SAR refine：densify 新增高斯 DC 亮度不得达到/超过初始点云亮度上沿（抑制过亮背景/壳层遮目标）
        L_refine_brightness_cap = image.new_zeros(())
        if w_cap > 0.0 and getattr(gaussians, "_refine_brightness_snapshotted", False):
            n0_cap = int(getattr(gaussians, "_refine_brightness_n0", 0))
            cap_m = getattr(gaussians, "_refine_brightness_cap_max", None)
            n_all = int(gaussians.get_xyz.shape[0])
            if cap_m is not None and n_all > n0_cap:
                margin_b = float(getattr(opt, "sar_refine_brightness_cap_margin", 0.02))
                margin_b = max(0.0, margin_b)
                limit = cap_m - margin_b
                med_m = getattr(gaussians, "_refine_brightness_cap_median", None)
                med_extra = float(getattr(opt, "sar_refine_brightness_cap_median_margin", 0.0) or 0.0)
                if med_m is not None and med_extra > 1e-12:
                    limit = torch.minimum(limit, med_m.to(limit.device) - med_extra)
                limit = torch.clamp(limit, min=0.0)
                if isinstance(limit, torch.Tensor) and not torch.isfinite(limit).all():
                    limit = torch.nan_to_num(limit, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
                luma_all = gaussians.get_dc_luminosity()
                ln_new = luma_all[n0_cap:]
                L_refine_brightness_cap = (torch.relu(ln_new - limit) ** 2).mean()
                if _loss_term_finite(L_refine_brightness_cap):
                    loss += w_cap * L_refine_brightness_cap

        n_gs = gaussians.get_xyz.shape[0]
        if n_gs == 0:
            raise RuntimeError(
                "高斯点数为 0，无法执行 loss.backward()。"
                "通常由 densify/prune 与 min_opacity/大屏点规则把所有点剪掉引起；"
                "已在 GaussianModel.densify_and_prune 中保留至少若干不透明点。"
                "若仍出现请检查 opacity_reset、min_opacity 或改用 --optimizer_type default。"
            )
        fr = gaussians._features_rest
        if fr.numel() > 0 and fr.shape[-2] != (gaussians.max_sh_degree + 1) ** 2 - 1:
            raise RuntimeError(
                f"f_rest 形状 {tuple(fr.shape)} 与 max_sh_degree={gaussians.max_sh_degree} 不匹配。"
            )

        # 插值方位软监督：相邻训练角之间 colmap_track 位姿 + 双邻域加权 L1
        global _SAR_INTERP_SOFT_LOGGED
        try:
            from sar.sar_interp_soft_supervision import (
                should_run_interp_soft,
                compute_interp_soft_supervision_loss,
            )

            if should_run_interp_soft(iteration, opt, dataset):
                L_soft, soft_info = compute_interp_soft_supervision_loss(
                    train_cameras=train_cameras_list,
                    source_path=dataset.source_path,
                    gaussians=gaussians,
                    pipe=pipe,
                    background=bg,
                    opt=opt,
                    dataset=dataset,
                    render_fn=render,
                    separate_sh=SPARSE_ADAM_AVAILABLE,
                    iteration=iteration,
                )
                if _loss_term_finite(L_soft) and float(L_soft.detach().item()) > 0:
                    loss = loss + L_soft
                if not _SAR_INTERP_SOFT_LOGGED:
                    _SAR_INTERP_SOFT_LOGGED = True
                    print(
                        f"[SAR] 插值方位软监督已启用: every={opt.sar_interp_soft_every}, "
                        f"weight={opt.sar_interp_soft_weight}, t={opt.sar_interp_soft_t_values}"
                    )
                if iteration % max(1, int(opt.sar_interp_soft_every)) == 0 and soft_info.get("loss_item"):
                    print(
                        f"[SAR interp soft] iter={iteration} "
                        f"pair={soft_info.get('a0')}°–{soft_info.get('a1')}° "
                        f"L={soft_info.get('loss_item'):.6f} "
                        f"{'; '.join(soft_info.get('notes', [])[:2])}"
                    )
        except ImportError as e:
            if not _SAR_INTERP_SOFT_LOGGED:
                _SAR_INTERP_SOFT_LOGGED = True
                print(f"[SAR] 插值软监督跳过（导入失败）: {e}")

        if not _loss_scalar_isfinite(loss):
            global _SAR_NAN_DIAG_ONCE
            if not _SAR_NAN_DIAG_ONCE:
                _SAR_NAN_DIAG_ONCE = True
                img_ok = _safe_cuda_isfinite(image)
                print(
                    f"\n[SAR NaN 诊断 iter {iteration}] Ll1={float(Ll1.detach()) if _loss_term_finite(Ll1) else Ll1} "
                    f"Lgrad={float(L_grad_target.detach()) if _loss_term_finite(L_grad_target) else L_grad_target} "
                    f"image_finite={img_ok} N_gs={gaussians.get_xyz.shape[0]}"
                )
            nf = gaussians.sanitize_nonfinite_parameters()
            if nf > 0:
                print(f"[SAR iter {iteration}] loss 非有限，已修复 {nf} 个参数元素并跳过 backward")
            else:
                print(f"\n[WARN iter {iteration}] 非有限 loss={loss.detach().item()}，跳过 backward/optimizer")
            gaussians.optimizer.zero_grad(set_to_none=True)
            gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        else:
            try:
                loss.backward()
            except RuntimeError as e:
                err = str(e).lower()
                if use_sar_mode and ("cuda" in err or "accelerator" in err or "illegal memory" in err):
                    from sar.rasterizer import sar_mark_cuda_unstable
                    sar_mark_cuda_unstable(f"iter {iteration} backward: {e}")
                    _zero_nonfinite_gradients(gaussians)
                    gaussians.sanitize_nonfinite_parameters(reset_optimizer=True)
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    _SAR_CUDA_ABORT = True
                    print(
                        f"\n[SAR] iter {iteration} backward CUDA 异常，训练中止: {e}\n"
                        f"  默认应走 Python SDGR（勿加 --sar_sdgr_cuda）。若仍失败：\n"
                        f"    set SAR_SDGR_FORCE_PYTHON=1\n"
                        f"  新开 output 目录重跑，勿续训已损坏 checkpoint。"
                    )
                    break
                raise
            try:
                from sar.sar_refine_bg_shadow_train import zero_frozen_target_gradients

                zero_frozen_target_gradients(gaussians, opt)
            except ImportError:
                pass
            if use_sar_mode:
                _clip_gaussian_gradients(
                    gaussians, float(getattr(opt, "sar_max_grad_norm", 2.0)),
                )
                from sar.rasterizer import sar_cuda_sync_check
                if not sar_cuda_sync_check(f"iter {iteration} backward", fatal=False):
                    _zero_nonfinite_gradients(gaussians)
                    gaussians.sanitize_nonfinite_parameters(reset_optimizer=True)
                    _SAR_CUDA_ABORT = True
                    print(
                        f"\n[SAR] iter {iteration} backward CUDA 异常，训练中止。"
                        f"请新开 output 并用 SAR_SDGR_FORCE_PYTHON=1 重跑。"
                    )
                    break

        if _SAR_CUDA_ABORT:
            break

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if _loss_scalar_isfinite(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration == 1 or iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
            progress_bar.update(1)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                try:
                    if use_sar_mode:
                        from sar.rasterizer import sar_cuda_sync_check
                        if not sar_cuda_sync_check(f"iter {iteration} pre-densify", fatal=False):
                            _SAR_CUDA_ABORT = True
                            print(f"\n[SAR] iter {iteration} densify 前 CUDA 异常，训练中止。")
                            break
                    _apply_densify_visibility_stats(
                        gaussians, viewspace_point_tensor, visibility_filter, radii,
                    )
                    try:
                        from sar.sar_refine_bg_shadow_train import mask_frozen_target_densify_accum

                        mask_frozen_target_densify_accum(gaussians, opt)
                    except ImportError:
                        pass
                except RuntimeError as e:
                    print(f"[SAR densify] iter {iteration} 跳过 visibility 统计: {e}")
                    _SAR_CUDA_ABORT = True
                    print(
                        "\n[SAR] CUDA 上下文已损坏，训练中止。"
                        "请新开 output 目录并用 SAR_SDGR_FORCE_PYTHON=1 重跑。"
                    )
                    break

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gm = _sar_dc_luminance_densify_grad_mult(
                        gaussians, opt, dataset, sar_dense_tier_log_holder
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.01,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        grad_thresh_mult=gm,
                    )
                    if not bool(getattr(opt, "sar_refine_disable_clamp_scaling", False)):
                        gaussians.clamp_scaling()
                    _sar_apply_dense_shadow_scaling_log_cap(gaussians, opt, dataset)
                    try:
                        from sar.sar_refine_bg_shadow_train import clamp_new_point_luma_inplace

                        clamp_new_point_luma_inplace(gaussians, opt)
                    except ImportError:
                        pass

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if _SAR_CUDA_ABORT:
                break

            # Optimizer step
            if iteration < opt.iterations and _loss_scalar_isfinite(loss):
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    if use_sar_mode and visibility_filter.numel() > 0:
                        visible = torch.zeros(
                            gaussians.get_xyz.shape[0], dtype=torch.bool, device="cuda",
                        )
                        vf = _normalize_gaussian_index(
                            visibility_filter, gaussians.get_xyz.shape[0],
                        )
                        if vf.numel() > 0:
                            visible[vf] = True
                    else:
                        visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    if _SAR_CUDA_ABORT:
        return {"ok": False, "last_iteration": _last_iter, "reason": "cuda"}
    return {"ok": True, "last_iteration": opt.iterations}

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    cfg_dump = vars(args).copy()
    if cfg_dump.get("scattering_masks") is not None:
        cfg_dump["scattering_masks"] = None
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**cfg_dump)))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        if torch.isfinite(Ll1):
            tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        if torch.isfinite(loss):
            tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            op_hist = scene.gaussians.get_opacity.detach().reshape(-1)
            op_finite = op_hist[torch.isfinite(op_hist)]
            if op_finite.numel() > 0:
                tb_writer.add_histogram("scene/opacity_histogram", op_finite, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # SAR配置参数
    parser.add_argument("--use_default_sar_config", action="store_true", default=True,
                       help="使用train.py中的SAR默认配置（MSTAR参数）")
    parser.add_argument("--no_default_sar_config", dest="use_default_sar_config", 
                       action="store_false",
                       help="禁用默认SAR配置，使用命令行参数或arguments/__init__.py中的默认值")
    parser.add_argument("--show_sar_config", action="store_true",
                       help="显示当前SAR配置并退出")
    parser.add_argument(
        "--sar_detail_focus",
        action="store_true",
        default=False,
        help="细节优先：弱化形状/尺寸盒约束、加强 densify、放宽目标 scaling、加快 SH 升阶（参见 apply_sar_detail_focus_overrides）",
    )

    args = parser.parse_args(sys.argv[1:])
    
    # 如果只是查看配置，打印后退出
    if args.show_sar_config:
        print_sar_config()
        sys.exit(0)
    
    # 应用SAR默认配置（如果启用）
    if args.use_default_sar_config:
        print("\n🎯 应用SAR默认配置（MSTAR数据集参数）")
        print("   （可通过 --no_default_sar_config 禁用）")
        args = apply_sar_config_to_args(args, sys.argv)
        print_sar_config()
    else:
        print("\n⚠️  使用命令行参数或默认配置（未应用MSTAR预设）")

    if getattr(args, "sar_detail_focus", False):
        apply_sar_detail_focus_overrides(args, sys.argv)
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
