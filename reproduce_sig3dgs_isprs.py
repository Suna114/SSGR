#!/usr/bin/env python3
# Copyright: usage script for local SAR-GS; original 3DGS copyright see LICENSE.md
"""
ISPRS 2026 · SIG-3DGS 论文对齐训练入口
====================================
Liu et al., "A differentiable method for novel view SAR image generation via
3D Gaussian Splatting", ISPRS Journal of Photogrammetry and Remote Sensing.

论文核心：已知多视角 SAR 与视角 (θ, φ) → MPA 成像几何下的可微 SAR 渲染 →
L1 + (1−SSIM) 加权重建损失（文中式 (22)，本仓库对应为 ``(1-λ)*L1 + λ*(1-SSIM)``，即 ``lambda_dssim``）→
反传优化 3D 高斯与球谐散射系数；**正文不依赖经典多视 SfM/COLMAP**。

本脚本在本仓库中的角色
------------------------
* 无 SfM 点云与颜色：与 ``train_sar_complete_nosfm.py`` 相同，均走 ``scene/dataset_readers.readMstarNoSfMSceneInfo``——
  ``mstar_nosfm_init/points3D.ply`` 不存在时以 ``np.random.seed(0)`` 生成随机点坐标，颜色由随机 SH 系数经 ``SH2RGB`` 写入 PLY
  （数值上接近中性灰、各点差异很小）。训练后顶点色由优化后的球谐决定。
* ``mstar_nosfm`` 的设定方式与 ``train_sar_complete_nosfm.py`` 一致（默认 False 时自动打开）；外参不读 ``sparse/0``。
* 可选 ``--paper_defaults``：位置学习率取文中 §4.2 给出的 **1.6e-4**，并使 ``position_lr_max_steps`` 与总迭代一致。
* 可选 ``--minimal_sar_extras``：关掉散射掩码加权并削弱工程向正则，使总损失更接近「纯重建项」（仍走 SAR 渲染路径）。

实现仍由 ``train.py::training``、``gaussian_renderer``、``scene`` 等模块完成；**并非**单文件重写整篇论文代码。

用法示例
--------
::

    python reproduce_sig3dgs_isprs.py -s data/MSTAR_subset -m output/sig3dgs_paperish \\
        --paper_defaults --iterations 30000

    # 更接近「仅 L1 + 结构项」、少工程正则（按需）
    python reproduce_sig3dgs_isprs.py -s data/MSTAR_subset -m output/sig3dgs_min \\
        --paper_defaults --minimal_sar_extras

``--`` 之后的参数会原样传给 ``train_sar_complete.parse_sar_arguments``（与 ``train_sar_complete.py`` 一致）。

注意：仿真数据集（如文中 OptiXSAR）需自行准备；本脚本不替代数据生成。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_diff_gaussian_rasterization() -> None:
    """
    train → gaussian_renderer 依赖 diff_gaussian_rasterization._C（CUDA 扩展）。
    未编译时 PyTorch 常报「cannot import name '_C' / partially initialized module」。
    """
    root = _repo_root()
    sub = root / "submodules" / "diff-gaussian-rasterization"
    if sub.is_dir():
        p = str(sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError as e:
        sub_pc = sub / "diff_gaussian_rasterization"
        print(
            "\n[错误] 未能加载 diff_gaussian_rasterization（缺少已编译扩展 _C）。\n"
            "请在已安装 PyTorch 的同一环境中编译标准 3D-GS 光栅化子模块：\n\n"
            f'  cd /d "{sub}"\n'
            "  pip install -e . --no-build-isolation\n\n"
            "若不用 --no-build-isolation，隔离构建环境可能缺少 torch 导致 setup 失败。\n"
            "另需：与 PyTorch 匹配的 CUDA Toolkit、Windows 上 Visual Studio C++ 生成工具。\n"
            "SAR 专用扩展（diff-sar-rasterization）需单独编译，参见 batch\\安装SAR_CUDA扩展.bat。\n\n"
            f"期望源码目录存在: {sub_pc}\n"
            f"原始 ImportError: {e}\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e



def _forward_help() -> None:
    print("--- train_sar_complete 全部参数（与 train_sar_complete.py 一致） ---\n")
    sys.argv = [
        sys.argv[0],
        "-s",
        "__path_placeholder__",
        "-m",
        "__path_placeholder__",
        "--help",
    ]
    from train_sar_complete import parse_sar_arguments

    try:
        parse_sar_arguments()
    except SystemExit:
        pass


def main() -> None:
    outer = argparse.ArgumentParser(
        description="ISPRS 2026 SIG-3DGS：论文设定下的训练入口（无 SfM）",
        add_help=False,
    )
    outer.add_argument("-s", "--source_path", type=str, default=None, help="数据集根目录（含 images/）")
    outer.add_argument("-m", "--model_path", type=str, default=None, help="训练输出目录")
    outer.add_argument(
        "--paper_defaults",
        action="store_true",
        help="§4.2：position_lr_init=1.6e-4，position_lr_max_steps=iterations",
    )
    outer.add_argument(
        "--minimal_sar_extras",
        action="store_true",
        help="关闭散射掩码加权并置零若干 SAR 正则（更接近论文主损失形式）",
    )
    outer.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="本说明 + train_sar_complete 的完整 --help",
    )

    pre, rest = outer.parse_known_args()

    if pre.help:
        print(__doc__)
        _ensure_diff_gaussian_rasterization()
        _forward_help()
        return

    if not pre.source_path or not pre.model_path:
        outer.print_help()
        print("\n错误: 必须提供 -s 与 -m。\n")
        raise SystemExit(2)

    sys.argv = [sys.argv[0], "-s", pre.source_path, "-m", pre.model_path] + rest

    _ensure_diff_gaussian_rasterization()

    import torch

    from train_sar_complete import parse_sar_arguments, preprocess_sar_dataset
    from train import training
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from argparse import ArgumentParser
    from utils.general_utils import safe_state

    args = parse_sar_arguments()
    # 与 train_sar_complete_nosfm.py 相同：仅在未显式打开时启用无 SfM 路径（保证点云/初值与之一致）
    if not getattr(args, "mstar_nosfm", False):
        args.mstar_nosfm = True

    if pre.paper_defaults:
        args.position_lr_init = 1.6e-4
        it = int(getattr(args, "iterations", 30_000))
        args.position_lr_max_steps = it

    if pre.minimal_sar_extras:
        args.filter_by_scattering = False
        zero_pairs = [
            ("sar_shape_constraint_weight", 0.0),
            ("sar_ground_constraint_weight", 0.0),
            ("sar_spread_constraint_weight", 0.0),
            ("sar_target_height_constraint_weight", 0.0),
            ("sar_target_dimension_constraint_weight", 0.0),
            ("sar_scale_constraint_weight", 0.0),
            ("sar_height_constraint_weight", 0.0),
            ("sar_shadow_constraint_weight", 0.0),
            ("sar_multiview_consistency_weight", 0.0),
        ]
        for name, val in zero_pairs:
            if hasattr(args, name):
                setattr(args, name, val)

    print("\n" + "=" * 72)
    print(f"reproduce_sig3dgs_isprs · 论文设定 / mstar_nosfm={getattr(args, 'mstar_nosfm', False)}")
    print("=" * 72)
    print(f"  -s  {args.source_path}")
    print(f"  -m  {args.model_path}")
    print(f"  iterations              {getattr(args, 'iterations', '?')}")
    print(f"  position_lr_init        {getattr(args, 'position_lr_init', '?')}")
    print(f"  position_lr_max_steps   {getattr(args, 'position_lr_max_steps', '?')}")
    print(f"  lambda_dssim (≈论文 β) {getattr(args, 'lambda_dssim', '?')}")
    print(f"  paper_defaults          {pre.paper_defaults}")
    print(f"  minimal_sar_extras      {pre.minimal_sar_extras}")
    print(f"  filter_by_scattering    {getattr(args, 'filter_by_scattering', '?')}")
    print("=" * 72 + "\n")

    scattering_masks = preprocess_sar_dataset(args)
    args.scattering_masks = scattering_masks

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model_params = ModelParams(ArgumentParser(), sentinel=True)
    opt_params = OptimizationParams(ArgumentParser())
    pipe_params = PipelineParams(ArgumentParser())

    dataset_params = model_params.extract(args)
    if getattr(args, "scattering_masks", None) is not None:
        dataset_params.scattering_masks = args.scattering_masks

    training(
        dataset_params,
        opt_params.extract(args),
        pipe_params.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    print("\n训练结束。输出目录:", args.model_path)


if __name__ == "__main__":
    main()
