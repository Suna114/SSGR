#!/usr/bin/env python3
"""
完整的SAR-GS训练脚本
集成了SDGR (SAR Differentiable Gaussian Splatting Rasterizer)

基于论文: SAR-GS: 3D Gaussian Splatting for Synthetic Aperture Radar Target Reconstruction

使用方法:
    python train_sar_complete.py -s <数据集路径> -m <输出路径>

默认训练配方对齐当前最优实验 exp15-5（sar_refine_target_bg + 方位分层 + 分区损失权重与梯度项等；
关闭某项请加文档中的 --no_*）。精简示例:

    python train_sar_complete.py -s data/9 -m output/exp_run

等价于原先一条长命令中含 --sar_refine_target_bg --sar_refine_light_densify --sar_stratify_azimuth
--sar_azimuth_bins 24 --auto_render_train_views --auto_novel_render 及各损失权重。

示例（旧式显式传参仍可用）:
    python train_sar_complete.py -s data/MSTAR_BMP2 -m output/sar_bmp2

参数说明:
    --use_sar_mode: 启用SAR模式（默认True）
    --use_sar_rendering: 使用SDGR渲染器（默认True）
    --filter_by_scattering: 使用散射强度掩码过滤点云和损失（默认True）
    默认使用与 convert 相同的语义分割（默认 0/1/2/3：背景、目标内部、目标边缘、阴影）；--merge_scattering_target_edge 与 convert 一致可改为 0/1/3；--no_semantic_segmentation 可改为简易三类
    语义开启时默认在散射掩码目录下保存 region_classification/（与 convert 同风格的叠加图）；--no_save_semantic_visualization 可关闭
    --shadow_threshold_factor: 阴影阈值因子（与 feature_extraction 一致）
    --target_mask_dilate_px: 目标二值掩码 3×3 膨胀迭代次数（与 convert 一致，默认 3；0 关闭）
    --merge_scattering_target_edge: 与 convert 一致，将边缘 (2) 并入 (1)，掩码仅三类
    --reuse_convert_scattering_masks: 默认复用 convert 写入的 scattering_masks/from_convert/（与 SfM 一致）
    --no-reuse_convert_scattering_masks: 强制按当前 CLI 参数重算掩码
    --shadow_loss_weight / --sar_target_depth_order_weight: 阴影损失权重与深度顺序正则；
      refine 预设默认开启「阴影先于目标」：--sar_shadow_depth_before_target（或关闭：--no_sar_shadow_depth_before_target）。
      依赖渲染深度符号；若层次感反了请关掉该项或改为旧语义。
    --sar_spread_constraint_weight：铺开约束会把暗高斯的 XY 跨度拉大（不是增多点数）；refine 预设已改为默认 0，
      需要时再手动调高。
    分阶段损失: --sar_target_only_loss_until_iter N（前 N 迭代仅拟合目标+边缘像素 L1，全图 SSIM 暂停；之后恢复加权背景/阴影）
    SAR 目标更清晰（GT 略糊、边缘仍可分）：--edge_loss_weight/--target_loss_weight 仅抬 L1；
      另可 --sar_target_gradient_l1_weight + --sar_target_gradient_edge_boost（如对轮廓加强×3）；无掩码 2 时用 Sobel 分位自动生成轮廓带（--sar_target_gradient_edge_quantile）。

    固定目标点云 + 背景/阴影精调（OptimizationParams 默认开启 exp15-5 对齐权重）:
      --sar_refine_target_bg / --no_sar_refine_target_bg：启用或关闭 sar/sar_refine_target_bg.py 预设（源码在同目录 sar/）；
        densify 新增点带 DC 亮度帽（见 sar_refine_brightness_cap_*；完全关闭亮度帽加 --sar_refine_brightness_cap_disable）。
      --sar_refine_uniform_loss：散射掩码分区 L1 均等权重（对比试验用，见 train.py sar_uniform_region_loss）
      --sar_refine_light_densify / --no_sar_refine_light_densify：允许少量 densify（默认开启；已注册在 OptimizationParams，train 可见）
      --sar_dense_by_dc_luminance：按 DC 亮度分两档调整 densify（中暗背景更易增生、最暗阴影更难；可选 --sar_dense_shadow_scaling_log_cap）
      --no_sar_refine_seed_bg_shadow_plates：关闭在目标外包络外自动铺设背景/遮光片状高斯（默认开启；可与轻量 densify 同时使用）
      --sar_refine_seed_placement_mode camera_ray | world_axis：camera_ray 用「均值训练相机中心→目标质心」的 ω；阴影 centroid−above_frac·diag·ω，背景 centroid+below_frac·diag·ω；压低 --sar_refine_seed_above_extent_frac 可避免阴影薄片过近均值相机；world_axis 为旧世界轴铺板
      --no_sar_refine_eval_suggestions：训练结束不在 model_path 写入 refine_eval_suggestions.txt

    视角均权（OptimizationParams 默认开启）：--sar_stratify_azimuth / --no_sar_stratify_azimuth，
      --sar_azimuth_bins N（未与 --sar_equal_view_sampling 同时手写时：默认自动取训练相机数为桶数；
        亦可显式指定 N；设 0 表示始终按训练相机数分桶）。
      --sar_equal_view_sampling 等价于打开方位分层，且在未指定 --sar_azimuth_bins 时桶数=训练视角数。

    插值方位软监督（缓解新视角像训练角）：--sar_interp_soft_supervision，
      --sar_interp_soft_every 500 --sar_interp_soft_weight 0.2；
      双邻域 L1（非旋转伪 GT），位姿与 render_novel colmap_track 一致。
    阴影几何/汇总（convert）：--shadow_geometry_azimuth_mode oblique | cardinal | mstar_axis_split | any；
      --target_dimension_summary_azimuth_mode oblique | near_zero | cardinal | mstar_axis_split | diagonal（默认与斜视几何一致）；
      diagonal=45/135/225/315°±容差；对应长方体脚本 --post_sfm_cardinal_box_azimuth_lane diagonal。

    训练结束后自动渲染（默认开启；可用 --no_* 跳过）:
      --auto_render_train_views / --no_auto_render_train_views：调用 scripts/rendering/render_and_save.py，
        仅渲染 **训练相机真实外参**（与 novel 网格不是同一回事，用于评判「训练角」拟合）。
      --auto_render_train_views_dir：上述输出根目录（默认 model_path/post_train_train_views）。
      --auto_novel_render / --no_auto_novel_render：调用 **scripts/rendering/render_novel_sar_views.py**
        （ SAR 俯角×方位网格，外参按该脚本生成/插值）。
      --auto_novel_render_script：覆盖 novel 脚本路径（默认即 render_novel_sar_views.py）。
      --auto_novel_azimuth_step / --auto_novel_azimuth_start / --auto_novel_azimuth_end
      --auto_novel_output_dir、--no_auto_novel_unique_folder（关闭则落回脚本默认 novel_views_grid，易覆盖）、
      --auto_novel_pose_mode、--auto_novel_quiet

    判别「训练角度」画质请以 --auto_render_train_views 或直接运行 render_and_save.py --render_train 为准；
    novel_views 中与训练角接近的格子仍是网格位姿，不等价于数据集里的相机条目。
"""

import os
import subprocess
import sys
import time
from typing import Optional
import torch
import argparse
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training, prepare_output_and_logger, apply_sar_detail_focus_overrides
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import numpy as np

# 导入SAR相关模块
try:
    from sar_scattering_filter import (
        compute_scattering_masks_for_dataset,
        visualize_scattering_mask,
        load_sar_grayscale_like_convert,
    )
    SCATTERING_FILTER_AVAILABLE = True
except ImportError as e:
    print(f"警告: 散射强度过滤模块不可用 ({e})")
    SCATTERING_FILTER_AVAILABLE = False


def parse_sar_arguments():
    """
    解析SAR-GS专用命令行参数
    
    注意：大部分SAR参数已在arguments/__init__.py的ModelParams中定义
    这里只添加训练脚本特有的参数
    """
    parser = ArgumentParser(
        description="SAR-GS Training Script",
        epilog="提示：--novel_pose_mode、--colmap_track_azimuth_space、--novel_pose_debug 等属于 "
        "scripts/rendering/render_novel_sar_views.py，训练时不要传入。",
    )
    
    # 基础参数（这些类会自动添加它们的参数到parser）
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # SAR训练专用参数（仅添加未在ModelParams/OptimizationParams中定义的参数）
    parser.set_defaults(filter_by_scattering=True, auto_novel_unique_folder=True)
    parser.add_argument('--filter_by_scattering', dest='filter_by_scattering', action='store_true',
                       help='使用散射强度掩码过滤点云和损失')
    parser.add_argument('--no_filter_by_scattering', dest='filter_by_scattering', action='store_false',
                       help='关闭散射强度掩码（预处理与加权损失跳过；等价于仅用全图像素训练）')
    
    from sar.semantic_mask_config import (
        register_semantic_mask_cli_args,
        register_scattering_mask_cache_cli_args,
    )
    register_semantic_mask_cli_args(parser)
    register_scattering_mask_cache_cli_args(parser)

    parser.add_argument(
        '--sar_detail_focus',
        action='store_true',
        default=False,
        help='细节优先预案：弱化盒约束、加强 densify、放宽目标 scaling、加快 SH 升阶（见 train.apply_sar_detail_focus_overrides）',
    )
    parser.add_argument(
        '--no_semantic_segmentation',
        action='store_true',
        default=False,
        help='禁用与 convert 一致的语义掩码(默认 0/1/2/3)，改用简易三类散射掩码',
    )
    parser.add_argument(
        '--reuse_convert_scattering_masks',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='优先复用 convert 写入的 scattering_masks/from_convert/<批次>/（默认开启；批次见 --scattering_masks_run）',
    )
    parser.add_argument(
        '--scattering_masks_dir',
        type=str,
        default=None,
        help='显式指定掩码目录（含 *_mask.npy）；设后跳过 convert 复用与重算',
    )
    parser.add_argument(
        '--sar_equal_view_sampling',
        action='store_true',
        default=False,
        help='训练时按方位角分桶均衡抽样（等价 --sar_stratify_azimuth）；'
        '若未写 --sar_azimuth_bins，则桶数自动取训练相机数量（内部 sar_azimuth_bins=0）',
    )
    parser.add_argument(
        '--sar_refine_uniform_loss',
        action='store_true',
        default=False,
        help='与 refine 配合：分区 L1 均等权重（等价于全图同权；需散射掩码仍建议保持 --filter_by_scattering）',
    )
    parser.add_argument(
        '--no_sar_refine_eval_suggestions',
        action='store_true',
        default=False,
        help='refine 模式下训练结束不写入 refine_eval_suggestions.txt',
    )
    parser.add_argument(
        '--auto_render_train_views',
        dest='auto_render_train_views',
        action='store_true',
        default=True,
        help='训练成功后运行 render_and_save.py：仅渲染训练集真实相机（默认开启）',
    )
    parser.add_argument(
        '--no_auto_render_train_views',
        dest='auto_render_train_views',
        action='store_false',
        help='训练结束后不自动渲染训练视角',
    )
    parser.add_argument(
        '--auto_render_train_views_dir',
        type=str,
        default=None,
        help='训练视角渲染输出根目录（默认写入 model_path/post_train_train_views）',
    )
    parser.add_argument(
        '--auto_novel_render',
        dest='auto_novel_render',
        action='store_true',
        default=True,
        help='训练成功后自动运行 scripts/rendering/render_novel_sar_views.py（默认开启）',
    )
    parser.add_argument(
        '--no_auto_novel_render',
        dest='auto_novel_render',
        action='store_false',
        help='训练结束后不自动运行新视角网格渲染',
    )
    parser.add_argument(
        '--auto_novel_render_script',
        type=str,
        default=None,
        help='覆盖默认 novel 渲染脚本路径（默认为仓库内 scripts/rendering/render_novel_sar_views.py）',
    )
    parser.add_argument(
        '--auto_novel_azimuth_step',
        type=int,
        default=15,
        help='自动新视角渲染的方位角步长（度）；与 15° 训练间隔对齐时整圈约 24 张；要更密可改为 1',
    )
    parser.add_argument(
        '--auto_novel_azimuth_start',
        type=int,
        default=0,
        help='自动新视角渲染方位角起始（度，含）',
    )
    parser.add_argument(
        '--auto_novel_azimuth_end',
        type=int,
        default=360,
        help='自动新视角渲染方位角结束（度，不含），默认 360 即整周',
    )
    parser.add_argument(
        '--auto_novel_output_dir',
        type=str,
        default=None,
        help='手动指定新视角渲染根目录。未指定且默认启用唯一文件夹时，自动写入 <model_path>/novel_views_runs/run_<时间戳]',
    )
    parser.add_argument(
        '--no_auto_novel_unique_folder',
        dest='auto_novel_unique_folder',
        action='store_false',
        help='不创建时间戳子目录，交由 render_novel_sar_views 默认写入 novel_views_grid（易与上次渲染互相覆盖）',
    )
    parser.add_argument(
        '--auto_novel_same_depression_only',
        action='store_true',
        default=False,
        help='自动渲染只使用训练俯视角、仅扫方位（更快，见 render_novel_sar_views.py --same_depression_only）',
    )
    parser.add_argument(
        '--auto_novel_depression_angles',
        type=str,
        default=None,
        help='自动渲染俯角列表（空格分隔字符串），如 "17" 或 "15 17 25 45"；不设则用渲染脚本默认',
    )
    parser.add_argument(
        '--auto_novel_pose_mode',
        type=str,
        default=None,
        choices=['colmap_track', 'colmap_ring', 'sar'],
        help='若指定则传入渲染脚本的 --novel_pose_mode',
    )
    parser.add_argument(
        '--auto_novel_quiet',
        action='store_true',
        default=False,
        help='自动渲染时传入 --quiet',
    )
    parser.add_argument(
        '--no_save_semantic_visualization',
        action='store_true',
        default=False,
        help='不导出语义叠加图（与 convert 相同风格的 region_classification_*.png）',
    )
    # background_loss_weight / target_loss_weight / edge_loss_weight / shadow_loss_weight /
    # sar_target_depth_order_* 已在 OptimizationParams 中定义
    
    # ===================================================================
    # 注意：以下参数已在arguments/__init__.py中定义，无需在这里重复：
    # 
    # ModelParams中已有：
    # - sar_mode, sar_camera_height, sar_platform_velocity, sar_prf
    # - sar_bandwidth, sar_image_size_azimuth, sar_image_size_range
    # - sar_camera_distribution_mode, sar_depression_angle, sar_radius_scale
    # 
    # OptimizationParams中已有：
    # - sar_shape_constraint_weight, sar_shape_constraint_start_iter
    # - sar_shape_constraint_end_iter
    # - sar_ground_constraint_weight / _start_iter / _end_iter：背景压平面（Z 方差）
    # - sar_background_brightness_threshold, sar_ground_plane_z_quantile, sar_ground_plane_min_points
    # - sar_stratify_azimuth, sar_azimuth_bins（方位分层训练，不偏好特定角度区间）
    # - background_loss_weight, target_loss_weight, edge_loss_weight（edge 与 target 在 train 中已合并为同一权重）, shadow_loss_weight
    # - sar_target_depth_order_weight, sar_target_depth_order_margin（目标深度先于背景/阴影）
    # - sh_degree_interval（球谐升阶间隔，表示能力）
    # 
    # PipelineParams中已有：
    # - sar_rendering
    # ===================================================================
    
    # 训练参数
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, 
                       default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, 
                       default=[7_000, 15_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])

    argv_tail = sys.argv[1:]
    _has_ld = "--sar_refine_light_densify" in argv_tail
    _has_nld = "--no_sar_refine_light_densify" in argv_tail
    if _has_ld and _has_nld:
        print(
            "\n⚠️  命令行同时写了 --sar_refine_light_densify 与 --no_sar_refine_light_densify；"
            "二者共享同一参数，后以 **解析顺序中最后出现的** 为准（当前 sar_refine_light_densify="
            f"{getattr(args, 'sar_refine_light_densify', '?')}）。\n"
            "   若最终为关闭 densify（False），且初值里背景高斯极少（常见于 mesh replace 后只有车体点云），"
            "现已默认在允许时自动在目标 AABB 外侧铺片状背景/遮光板（轻量 densify 下在满足条件时也会铺；详见 sar_seed_refine_bg_shadow.py；可用 --no_sar_refine_seed_bg_shadow_plates 关闭）；"
            "仍可保留轻量 densify、或在 convert / 初始 PLY 中补充地平/布景点。\n"
        )

    def _argv_specifies_sar_azimuth_bins(argv):
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--sar_azimuth_bins" or a.startswith("--sar_azimuth_bins="):
                return True
            i += 1
        return False

    if getattr(args, "sar_equal_view_sampling", False):
        setattr(args, "sar_stratify_azimuth", True)
        # 均权应与数据集视角数一致：未显式指定桶数时用 0，train.py 中解释为 len(训练相机)
        if not _argv_specifies_sar_azimuth_bins(sys.argv[1:]):
            setattr(args, "sar_azimuth_bins", 0)

    # 默认方位分层时桶数 = 训练相机数（与图像数量一致），除非用户显式 --sar_azimuth_bins
    if (
        getattr(args, "sar_stratify_azimuth", False)
        and not _argv_specifies_sar_azimuth_bins(sys.argv[1:])
    ):
        setattr(args, "sar_azimuth_bins", 0)

    refin = bool(getattr(args, "sar_refine_target_bg", False))
    detail = bool(getattr(args, "sar_detail_focus", False))
    if refin and detail:
        print(
            "\n[提示] 同时指定了 --sar_refine_target_bg 与 --sar_detail_focus："
            "二者目标相反，已采用 refine 预设并忽略 sar_detail_focus。\n"
        )
    elif refin:
        try:
            from sar.sar_refine_target_bg import apply_sar_refine_target_bg_preset

            apply_sar_refine_target_bg_preset(args, sys.argv[1:])
        except ImportError as e:
            print(f"无法导入 sar.sar_refine_target_bg: {e}")
            raise SystemExit(1) from e
    elif detail:
        apply_sar_detail_focus_overrides(args, sys.argv)
    args.save_iterations.append(args.iterations)
    
    return args


def _resolve_novel_render_script(repo: str, override: Optional[str]) -> str:
    if override is not None and str(override).strip():
        p = str(override).strip().strip('"')
        return os.path.normpath(p if os.path.isabs(p) else os.path.join(repo, p))
    return os.path.normpath(os.path.join(repo, "scripts", "rendering", "render_novel_sar_views.py"))


def run_auto_render_train_views_after_training(args) -> int:
    """
    调用 render_and_save.py，仅渲染 Scene 中的训练相机（与 GT 同目录结构便于 compare_render_gt）。
    """
    repo = os.path.dirname(os.path.abspath(__file__))
    script = os.path.normpath(os.path.join(repo, "scripts", "rendering", "render_and_save.py"))
    if not os.path.isfile(script):
        print(f"[auto_render_train_views] 未找到脚本: {script}")
        return 1
    mp = os.path.abspath(args.model_path)
    it = int(getattr(args, "iterations", 0))
    out_root = getattr(args, "auto_render_train_views_dir", None)
    if not out_root:
        out_root = os.path.join(mp, "post_train_train_views")
    cmd = [
        sys.executable,
        script,
        "-m",
        mp,
        "--iteration",
        str(it),
        "--render_train",
        "--no_render_test",
        "--output_dir",
        os.path.abspath(out_root),
    ]
    print("\n" + "=" * 70)
    print("[auto_render_train_views] 真实训练相机位姿渲染（render_and_save.py）")
    print(f"脚本: {script}")
    print("命令:", " ".join(cmd))

    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        p = subprocess.run(cmd, cwd=repo)
        return int(p.returncode)
    except OSError as e:
        print(f"[auto_render_train_views] 无法启动子进程: {e}")
        return 1


def run_auto_novel_render_after_training(args) -> int:
    """
    训练结束后以子进程调用 render_novel_sar_views.py，避免与当前进程 CUDA 状态纠缠。
    返回子进程退出码（0 表示成功）。
    """
    repo = os.path.dirname(os.path.abspath(__file__))
    ov = getattr(args, "auto_novel_render_script", None)
    script = _resolve_novel_render_script(repo, ov if isinstance(ov, str) else None)
    if not os.path.isfile(script):
        print(f"[auto_novel_render] 未找到脚本: {script}")
        return 1

    mp = os.path.abspath(args.model_path)
    it = int(getattr(args, "iterations", 0))
    cmd = [
        sys.executable,
        script,
        "-m",
        mp,
        "--iteration",
        str(it),
        "--azimuth_start",
        str(int(getattr(args, "auto_novel_azimuth_start", 0))),
        "--azimuth_end",
        str(int(getattr(args, "auto_novel_azimuth_end", 360))),
        "--azimuth_step",
        str(int(getattr(args, "auto_novel_azimuth_step", 15))),
    ]
    out_dir_manual = getattr(args, "auto_novel_output_dir", None)
    use_unique = bool(getattr(args, "auto_novel_unique_folder", True))
    if out_dir_manual:
        resolved_out = os.path.abspath(out_dir_manual)
    elif use_unique:
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{time.time_ns() % 1_000_000:06d}"
        resolved_out = os.path.join(mp, "novel_views_runs", f"run_{run_id}")
    else:
        resolved_out = None

    if resolved_out is not None:
        cmd.extend(["--output_dir", resolved_out])
    if getattr(args, "auto_novel_same_depression_only", False):
        cmd.append("--same_depression_only")
    dep_angles = getattr(args, "auto_novel_depression_angles", None)
    if isinstance(dep_angles, str) and dep_angles.strip():
        cmd.extend(["--depression_angles", dep_angles.strip()])
    npm = getattr(args, "auto_novel_pose_mode", None)
    if npm:
        cmd.extend(["--novel_pose_mode", npm])
    if getattr(args, "auto_novel_quiet", False):
        cmd.append("--quiet")

    print("\n" + "=" * 70)
    print("[auto_novel_render] SAR 网格新视角渲染（默认即 render_novel_sar_views.py）")
    print(f"脚本: {script}")
    print(
        "说明: 输出位姿为该脚本生成的俯仰×方位网格（含插值），"
        "**不是**数据集里每条训练相机的精确条目；评判训练拟合请加 --auto_render_train_views。"
    )
    print("=" * 70)
    if resolved_out is not None:
        print(f"[auto_novel_render] 本次输出目录: {resolved_out}")
    else:
        print("[auto_novel_render] 未传 --output_dir，将使用渲染脚本自带默认目录（常为 novel_views_grid）")
    print("命令:", " ".join(cmd))

    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        p = subprocess.run(cmd, cwd=repo)
        return int(p.returncode)
    except OSError as e:
        print(f"[auto_novel_render] 无法启动子进程: {e}")
        return 1


def preprocess_sar_dataset(args):
    """
    预处理 SAR 数据集：计算散射/语义掩码。

    默认优先复用 convert 特征提取写入的 scattering_masks/from_convert/（与 SfM 语义分割一致）。
    若无缓存或 --no-reuse_convert_scattering_masks，则在 run_<时间戳> 下重算。
    路径写入 args.scattering_masks_cache_dir。

    掩码所用的灰度图与 convert 特征提取一致（cv2 灰度 + preprocess_sar_image），
    可与 <source_path>/adaptive_threshold_visualization 下的区域划分图对照。
    """
    if not args.filter_by_scattering or not SCATTERING_FILTER_AVAILABLE:
        print("\n跳过散射强度掩码预处理")
        return None
    
    print("\n" + "="*70)
    print("SAR数据集预处理：计算散射强度掩码")
    print("="*70)
    
    images_folder = os.path.join(args.source_path, "images")
    if not os.path.exists(images_folder):
        # 尝试其他可能的图像文件夹名称
        for alt_name in ["input", "imgs", "image"]:
            alt_path = os.path.join(args.source_path, alt_name)
            if os.path.exists(alt_path):
                images_folder = alt_path
                break
    
    if not os.path.exists(images_folder):
        print(f"警告: 找不到图像文件夹 {images_folder}")
        return None

    from sar.semantic_mask_config import (
        apply_convert_semantic_params_to_args,
        diagnose_convert_masks_load,
        load_convert_scattering_masks,
        list_convert_mask_runs,
        load_scattering_masks_from_dir,
        read_convert_semantic_params,
        summarize_scattering_mask_labels,
        semantic_mask_kwargs_from_args,
    )

    masks_dir_override = getattr(args, "scattering_masks_dir", None)
    if masks_dir_override:
        masks = load_scattering_masks_from_dir(masks_dir_override, images_folder)
        if masks:
            args.scattering_masks_cache_dir = masks_dir_override
            print(f"  使用指定掩码目录: {masks_dir_override} ({len(masks)} 张)")
        else:
            print(f"  警告: --scattering_masks_dir 无效或不完整: {masks_dir_override}")
            return None
    elif getattr(args, "reuse_convert_scattering_masks", True):
        run_id = getattr(args, "scattering_masks_run", "latest")
        loaded = load_convert_scattering_masks(
            args.source_path, images_folder, run_id=run_id
        )
        if loaded is not None:
            masks, cache_dir = loaded
            args.scattering_masks_cache_dir = cache_dir
            runs = list_convert_mask_runs(args.source_path)
            print(f"  复用 convert 语义掩码（与 SfM 一致）: {cache_dir}")
            print(f"  共 {len(masks)} 张；from_convert 历史批次: {len(runs)} 个")
            conv_params = read_convert_semantic_params(cache_dir)
            if conv_params:
                applied = apply_convert_semantic_params_to_args(
                    args, conv_params, sys.argv[1:]
                )
                print(f"  批次语义参数: {conv_params}")
                if applied:
                    print(
                        "  已同步到训练 args（与 convert 一致）: "
                        + ", ".join(applied[:12])
                        + (" ..." if len(applied) > 12 else "")
                    )
                args._semantic_params_from_convert = conv_params
            label_pct = summarize_scattering_mask_labels(masks)
            if label_pct:
                print(
                    "  标签占比(全数据集像素): "
                    f"背景={label_pct.get('label_0', 0):.1f}% "
                    f"目标内部={label_pct.get('label_1', 0):.1f}% "
                    f"边缘={label_pct.get('label_2', 0):.1f}% "
                    f"阴影={label_pct.get('label_3', 0):.1f}%"
                )
                print(
                    f"  目标+边缘={label_pct.get('label_target_plus_edge', 0):.1f}% "
                    f"（target_only 阶段 L1 主要在此）; "
                    f"单张中位数={label_pct.get('per_image_target_edge_median_pct', 0):.1f}%"
                )
                if label_pct.get("label_other", 0) > 0.01:
                    print(f"  ⚠️ 非标准标签像素={label_pct.get('label_other', 0):.2f}%")
        else:
            masks = None
            diag = diagnose_convert_masks_load(args.source_path, images_folder, run_id=run_id)
            if diag:
                print(f"  未命中 convert 缓存: {diag.get('reason')}")
    else:
        masks = None

    if masks is None:
        os.makedirs(os.path.join(args.source_path, "scattering_masks"), exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + f"_{time.time_ns() % 1_000_000:06d}"
        cache_dir = os.path.join(args.source_path, "scattering_masks", f"run_{run_id}")
        os.makedirs(cache_dir, exist_ok=True)
        args.scattering_masks_cache_dir = cache_dir
        print(f"  未找到 convert 掩码或已禁用复用，本次重算输出: {cache_dir}")
        print(f"  文件命名: <图像 stem>_mask.npy")
        sk = semantic_mask_kwargs_from_args(args)
        masks = compute_scattering_masks_for_dataset(
            images_folder,
            cache_dir=cache_dir,
            semantic_segmentation=not getattr(args, "no_semantic_segmentation", False),
            **{k: v for k, v in sk.items()},
        )

    sem_on = not getattr(args, 'no_semantic_segmentation', False)
    save_sem_vis = sem_on and not getattr(args, 'no_save_semantic_visualization', False)
    if masks and save_sem_vis:
        from pathlib import Path

        data_prep = Path(__file__).resolve().parent / "scripts" / "data_prep"
        dp = str(data_prep)
        if dp not in sys.path:
            sys.path.insert(0, dp)
        try:
            from feature_extraction import visualize_scattering_regions  # type: ignore
        except Exception as e:
            print(f"  ⚠️ 无法导入 visualize_scattering_regions，跳过语义可视化导出: {e}")
        else:
            region_dir = os.path.join(args.scattering_masks_cache_dir, "region_classification")
            os.makedirs(region_dir, exist_ok=True)
            args.semantic_visualization_dir = region_dir
            n_ok = 0
            for fname in sorted(masks.keys()):
                img_path = os.path.join(images_folder, fname)
                if not os.path.isfile(img_path):
                    continue
                try:
                    img = load_sar_grayscale_like_convert(img_path)
                    if getattr(img, "dtype", None) != np.float32:
                        img = img.astype(np.float32)
                    out = visualize_scattering_regions(
                        img, masks[fname], region_dir, fname, verbose=False
                    )
                    if out:
                        n_ok += 1
                except Exception as ex:
                    print(f"  ⚠️ 语义可视化失败 {fname}: {ex}")
            print(f"\n语义区域叠加图已保存（共 {n_ok} 张）: {region_dir}")
    
    # 可视化第一个掩码（底图与掩码计算使用同一种读图+preprocess_sar_image）
    if masks:
        first_image = list(masks.keys())[0]
        img_path = os.path.join(images_folder, first_image)
        img = load_sar_grayscale_like_convert(img_path).astype(np.float32)
        vis_path = os.path.join(args.scattering_masks_cache_dir, "mask_visualization_example.png")
        visualize_scattering_mask(img, masks[first_image], vis_path)
        print(f"\n散射强度掩码可视化示例: {vis_path}")
    
    return masks


def main():
    """
    主训练函数
    """
    # 解析参数
    args = parse_sar_arguments()
    
    print("\n" + "="*70)
    print("SAR-GS: 3D Gaussian Splatting for SAR Target Reconstruction")
    print("基于论文: Mapping and Projection Algorithm + SDGR")
    print("="*70)
    
    # 显示配置
    print(f"\n【训练配置】")
    print(f"  数据集路径: {args.source_path}")
    print(f"  输出路径: {args.model_path}")
    print(f"  训练迭代次数: {args.iterations}")
    print(f"\n【SAR模式】")
    print(f"  启用SAR模式: {getattr(args, 'sar_mode', True)}")
    print(f"  使用SDGR渲染器: {getattr(args, 'sar_rendering', True)}")
    _sar_fast = bool(getattr(args, 'sar_fast_mode', True))
    print(f"  sar_fast_mode: {_sar_fast}  (加 --no_sar_fast_mode 为 False，走 SAR-3DGS SDGR)")
    try:
        from diff_sar_rasterization import SAR_RASTERIZATION_AVAILABLE
        print(f"  diff_sar_rasterization CUDA 扩展: {'已安装' if SAR_RASTERIZATION_AVAILABLE else '未安装'}")
    except ImportError:
        print(f"  diff_sar_rasterization CUDA 扩展: 未安装")
    _sdgr_cuda = bool(getattr(args, "sar_sdgr_cuda", True))
    if not _sar_fast:
        print(f"  sar_sdgr_cuda: {_sdgr_cuda}  (默认 True=CUDA alpha 混合；更稳加 --no_sar_sdgr_cuda)")
    print(f"  散射强度过滤: {args.filter_by_scattering}")
    print(f"\n【SAR系统参数】")
    print(f"  平台高度: {args.sar_camera_height:.1f} m")
    print(f"  平台速度: {args.sar_platform_velocity:.1f} m/s")
    print(f"  脉冲重复频率: {getattr(args, 'sar_prf', 2500.0):.1f} Hz")
    print(f"  带宽: {getattr(args, 'sar_bandwidth', 500e6)/1e6:.0f} MHz")
    print(f"  相机分布: {args.sar_camera_distribution_mode}")
    if bool(getattr(args, "sar_stratify_azimuth", False)):
        print(f"\n【方位均权抽样】")
        print(f"  sar_stratify_azimuth: True")
        _ab = int(getattr(args, "sar_azimuth_bins", 24))
        if _ab <= 0:
            print(f"  sar_azimuth_bins: 自动（训练开始后桶数 = 训练相机数量，当前参数值 {_ab}）")
        else:
            print(f"  sar_azimuth_bins: {_ab}")
        if getattr(args, "sar_equal_view_sampling", False):
            print(f"  （由 --sar_equal_view_sampling 启用；需相机或 MSTAR 文件名可解析方位）")
    if bool(getattr(args, "sar_interp_soft_supervision", False)):
        print(f"\n【插值方位软监督】")
        print(f"  sar_interp_soft_supervision: True")
        print(f"  every={getattr(args, 'sar_interp_soft_every', 500)}, weight={getattr(args, 'sar_interp_soft_weight', 0.2)}")
        print(f"  t_values={getattr(args, 'sar_interp_soft_t_values', '0.25,0.5,0.75')}, azimuth_space={getattr(args, 'sar_interp_soft_azimuth_space', 'filename')}")
        print(f"  目标: 双邻域加权 L1（blend={getattr(args, 'sar_interp_soft_use_blend_target', False)}）")
    print(f"\n【散射/语义掩码】")
    if args.filter_by_scattering:
        sem_on = not getattr(args, 'no_semantic_segmentation', False)
        print(f"  模式: {'convert 语义 0/1/3（目标含原边缘）' if sem_on else '简易三类'}")
        if getattr(args, "_semantic_params_from_convert", None):
            print(f"  参数来源: convert 批次 semantic_params.json（与 SfM 掩码一致）")
        print(f"  目标分位数: {args.target_percentile}%")
        print(f"  背景分位数: {args.background_percentile}%")
        if sem_on:
            print(f"  shadow_threshold_factor: {getattr(args, 'shadow_threshold_factor', 0.1)}")
            if not getattr(args, 'no_save_semantic_visualization', False):
                print(
                    "  语义叠加图目录（预处理写入）: "
                    "<scattering_masks/run_*>/region_classification/（与 convert 中 region_classification_*.png 一致）"
                )
            else:
                print("  语义叠加图: 已用 --no_save_semantic_visualization 关闭导出")
            print(
            f"  损失权重 - 背景: {args.background_loss_weight}, 阴影: {getattr(args, 'shadow_loss_weight', 0.55)}, "
            f"目标(内部+边缘同一权重): {args.target_loss_weight}"
            )
        print(
            f"  深度顺序约束: weight={getattr(args, 'sar_target_depth_order_weight', 0.05)}, "
            f"margin={getattr(args, 'sar_target_depth_order_margin', 1e-3)}"
        )
    if getattr(args, 'sar_refine_target_bg', False):
        print("\n【sar_refine_target_bg】")
        print(f"  均等分区 L1 (--sar_refine_uniform_loss): {getattr(args, 'sar_uniform_region_loss', False)}")
        print(f"  轻量 densify (--sar_refine_light_densify): {getattr(args, 'sar_refine_light_densify', False)}")
        print(
            "  片状背景/遮光板 (--no_sar_refine_seed_bg_shadow_plates 可关；轻量 densify 且允许时也铺板): "
            f"{getattr(args, 'sar_refine_seed_bg_shadow_plates', True)}"
        )
        print(f"  densify_until_iter: {getattr(args, 'densify_until_iter', '?')}")
    print(f"\n【形状约束】")
    if hasattr(args, 'sar_shape_constraint_weight'):
        print(f"  约束权重: {args.sar_shape_constraint_weight}")
        print(f"  约束迭代范围: {args.sar_shape_constraint_start_iter} - {args.sar_shape_constraint_end_iter}")
    else:
        print(f"  使用OptimizationParams默认值")
    print("="*70 + "\n")
    
    # 预处理：计算散射强度掩码
    scattering_masks = preprocess_sar_dataset(args)
    
    # 将散射强度掩码保存到args中，供训练使用
    args.scattering_masks = scattering_masks
    if getattr(args, "scattering_masks_cache_dir", None):
        print(f"散射掩码目录（本轮）: {args.scattering_masks_cache_dir}\n")
    if getattr(args, "semantic_visualization_dir", None):
        print(f"语义叠加图目录: {args.semantic_visualization_dir}\n")
    # 无需重复设置
    
    # 初始化训练
    print("\n开始训练...")
    safe_state(args.quiet)
    
    # 启动训练
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 提取参数组
    from arguments import ModelParams, OptimizationParams, PipelineParams
    model_params = ModelParams(ArgumentParser(), sentinel=True)
    opt_params = OptimizationParams(ArgumentParser())
    pipe_params = PipelineParams(ArgumentParser())
    
    dataset_params = model_params.extract(args)
    if getattr(args, "scattering_masks", None) is not None:
        dataset_params.scattering_masks = args.scattering_masks

    train_status = training(
        dataset_params,
        opt_params.extract(args),
        pipe_params.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from
    )

    if not train_status.get("ok", True):
        last_it = train_status.get("last_iteration", "?")
        reason = train_status.get("reason", "unknown")
        print("\n" + "=" * 70)
        print(f"训练异常中止（约 iter {last_it}，原因: {reason}）")
        if reason == "cuda":
            print("  SDGR 的 diff-sar-rasterization CUDA 在本机/本数据上不稳定。")
            print("  请新开 output 目录，加 --no_sar_sdgr_cuda 或 set SAR_SDGR_FORCE_PYTHON=1，")
            print("  然后再跑同一命令（不要续训已损坏的 checkpoint）。")
        print("=" * 70)
        import sys
        sys.exit(1)

    print("\n" + "="*70)
    print("训练完成!")
    print(f"模型保存在: {args.model_path}")
    if getattr(args, "sar_refine_target_bg", False) and not getattr(
        args, "no_sar_refine_eval_suggestions", False
    ):
        try:
            from sar.sar_refine_target_bg import write_refine_eval_suggestions

            path = write_refine_eval_suggestions(
                args.model_path, args.source_path, int(args.iterations)
            )
            print(f"新视角对比示例命令已写入: {path}")
        except Exception as e:
            print(f"写入 refine_eval_suggestions 失败: {e}")

    if getattr(args, "auto_render_train_views", False):
        trc = run_auto_render_train_views_after_training(args)
        if trc != 0:
            print(f"[auto_render_train_views] 子进程退出码 {trc}；训练相机渲染失败，请先查看上方日志。")
        else:
            print("[auto_render_train_views] 已完成；对比 render/train/renders 与 train/gt 即真实训练姿态。")

    if getattr(args, "auto_novel_render", False):
        code = run_auto_novel_render_after_training(args)
        if code != 0:
            print(f"[auto_novel_render] 子进程退出码 {code}，请根据上方渲染日志排查。")
        else:
            print("[auto_novel_render] 已完成。")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()

