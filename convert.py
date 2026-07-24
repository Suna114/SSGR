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
import struct
import os
import logging
import shutil
# 已删除sqlite3导入（此代码专用于SAR图像，不使用COLMAP数据库）
import traceback
import datetime
import importlib.util
import numpy as np
import cv2
import sys
from contextlib import redirect_stderr, redirect_stdout

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_DATA_PREP = os.path.join(_ROOT, "scripts", "data_prep")
if _DATA_PREP not in sys.path:
    sys.path.insert(0, _DATA_PREP)

# 已删除COLMAP相关导入（此代码专用于SAR图像）
from feature_extraction import extract_features_multiprocess, save_features_to_pickle_format
from feature_matching import match_features_sar_sift, save_matches_to_pickle_format
from sar_utils import parse_mstar_filename
from config import parse_arguments
from sfm import SFM
import argparse

import warnings

# 抑制所有不必要的警告
warnings.filterwarnings("ignore")
os.environ['TIFFIO_IGNORE_WARNINGS'] = '1'
os.environ['OPENCV_IO_ENABLE_JASPER'] = '0'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# 设置日志级别，避免不必要的日志输出
logging.getLogger('PIL').setLevel(logging.WARNING)

# 已删除OutputCapture类（此代码专用于SAR图像，不需要COLMAP输出捕获）


class StepLogger:
    """简单的去重日志器，避免重复输出"""

    def __init__(self):
        self._message_cache = set()

    def info(self, message, dedup=False):
        if dedup and message in self._message_cache:
            return
        print(message)
        if dedup:
            self._message_cache.add(message)


step_logger = StepLogger()


# 已删除COLMAP SfM相关函数（此代码专用于SAR图像，使用自定义sfm.py）


# 已删除COLMAP SfM总结报告函数（此代码专用于SAR图像）
# ========== 添加全局错误日志设置 ==========
def setup_error_logging(source_path, quiet=False):
    """设置错误日志系统"""
    log_dir = os.path.join(source_path, "error_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"error_log_{timestamp}.txt")

    # 配置logging
    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if not quiet:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True,
    )

    if not quiet:
        print(f"📝 错误日志将保存到: {log_file}")
    return log_file


def log_error(error_msg, exc_info=True):
    """统一的错误日志记录函数"""
    if exc_info:
        logging.error(error_msg, exc_info=True)
    else:
        logging.error(error_msg)

    # 同时在控制台显示
    print(f"❌ {error_msg}")


# SAR图像SfM重建脚本（专用于SAR图像，已删除COLMAP相关代码）
args = parse_arguments()
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"


def create_custom_features_and_matches():
    """使用自定义SAR-SIFT特征提取和匹配 - 修复版本"""
    print("🎯 使用自定义SAR-SIFT特征提取和匹配...")

    # 创建必要的目录（SAR图像不需要COLMAP目录结构）
    os.makedirs(args.source_path + "/input", exist_ok=True)
    os.makedirs(args.source_path + "/images", exist_ok=True)

    # 收集图像文件
    import glob
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(args.source_path, "input", ext)))

    if not image_files:
        # 如果没有在input目录找到图像，尝试在根目录查找
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(args.source_path, ext)))

        # 将找到的图像复制到input目录
        for img_file in image_files:
            shutil.copy2(img_file, os.path.join(args.source_path, "input", os.path.basename(img_file)))
            shutil.copy2(img_file, os.path.join(args.source_path, "images", os.path.basename(img_file)))

    print(f"找到 {len(image_files)} 张图像")

    if len(image_files) == 0:
        log_error("未找到任何图像文件", exc_info=False)
        return False

    if not getattr(args, "geometry_prior_rough_model", False):
        try:
            from sar.dataset_pose_defaults import apply_360_ring_convert_defaults

            apply_360_ring_convert_defaults(args, image_files)
        except Exception as e:
            print(f"⚠️ 360° 圆环默认参数推断失败（继续 convert）: {e}")

    # SfM 比例盒 before_filter 需要 target_dimension_constraints_summary.txt；
    # standalone 路径须早于特征提取与 SfM，且写出含「高度/长度/宽度 … 中位数=」的段落供 summary_parser 解析。
    if getattr(args, "convert_use_standalone_shadow_target_dimensions", False):
        print(
            "📐 [优先] standalone 阴影几何汇总 → target_dimension_constraints_summary.txt"
            "（供 SfM 阴影比例盒 before_filter）"
        )
        try:
            from shadow_target_dimensions_standalone import (
                run_standalone_target_dimension_summary_for_convert,
            )

            run_standalone_target_dimension_summary_for_convert(args)
        except Exception as e:
            print(f"⚠️ standalone 阴影汇总失败（将继续特征提取，SfM 比例盒可能缺少尺寸）: {e}")
            import traceback
            traceback.print_exc()

    try:
        # ========== 特征提取 ==========
        print("开始特征提取...")
        keypoints, descriptors, original_keypoints = extract_features_multiprocess(
            image_files, args.feature_type, args
        )

        if len(keypoints) == 0:
            log_error("特征提取失败，没有提取到任何特征点", exc_info=False)
            return False

        # 检查每张图像的特征点数量
        feature_stats = {}
        for img_name, kps in keypoints.items():
            feature_stats[img_name] = len(kps)

        # 输出特征点统计
        print(f"📊 特征点统计:")
        for img_name, count in sorted(feature_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {img_name}: {count} 个特征点")

        total_features = sum(feature_stats.values())
        print(f"   总共提取 {total_features} 个特征点，平均每张图像 {total_features / len(feature_stats):.1f} 个特征点")

        # ========== 创建可视化目录 ==========
        if args.save_match_images:
            # 创建特征点可视化目录
            feature_viz_dir = os.path.join(args.source_path, "feature_visualization")
            os.makedirs(feature_viz_dir, exist_ok=True)

            # 创建匹配图目录
            matches_dir = os.path.join(args.source_path, "feature_matches")
            os.makedirs(matches_dir, exist_ok=True)

            print(f"   特征点图将保存到: {feature_viz_dir}")
            print(f"   特征匹配图将保存到: {matches_dir}")

        # ========== 特征匹配 ==========
        print("开始特征匹配...")
        # 添加全局匹配提示
        if args.max_pairs_per_image == 0:
            print(f"   匹配模式: 🔄 全局匹配 (所有可能的图像对)")
        else:
            print(f"   匹配模式: 🎯 基于方位角选择 (每张图像最多 {args.max_pairs_per_image} 对)")
        print(f"   匹配参数:")
        print(f"     - 双向匹配: {getattr(args, 'use_bidirectional_matching', True)}")
        print(f"     - 匹配方法: {getattr(args, 'matching_method', 'bidirectional')}")
        print(f"     - 最大比率: {args.max_ratio}")
        print(f"     - RANSAC迭代: {args.ransac_iterations}")
        print(f"     - RANSAC误差阈值: {args.ransac_error_threshold}")
        # 只在非全局匹配模式下显示方位角差异
        if args.max_pairs_per_image != 0:
            print(f"     - 最大方位角差异: {args.max_azimuth_diff}°")

        print(f"     - 保存匹配图: {args.save_match_images}")
        print(f"     - 最大匹配图数量: {args.max_match_pairs}")

        # 确保正确传递所有参数到匹配函数
        matches_dict = match_features_sar_sift(
            keypoints, descriptors, original_keypoints, image_files, args
        )

        # ========== 匹配结果分析 ==========
        if matches_dict is None or len(matches_dict) == 0:
            print("⚠️ 未找到任何特征匹配")
            # 不立即返回False，继续创建数据库，因为可能有些图像对没有匹配但其他图像对可能有
        else:
            print(f"✅ 特征匹配完成: 找到 {len(matches_dict)} 对有效匹配")

            # 分析匹配统计
            match_stats = analyze_match_statistics(matches_dict, min_matches_threshold=3)

            # 输出匹配质量信息
            total_matches = sum(len(matches) for matches in matches_dict.values())
            avg_matches_per_pair = total_matches / len(matches_dict) if matches_dict else 0
            print(f"   总匹配点数: {total_matches}")
            print(f"   平均每对匹配点数: {avg_matches_per_pair:.1f}")

            # 检查匹配图是否保存
            if args.save_match_images:
                matches_dir = os.path.join(args.source_path, "feature_matches")
                if os.path.exists(matches_dir):
                    match_images = [f for f in os.listdir(matches_dir) if
                                    f.startswith("matches_") and f.endswith(('.png', '.jpg'))]
                    print(f"   保存的匹配图数量: {len(match_images)}")
                    if len(match_images) > 0:
                        print(f"   匹配图样例: {match_images[:3]}")  # 显示前3个匹配图文件名
                    else:
                        print("   ⚠️ 警告: 没有找到任何匹配图文件")
                else:
                    print("   ❌ 错误: 匹配图目录不存在")

        # ========== 保存特征和匹配为pickle格式（用于sfm.py） ==========
        # 默认启用保存，除非明确禁用
        save_for_sfm = getattr(args, 'save_features_for_sfm', True)
        if save_for_sfm:
            print("💾 保存特征为pickle格式（用于sfm.py）...")
            try:
                feat_dir = save_features_to_pickle_format(original_keypoints, descriptors, args)
                if feat_dir:
                    print(f"✅ 特征已保存到: {feat_dir}")
            except Exception as e:
                print(f"⚠️ 保存特征为pickle格式时出错: {e}")
                import traceback
                traceback.print_exc()

        save_matches_for_sfm = getattr(args, 'save_matches_for_sfm', True)
        if save_matches_for_sfm and matches_dict and len(matches_dict) > 0:
            print("💾 保存匹配为pickle格式（用于sfm.py）...")
            try:
                matches_dir = save_matches_to_pickle_format(matches_dict, original_keypoints, args)
                if matches_dir:
                    print(f"✅ 匹配已保存到: {matches_dir}")
            except Exception as e:
                print(f"⚠️ 保存匹配为pickle格式时出错: {e}")
                import traceback
                traceback.print_exc()

        # SAR图像使用自定义sfm.py进行重建，不需要COLMAP数据库
        print("✅ 特征和匹配已保存为pickle格式，准备使用自定义sfm.py进行SAR重建")
        return True

    except Exception as e:
        log_error(f"自定义特征处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_match_statistics(matches_dict, min_matches_threshold=5):
    """
    分析匹配统计信息
    """
    print(f"\n🔍 匹配统计分析 (最小匹配数阈值: {min_matches_threshold})")
    print("=" * 60)

    image_connections = {}
    image_match_counts = {}

    for (img1, img2), matches in matches_dict.items():
        match_count = len(matches)
        if match_count >= min_matches_threshold:
            image_connections[img1] = image_connections.get(img1, 0) + 1
            image_connections[img2] = image_connections.get(img2, 0) + 1

            if img1 not in image_match_counts:
                image_match_counts[img1] = []
            if img2 not in image_match_counts:
                image_match_counts[img2] = []

            image_match_counts[img1].append((img2, match_count))
            image_match_counts[img2].append((img1, match_count))

    total_images = len(image_connections)
    total_pairs = sum(image_connections.values()) // 2

    print(f"📊 总体统计:")
    print(f"   - 具有至少{min_matches_threshold}个匹配点的图像数量: {total_images}")
    print(f"   - 满足条件的图像对数量: {total_pairs}")
    print(f"   - 总匹配对数: {len(matches_dict)}")

    # 显示连接最多的图像
    if image_connections:
        print(f"\n🏆 连接最多的图像 (前10名):")
        sorted_images = sorted(image_connections.items(), key=lambda x: x[1], reverse=True)[:10]

        for i, (img_name, connection_count) in enumerate(sorted_images, 1):
            matches_with_this_image = image_match_counts.get(img_name, [])
            avg_matches = sum(match_count for _, match_count in matches_with_this_image) / len(
                matches_with_this_image) if matches_with_this_image else 0
            print(f"   {i:2d}. {img_name}: {connection_count}个连接, 平均{avg_matches:.1f}个匹配点")

    # 显示最强的匹配对
    print(f"\n💪 最强的匹配对 (前15名):")
    strong_pairs = []
    for (img1, img2), matches in matches_dict.items():
        match_count = len(matches)
        if match_count >= min_matches_threshold:
            strong_pairs.append((img1, img2, match_count))

    strong_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (img1, img2, match_count) in enumerate(strong_pairs[:15], 1):
        print(f"   {i:2d}. {img1} ↔ {img2}: {match_count}个匹配点")

    # 连接性分析
    print(f"\n📈 连接性分析:")
    if total_images > 0:
        connection_counts = list(image_connections.values())
        max_connections = max(connection_counts)
        min_connections = min(connection_counts)
        avg_connections = sum(connection_counts) / len(connection_counts)

        print(f"   - 最大连接数: {max_connections}")
        print(f"   - 最小连接数: {min_connections}")
        print(f"   - 平均连接数: {avg_connections:.1f}")

        connection_ranges = {
            "1-2个连接": 0,
            "3-5个连接": 0,
            "6-10个连接": 0,
            "10+个连接": 0
        }

        for count in connection_counts:
            if count <= 2:
                connection_ranges["1-2个连接"] += 1
            elif count <= 5:
                connection_ranges["3-5个连接"] += 1
            elif count <= 10:
                connection_ranges["6-10个连接"] += 1
            else:
                connection_ranges["10+个连接"] += 1

        print(f"   - 连接分布:")
        for range_name, count in connection_ranges.items():
            percentage = (count / total_images) * 100
            print(f"     * {range_name}: {count}张图像 ({percentage:.1f}%)")

    print("=" * 60)
    return {
        'total_images': total_images,
        'total_pairs': total_pairs,
        'image_connections': image_connections,
        'strong_pairs': strong_pairs
    }

# 已删除COLMAP数据库验证函数（此代码专用于SAR图像，不使用COLMAP数据库）

# 已删除COLMAP特征提取和匹配函数（此代码专用于SAR图像，使用SAR-SIFT）

# 已删除图像去畸变函数（SAR图像不需要去畸变，自定义sfm.py直接输出点云）


def _print_target_dimension_summary_digest(source_path: str, visualization_dir: str) -> None:
    """
    终端播报：仅摘录 shadow_target_dimensions_standalone 写入的 txt 元信息与推荐比例。
    不再逐张展开旧版「斜视 15° 步进」式详表（避免与当前 90°/270° 侧视汇总口径混淆）。
    """
    vd = (visualization_dir or "adaptive_threshold_visualization").strip() or "adaptive_threshold_visualization"
    summary_file = os.path.join(source_path, vd, "target_dimension_constraints_summary.txt")
    print("\n" + "=" * 70)
    print("【目标尺寸汇总】")
    print(
        "口径: scripts/data_prep/shadow_target_dimensions_standalone.py；"
        "当前推荐 `--target_dimension_summary_azimuth_mode mstar_axis_split`（仅 90°/270°±容差，L=Tr、W=Taz）。"
        "以下为汇总文件摘录，精确筛选条件以「方位角筛选」行为准。"
    )
    print("=" * 70)
    try:
        if not os.path.isfile(summary_file):
            print("⚠️ 未找到目标尺寸汇总文件")
            print(f"   预期路径: {summary_file}")
            print(
                "   请确认已运行 convert（含 --convert_use_standalone_shadow_target_dimensions），"
                "或手动运行 shadow_target_dimensions_standalone.py 写出上述 txt；"
                "并检查方位模式与数据是否匹配（侧视样本过少时汇总可能为空）。"
            )
            print("=" * 70 + "\n")
            return
        with open(summary_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith("来源:"):
                print(s)
            elif s.startswith("方位角筛选:"):
                print(s)
            elif s.startswith("单张几何模式"):
                print(s)
            elif s.startswith("鲁棒剔除:"):
                print(s)
            elif "成功算出 target_dimensions:" in s:
                print(s)
            elif s.startswith("有效图像数量:"):
                print(s)
        in_rec = False
        for line in lines:
            s = line.strip()
            if "【推荐的点云约束参数】" in s:
                in_rec = True
                print("\n" + s)
                continue
            if in_rec:
                if s.startswith("【") and "推荐的点云约束参数" not in s:
                    break
                if s:
                    print("  " + s)
        print(f"\n完整汇总报告: {summary_file}")
    except Exception as e:
        print(f"⚠️ 读取汇总文件时出错: {e}")
        traceback.print_exc()
    print("=" * 70 + "\n")


def _sparse_model_ready(source_path: str) -> bool:
    """sparse/0 是否已有可复用的点云模型。"""
    sparse0 = os.path.join(source_path, "sparse", "0")
    return os.path.isfile(os.path.join(sparse0, "points3D.bin")) or os.path.isfile(
        os.path.join(sparse0, "points3D.txt")
    )


def _require_sparse_model_or_exit(source_path: str, *, flag_label: str) -> None:
    if _sparse_model_ready(source_path):
        print(f"⏭️ {flag_label}：复用已有 sparse/0 点云，跳过 SfM 重建")
        return
    log_error(
        f"{flag_label}：未找到 sparse/0/points3D.bin（或 .txt），无法复用上次的模型",
        exc_info=False,
    )
    exit(1)


def _ensure_target_dimension_summary_for_geometry_prior(source_path: str, cli_args) -> None:
    """Ensure summary-box priors have the target dimension txt, even in skip modes."""
    if not (
        bool(getattr(cli_args, "geometry_prior_gs_model", False))
        or bool(getattr(cli_args, "geometry_prior_rough_model", False))
        or bool(getattr(cli_args, "convert_use_standalone_shadow_target_dimensions", False))
    ):
        return
    viz_dir = getattr(cli_args, "visualization_dir", "adaptive_threshold_visualization") or "adaptive_threshold_visualization"
    summary_file = os.path.join(source_path, viz_dir, "target_dimension_constraints_summary.txt")
    if os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0:
        return
    try:
        from shadow_target_dimensions_standalone import (
            run_standalone_target_dimension_summary_for_convert,
        )

        print("📐 缺少目标尺寸汇总，自动运行 standalone 阴影几何汇总...")
        run_standalone_target_dimension_summary_for_convert(cli_args)
    except Exception as e:
        print(f"⚠️ 自动生成目标尺寸汇总失败（继续 convert）: {e}")


def _run_post_sparse_geometry_stages(source_path: str, cli_args) -> None:
    """sparse/0 就绪后：比例盒 → 圆环外参 → 接地点居中 → post_sfm。"""
    _ensure_target_dimension_summary_for_geometry_prior(source_path, cli_args)

    try:
        from post_sfm.sfm_configured_summary_box import (
            run_sfm_configured_summary_box_mesh_prior,
        )

        run_sfm_configured_summary_box_mesh_prior(source_path, cli_args)
    except Exception as e:
        print(f"⚠️ sfm_summary_box 步骤失败（继续 convert）: {e}")

    try:
        from sar.gs_satellite_poses import maybe_apply_gs_satellite_poses_to_sparse

        maybe_apply_gs_satellite_poses_to_sparse(
            source_path, cli_args, transform_points=False
        )
    except Exception as e:
        print(f"⚠️ gs_satellite_poses 步骤失败（继续 convert）: {e}")

    try:
        from sar.scene_anchor import maybe_center_colmap_scene_at_sar_anchor

        maybe_center_colmap_scene_at_sar_anchor(
            source_path, cli_args, tag="convert_after_gs_ring"
        )
    except Exception as e:
        print(f"⚠️ SAR 场景接地点居中失败（继续 convert）: {e}")

    try:
        from sar.scene_scale_closure import sync_target_envelope_from_sparse_points3d

        sync_target_envelope_from_sparse_points3d(source_path, cli_args)
    except Exception as e:
        print(f"⚠️ 点云包络回写 sar_scale_params 失败（继续 convert）: {e}")

    try:
        from sar.optical_envelope_scale_closure import maybe_optical_envelope_scale_closure

        maybe_optical_envelope_scale_closure(source_path, cli_args)
    except Exception as e:
        print(f"⚠️ optical verify 尺度闭合失败（继续 convert）: {e}")

    if (getattr(cli_args, "post_sfm_modules", "") or "").strip():
        try:
            from post_sfm.runner import run_post_sfm_modules

            run_post_sfm_modules(source_path, cli_args)
        except Exception as e:
            log_error(f"post_sfm 流水线失败: {e}")
            exit(1)

    try:
        from sar.colmap_pose_sync import sync_sar_scale_params_cameras_from_sparse

        sync_sar_scale_params_cameras_from_sparse(
            source_path, args=cli_args, verbose=True
        )
    except Exception as e:
        print(f"⚠️ sar_scale_params ↔ sparse 外参同步失败（继续 convert）: {e}")


def _run_auto_optical_verify(source_path: str, cli_args) -> None:
    """convert 结束时导入 verify 脚本并生成 optical 评测报告。"""
    if not bool(getattr(cli_args, "convert_auto_verify_optical", False)):
        return
    script = os.path.join(_ROOT, "scripts", "verify_sar_imaging_envelope_match.py")
    if not os.path.isfile(script):
        print(f"⚠️ convert_auto_verify_optical：找不到 {script}")
        return
    try:
        from sar.imaging_mode import dataset_convention_debug_summary, resolve_dataset_world_yaw_deg

        yaw = float(resolve_dataset_world_yaw_deg(cli_args, default_yaw_deg=-90.0))
        print(
            "\n[convert_auto_verify_optical] "
            f"{dataset_convention_debug_summary(source_path)}; world_yaw={yaw:.1f} deg"
        )
    except Exception:
        yaw = float(getattr(cli_args, "sfm_mesh_prior_extra_world_yaw_deg", -90.0) or -90.0)
    old_argv = list(sys.argv)
    try:
        spec = importlib.util.spec_from_file_location(
            "verify_sar_imaging_envelope_match_convert_auto",
            script,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {script}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        sys.argv = [
            script,
            "-s",
            source_path,
            "--use_nominal_box",
            "--projection_mode",
            "optical",
            "--world_yaw_deg",
            f"{yaw:g}",
        ]
        print("\n[convert_auto_verify_optical] 自动运行 optical 成像包络评测")
        mod.main()
    except Exception as e:
        print(f"⚠️ convert_auto_verify_optical 失败（继续 convert）: {e}")
    finally:
        try:
            sys.argv = old_argv
        except Exception:
            pass


def _quiet_output_path() -> str:
    return os.path.abspath(args.source_path)


def _open_convert_quiet_log():
    log_dir = os.path.join(args.source_path, "convert_logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"convert_{timestamp}.txt")
    return open(log_file, "a", encoding="utf-8", errors="replace"), log_file


def _main_impl():
    """主函数"""
    # 设置错误日志
    log_file = setup_error_logging(args.source_path)

    try:
        skip_matching = bool(getattr(args, "skip_matching", False))
        skip_sfm = bool(getattr(args, "skip_sfm", False))

        if not skip_matching:
            os.makedirs(args.source_path + "/input", exist_ok=True)
            custom_success = create_custom_features_and_matches()
            if not custom_success:
                log_error("SAR特征提取和匹配失败，退出流程", exc_info=False)
                exit(1)
        else:
            print("⏭️ skip_matching：跳过 SAR-SIFT 特征提取与匹配")

        if not skip_matching and not skip_sfm:
            print("🔧 使用自定义sfm.py进行SAR SfM重建...")
            sfm_success = run_custom_sfm_reconstruction()
            if not sfm_success:
                log_error("SfM失败，退出流程", exc_info=False)
                exit(1)
        else:
            flag_label = "skip_matching" if skip_matching else "skip_sfm"
            _require_sparse_model_or_exit(args.source_path, flag_label=flag_label)

        _run_post_sparse_geometry_stages(args.source_path, args)

        print("✅ SAR重建完成，点云已直接生成")

        viz_dir = getattr(args, "visualization_dir", "adaptive_threshold_visualization") or "adaptive_threshold_visualization"
        _print_target_dimension_summary_digest(args.source_path, viz_dir)
        _run_auto_optical_verify(args.source_path, args)
        if (args.resize):
            print("Copying and resizing...")

            # Resize images.
            os.makedirs(args.source_path + "/images_2", exist_ok=True)
            os.makedirs(args.source_path + "/images_4", exist_ok=True)
            os.makedirs(args.source_path + "/images_8", exist_ok=True)
            # Get the list of files in the source directory
            files = os.listdir(args.source_path + "/images")
            # Copy each file from the source directory to the destination directory
            for file in files:
                source_file = os.path.join(args.source_path, "images", file)

                destination_file = os.path.join(args.source_path, "images_2", file)
                shutil.copy2(source_file, destination_file)
                exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
                if exit_code != 0:
                    log_error(f"50% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)

                destination_file = os.path.join(args.source_path, "images_4", file)
                shutil.copy2(source_file, destination_file)
                exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
                if exit_code != 0:
                    log_error(f"25% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)

                destination_file = os.path.join(args.source_path, "images_8", file)
                shutil.copy2(source_file, destination_file)
                exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
                if exit_code != 0:
                    log_error(f"12.5% resize failed with code {exit_code}. Exiting.")
                    exit(exit_code)

        print("✅ 所有流程完成!")

    except Exception as e:
        log_error(f"主流程执行失败: {e}")
        exit(1)
# 已删除COLMAP数据库读取测试函数（此代码专用于SAR图像）


# 在 convert.py 中添加以下函数
# 已删除COLMAP重建质量检查函数（此代码专用于SAR图像，使用自定义sfm.py）

# 已删除COLMAP二进制文件读取函数（此代码专用于SAR图像，使用自定义sfm.py输出PLY格式点云）


def main():
    quiet = bool(getattr(args, "convert_output_path_only", False))
    if quiet:
        stream, _log_path = _open_convert_quiet_log()
        with stream, redirect_stdout(stream), redirect_stderr(stream):
            _main_impl()
        print(_quiet_output_path())
        return
    _main_impl()


def run_custom_sfm_reconstruction():
    """
    使用自定义sfm.py进行SfM重建
    """
    print("🚀 开始使用自定义sfm.py进行SfM重建...")

    try:
        from sar.sfm_options import build_sfm_options

        sfm_opts = build_sfm_options(args)
        if getattr(sfm_opts, "sfm_point_height_length_ratio", None) is None:
            print(
                f"📐 SO-RCG 点云高度/长宽比：优先从阴影汇总加载 "
                f"（{getattr(sfm_opts, 'target_dimension_summary_file', '')}；"
                f"特征阶段须已成功写出 target_dimension_constraints_summary.txt）"
            )
        else:
            print(
                f"📐 SO-RCG 点云高度/长度比：使用固定阈值 "
                f"{sfm_opts.sfm_point_height_length_ratio}（已 --sfm_ignore_target_dimension_summary_ratios）"
            )

        # 打印方法选择信息
        if sfm_opts.sfm_use_sorc:
            print(f"✅ 将使用SO-RCG方法进行SAR SfM重建")
            print(f"   - 最大迭代次数: {sfm_opts.sfm_sorc_max_iterations}")
            print(f"   - 收敛容差: {sfm_opts.sfm_sorc_tolerance}")
            print(f"   - 使用5个DoF约束: {sfm_opts.sfm_sorc_use_5dof}")
        else:
            print(f"ℹ️ 将使用矩阵分解方法进行SAR SfM重建")
        
        # 确保图像目录存在（复制图像到sfm.py期望的位置）
        images_dir = os.path.join(sfm_opts.data_dir, sfm_opts.dataset, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # 复制图像文件
        input_dir = os.path.join(args.source_path, "input")
        if os.path.exists(input_dir):
            import glob
            import shutil
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            
            for img_file in image_files:
                img_name = os.path.basename(img_file)
                dest_path = os.path.join(images_dir, img_name)
                if not os.path.exists(dest_path):
                    shutil.copy2(img_file, dest_path)
            
            print(f"✅ 已准备 {len(image_files)} 张图像到 {images_dir}")
        
        # 后处理参数（将fund_method转换为OpenCV常量）
        # 注意：sfm.cli.PostprocessArgs 需在 fund_method 为字符串时调用（见下方）
        if hasattr(sfm_opts, 'fund_method') and isinstance(sfm_opts.fund_method, str):
            try:
                sfm_opts.fund_method = getattr(cv2, sfm_opts.fund_method)
            except AttributeError:
                print(f"⚠️ 警告: 无法找到fund_method {sfm_opts.fund_method}，使用默认值FM_RANSAC")
                sfm_opts.fund_method = cv2.FM_RANSAC
        
        # 创建SFM对象并运行
        # sfm.py的__init__方法会自动处理custom标定矩阵
        sfm = SFM(sfm_opts)
        sfm.Run()

        # 将 MSTAR 自动闭合的 mesh 缩放同步回 convert args（summary box 在 SfM 之后运行）
        if getattr(args, "sfm_mstar_auto_image_fill_match", False):
            ms = getattr(sfm, "sar_params", {}).get("mstar_computed_mesh_uniform_scale")
            if ms is not None:
                args.sfm_mesh_prior_final_uniform_scale = float(ms)
        
        print("✅ 自定义sfm.py重建完成")
        return True
        
    except Exception as e:
        log_error(f"自定义sfm.py重建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def _print_target_dimension_summary_digest(source_path: str, visualization_dir: str) -> None:
    """Print a compact digest of target_dimension_constraints_summary.txt."""
    summary_file = os.path.join(source_path, visualization_dir, "target_dimension_constraints_summary.txt")
    print("=" * 70)
    print("=== target dimension summary digest ===")
    print(f"source: {summary_file}")
    if not os.path.isfile(summary_file):
        print("missing summary file")
        print("=" * 70)
        return

    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except Exception as exc:
        print(f"read failed: {exc}")
        print("=" * 70)
        return

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("来源:"):
            print(s)
        elif s.startswith("方位角筛选:"):
            print(s)
        elif s.startswith("单张几何模式"):
            print(s)
        elif s.startswith("clean-axis L/W extraction:"):
            print(s)
        elif s.startswith("clean-axis L_median_px:"):
            print(s)
        elif s.startswith("clean-axis W_median_px:"):
            print(s)
        elif s.startswith("clean-axis L_med/W_med:"):
            print(s)
        elif s.startswith("clean-axis samples (0/90/270):"):
            print(s)
        elif s.startswith("  ") and " az=" in s and "used_px=" in s:
            print(s)
        elif "成功算出 target_dimensions:" in s:
            print(s)
        elif s.startswith("有效图像数量:"):
            print(s)

    in_rec = False
    for line in lines:
        s = line.strip()
        if s.startswith("【推荐的点云约束参数】"):
            in_rec = True
            print()
            print(s)
            continue
        if in_rec:
            if s.startswith("【") and "推荐的点云约束参数" not in s:
                break
            if s:
                print("  " + s)
    print("=" * 70)


if __name__ == "__main__":
    main()
