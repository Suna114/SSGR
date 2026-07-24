# config.py
import os
import argparse

# 全局阈值倍率因子（在parse_arguments函数解析后会被更新）
TARGET_THRESHOLD_FACTOR = 0.3      # 目标区域Harris阈值因子（默认值，解析后会被更新）
BACKGROUND_THRESHOLD_FACTOR = 300.0  # 背景区域Harris阈值因子（默认值，解析后会被更新）
TRANSITION_THRESHOLD_FACTOR = 0.5  # 过渡区域Harris阈值因子（默认值，解析后会被更新）

# ---------------------------------------------------------------------------
# Convert / SfM 预设包（原 sar/dataset_pose_defaults.py，集中维护）
# ---------------------------------------------------------------------------
from typing import Any, Dict, Optional, Sequence, Tuple

# 一键「大致几何模型 / 阴影比例盒」
GEOMETRY_PRIOR_ROUGH_MODEL_PRESET: Dict[str, Any] = {
    "disable_cyclic_depression": True,
    "sar_camera_distribution_mode": "ring",
    "target_dimension_summary_azimuth_mode": "mstar_axis_split",
    "shadow_geometry_azimuth_mode": "mstar_axis_split",
    "convert_use_standalone_shadow_target_dimensions": True,
    "sfm_mesh_prior_sar_imaging_scale_match": False,
    "sfm_mesh_prior_use_mstar_pixel_resolution": True,
    "sfm_mstar_init_from_pixel_resolution": True,
    "sfm_mstar_focal_scale": 0.5,
    "sfm_mstar_range_resolution_m": 0.3,
    "sfm_mstar_azimuth_resolution_m": 0.3,
    "sfm_sorc_height_only_search_ratio": 0.35,
    "post_sfm_summary_box_align_camera_centers": False,
    "sfm_mesh_prior_nominal_coarse_mode": "ground_contact",
    "sfm_center_scene_anchor": "ground_contact",
    "sfm_mesh_prior_final_uniform_scale": 1.0,
    "sfm_summary_box_mesh_prior_mode": "replace",
    "sfm_summary_box_surface_mode": "box",
    "sfm_mesh_prior_sample_count": 10_000,
    "post_sfm_mesh_sample_count": 10_000,
    "sfm_mesh_prior_poisson_disk": True,
    "sfm_mesh_prior_vertex_axis_sign": "1,-1,1",
    "sfm_mesh_prior_extra_world_yaw_deg": -90.0,
    "sfm_mesh_prior_save_transform_json": "geometry_prior/sfm_summary_transform.json",
    "target_percentile": 95.0,
    "target_mask_min_component_area_px": 15,
    "target_dimension_filter_percentile": 99.0,
}

GEOMETRY_PRIOR_ROUGH_MODEL_CLI_FLAGS: Dict[str, Tuple[str, ...]] = {
    "disable_cyclic_depression": ("--disable_cyclic_depression",),
    "sar_camera_distribution_mode": ("--sar_camera_distribution_mode",),
    "target_dimension_summary_azimuth_mode": ("--target_dimension_summary_azimuth_mode",),
    "shadow_geometry_azimuth_mode": ("--shadow_geometry_azimuth_mode",),
    "convert_use_standalone_shadow_target_dimensions": (
        "--convert_use_standalone_shadow_target_dimensions",
    ),
    "sfm_mesh_prior_sar_imaging_scale_match": ("--sfm_mesh_prior_sar_imaging_scale_match",),
    "sfm_mesh_prior_use_mstar_pixel_resolution": ("--sfm_mesh_prior_use_mstar_pixel_resolution",),
    "sfm_mstar_init_from_pixel_resolution": ("--sfm_mstar_init_from_pixel_resolution",),
    "sfm_mstar_focal_scale": ("--sfm_mstar_focal_scale",),
    "sfm_sorc_height_only_search_ratio": ("--sfm_sorc_height_only_search_ratio",),
    "post_sfm_summary_box_align_camera_centers": ("--post_sfm_summary_box_align_camera_centers",),
    "sfm_mesh_prior_nominal_coarse_mode": ("--sfm_mesh_prior_nominal_coarse_mode",),
    "sfm_center_scene_anchor": ("--sfm_center_scene_anchor",),
    "sfm_mesh_prior_final_uniform_scale": ("--sfm_mesh_prior_final_uniform_scale",),
    "sfm_summary_box_mesh_prior_mode": ("--sfm_summary_box_mesh_prior_mode",),
    "sfm_summary_box_surface_mode": ("--sfm_summary_box_surface_mode",),
    "sfm_mesh_prior_sample_count": ("--sfm_mesh_prior_sample_count",),
    "post_sfm_mesh_sample_count": ("--post_sfm_mesh_sample_count",),
    "sfm_mesh_prior_poisson_disk": (
        "--sfm_mesh_prior_poisson_disk",
        "--sfm_mesh_prior_uniform_sample",
    ),
    "sfm_mesh_prior_vertex_axis_sign": ("--sfm_mesh_prior_vertex_axis_sign",),
    "sfm_mesh_prior_extra_world_yaw_deg": ("--sfm_mesh_prior_extra_world_yaw_deg",),
    "sfm_mesh_prior_save_transform_json": ("--sfm_mesh_prior_save_transform_json",),
    "target_percentile": ("--target_percentile",),
    "target_mask_min_component_area_px": ("--target_mask_min_component_area_px",),
    "target_dimension_filter_percentile": ("--target_dimension_filter_percentile",),
}

# 3DGS 训练适配：在 rough_model 基础上追加
GEOMETRY_PRIOR_GS_MODEL_PRESET: Dict[str, Any] = {
    "sfm_use_optical_equivalent_depression": True,
    "sfm_ba_fix_angles_height_only": True,
    "sfm_export_gs_satellite_poses": True,
    "sfm_sorc_use_gpu": True,
    "sfm_sorc_height_only_max_iterations": 15,
    # 光学透视专用：不用 SAR 斜距 auto_fill（尺度由 optical verify 闭合）
    "sfm_gs_optical_perspective_only": True,
    "sfm_mstar_auto_image_fill_match": False,
    "sfm_mstar_auto_image_fill_all_views": False,
    # 圆环方位 -1：俯视光学下目标随 φ 旋转方向与 MSTAR 图一致
    "sfm_gs_optical_ring_azimuth_sign": -1,
    # 比例盒底面中心固定原点，不跟随 SO-RCG sparse footprint 平移
    "sfm_mesh_prior_nominal_coarse_mode": "origin_ground_contact",
    # SAR-like 目标散射图更接近平面散射支撑；避免六面盒的侧面/顶面进入 3DGS 初始化形成低亮度膜。
    "sfm_summary_box_surface_mode": "plane",
    # 目标 3D 米制固定（BTR70 W≈4.95 m）；L 由 summary L/W 校正，画幅占比仅调 camera_height
    "sfm_mstar_fixed_physical_target_dims": True,
    "sfm_mstar_fixed_physical_snap_lw_from_summary": True,
    "sfm_mstar_vehicle_preset": "generic",
    # BTR70-15：文件名 φ_sar=15°（侧视）→ 光学圆环 φ_render=75°（俯视补角 90°−φ_sar）
    "sfm_optical_equivalent_depression_method": "complement_overhead",
    # gs_satellite_poses 后按 optical verify ru/rv≈1 自动求 sfm_mstar_camera_height_scale
    "sfm_mstar_optical_envelope_auto_scale": True,
    "sfm_mstar_optical_envelope_tune_mesh": True,
    "sfm_mstar_optical_envelope_target_ratio": 1.02,
    "sfm_mstar_optical_envelope_min_ratio": 1.0,
    "sfm_mstar_optical_envelope_mesh_min": 0.85,
    "sfm_mstar_optical_envelope_mesh_max": 2.5,
    "sfm_mstar_optical_envelope_undersize_weight": 2.5,
    "sfm_mstar_optical_side_u_refine": True,
    "convert_auto_verify_optical": True,
}

GEOMETRY_PRIOR_GS_MODEL_CLI_FLAGS: Dict[str, Tuple[str, ...]] = {
    "sfm_use_optical_equivalent_depression": ("--sfm_use_optical_equivalent_depression",),
    "sfm_optical_equivalent_depression_method": ("--sfm_optical_equivalent_depression_method",),
    "sfm_ba_fix_angles_height_only": ("--sfm_ba_fix_angles_height_only",),
    "sfm_export_gs_satellite_poses": (
        "--sfm_export_gs_satellite_poses",
        "--no-sfm_export_gs_satellite_poses",
    ),
    "sfm_sorc_use_gpu": ("--sfm_sorc_use_gpu",),
    "sfm_sorc_height_only_max_iterations": ("--sfm_sorc_height_only_max_iterations",),
    "sfm_mstar_auto_image_fill_match": ("--sfm_mstar_auto_image_fill_match",),
    "sfm_mstar_auto_image_fill_all_views": ("--sfm_mstar_auto_image_fill_all_views",),
    "sfm_mstar_fixed_physical_target_dims": ("--sfm_mstar_fixed_physical_target_dims",),
    "sfm_mstar_fixed_physical_snap_lw_from_summary": (
        "--sfm_mstar_fixed_physical_snap_lw_from_summary",
        "--no_sfm_mstar_fixed_physical_snap_lw_from_summary",
    ),
    "sfm_mstar_vehicle_preset": ("--sfm_mstar_vehicle_preset",),
    "sfm_mstar_optical_envelope_auto_scale": (
        "--sfm_mstar_optical_envelope_auto_scale",
        "--no_sfm_mstar_optical_envelope_auto_scale",
    ),
    "sfm_mstar_optical_envelope_tune_mesh": (
        "--sfm_mstar_optical_envelope_tune_mesh",
        "--no_sfm_mstar_optical_envelope_tune_mesh",
    ),
    "sfm_mstar_optical_envelope_target_ratio": ("--sfm_mstar_optical_envelope_target_ratio",),
    "sfm_mstar_optical_envelope_min_ratio": ("--sfm_mstar_optical_envelope_min_ratio",),
    "sfm_mstar_optical_envelope_mesh_min": ("--sfm_mstar_optical_envelope_mesh_min",),
    "sfm_mstar_optical_envelope_mesh_max": ("--sfm_mstar_optical_envelope_mesh_max",),
    "sfm_mstar_optical_envelope_undersize_weight": (
        "--sfm_mstar_optical_envelope_undersize_weight",
    ),
    "sfm_gs_optical_perspective_only": (
        "--sfm_gs_optical_perspective_only",
        "--no_sfm_gs_optical_perspective_only",
        "--sfm_gs_sar_slant_imaging",
    ),
    "sfm_gs_optical_ring_azimuth_sign": ("--sfm_gs_optical_ring_azimuth_sign",),
    "sfm_summary_box_surface_mode": ("--sfm_summary_box_surface_mode",),
    "sfm_mstar_optical_side_u_refine": (
        "--sfm_mstar_optical_side_u_refine",
        "--no_sfm_mstar_optical_side_u_refine",
    ),
    "convert_auto_verify_optical": (
        "--convert_auto_verify_optical",
        "--no_convert_auto_verify_optical",
    ),
}

# convert → sfm.py 需挂到 opts 上的 CLI 属性名（与 argparse dest 一致）
SFM_OPTS_ATTR_NAMES: Tuple[str, ...] = (
    "sfm_mstar_init_from_pixel_resolution",
    "sfm_mstar_focal_scale",
    "sfm_mstar_range_resolution_m",
    "sfm_mstar_azimuth_resolution_m",
    "sfm_mstar_camera_height_scale",
    "sfm_mstar_auto_image_fill_match",
    "sfm_mstar_auto_image_fill_all_views",
    "sfm_mstar_fixed_physical_target_dims",
    "sfm_mstar_target_length_m",
    "sfm_mstar_target_width_m",
    "sfm_mstar_target_height_m",
    "sfm_mstar_target_height_boost",
    "sfm_mstar_vehicle_preset",
    "sfm_mstar_fill_ref_azimuth_deg",
    "sfm_mstar_default_length_px",
    "sfm_mstar_default_width_px",
    "sfm_sorc_height_only_search_ratio",
    "sfm_mesh_prior_use_mstar_pixel_resolution",
    "sfm_mesh_prior_nominal_coarse_mode",
    "sfm_summary_box_surface_mode",
    "sfm_center_scene_anchor",
    "sfm_center_scene_ground_y_percentile",
    "sfm_ba_fix_angles_height_only",
    "sfm_use_optical_equivalent_depression",
    "sfm_optical_equivalent_depression_method",
    "sfm_optical_equivalent_depression_scale",
    "sfm_gs_optical_depression_horizon_deg",
    "sfm_mstar_optical_side_u_refine",
    "sfm_mstar_optical_side_u_tol",
    "sfm_mstar_optical_envelope_target_ratio",
    "sfm_mstar_optical_envelope_min_ratio",
    "sfm_mstar_optical_envelope_mesh_min",
    "sfm_mstar_optical_envelope_mesh_max",
    "sfm_mstar_optical_envelope_undersize_weight",
    "sfm_export_gs_satellite_poses",
    "sfm_center_scene_after_prior_replace",
    "sfm_use_sorc",
    "sfm_sorc_max_iterations",
    "sfm_sorc_height_only_max_iterations",
    "sfm_sorc_tolerance",
    "sfm_sorc_use_5dof",
    "sfm_sorc_num_threads",
    "sfm_sorc_use_gpu",
    "sfm_sorc_gpu_batch_size",
    "sfm_enable_position_constraint",
    "sfm_position_tolerance_ratio",
    "sfm_enable_depression_constraint",
    "sfm_depression_min",
    "sfm_depression_max",
    "sfm_enable_point_height_constraint",
    "sfm_point_height_constraint_mode",
    "sfm_enable_point_width_constraint",
    "disable_cyclic_depression",
    "sfm_disable_auto_height_only",
    "geometry_prior_gs_model",
    "geometry_prior_rough_model",
    "sar_camera_height",
    "sar_scene_scale",
    "sar_mstar_filename_depression_convention",
    "sar_camera_distribution_mode",
)


def _cli_flag_specified(argv: Sequence[str], option_names: Sequence[str]) -> bool:
    for tok in argv:
        for name in option_names:
            if tok == name or tok.startswith(f"{name}="):
                return True
    return False


def apply_geometry_prior_rough_model_preset(args: Any, argv: Optional[Sequence[str]] = None) -> None:
    """``--geometry_prior_rough_model``：阴影比例盒 + MSTAR 像素闭合初值等。"""
    if not bool(getattr(args, "geometry_prior_rough_model", False)):
        return
    if argv is None:
        import sys
        argv = sys.argv
    applied: list[str] = []
    for key, val in GEOMETRY_PRIOR_ROUGH_MODEL_PRESET.items():
        flags = GEOMETRY_PRIOR_ROUGH_MODEL_CLI_FLAGS.get(key, ())
        if flags and _cli_flag_specified(argv, flags):
            continue
        setattr(args, key, val)
        applied.append(key)
    if not bool(getattr(args, "convert_output_path_only", False)):
        print(
        "\n[geometry_prior_rough_model] 已启用大致几何模型 / 阴影比例盒先验预设\n"
        "  ring + 90/270 侧视 L/W + MSTAR 像素米制比例盒 + replace；表面采样默认 10000 点\n"
        f"  已应用 {len(applied)} 项；显式 CLI 参数不会被覆盖\n"
        "  示例: python convert.py -s <数据> --geometry_prior_rough_model\n"
        )


def apply_geometry_prior_gs_model_preset(args: Any, argv: Optional[Sequence[str]] = None) -> None:
    """``--geometry_prior_gs_model``：rough_model + 3DGS 圆环位姿 + height-only SO-RCG。"""
    if not bool(getattr(args, "geometry_prior_gs_model", False)):
        return
    if argv is None:
        import sys
        argv = sys.argv
    if not bool(getattr(args, "geometry_prior_rough_model", False)):
        args.geometry_prior_rough_model = True
    apply_geometry_prior_rough_model_preset(args, argv)
    applied: list[str] = []
    for key, val in GEOMETRY_PRIOR_GS_MODEL_PRESET.items():
        flags = GEOMETRY_PRIOR_GS_MODEL_CLI_FLAGS.get(key, ())
        if flags and _cli_flag_specified(argv, flags):
            continue
        setattr(args, key, val)
        applied.append(key)
    if not bool(getattr(args, "convert_output_path_only", False)):
        print(
        "\n[geometry_prior_gs_model] 已启用 3DGS 光学透视预设\n"
        "  含 rough_model + 俯视针孔(complement_overhead: 15°→75°) + 圆环 COLMAP 外参\n"
        "  不用 SAR 斜距 auto_fill；尺度由 optical verify 闭合；固定 W + summary L/W 校正 L\n"
        f"  额外应用 {len(applied)} 项\n"
        "  示例: python convert.py -s data/360-1 --geometry_prior_gs_model\n"
        )


def apply_config_presets(args: Any, argv: Optional[Sequence[str]] = None) -> None:
    """parse_arguments 末尾：应用全部 geometry prior 预设。"""
    apply_geometry_prior_rough_model_preset(args, argv)
    apply_geometry_prior_gs_model_preset(args, argv)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SAR-SIFT 3D Reconstruction')
    # GPU相关参数
    parser.add_argument("--gpu_device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="GPU设备选择")
    parser.add_argument("--no_gpu", action='store_true', help="完全禁用GPU")
    parser.add_argument("--use_cpu", action='store_true', help="强制使用CPU")
    parser.add_argument("--gpu_backend", type=str, default="pytorch", choices=["cupy", "pytorch"], help="GPU加速后端")

    # 处理流程控制
    parser.add_argument(
        "--convert_output_path_only",
        dest="convert_output_path_only",
        action="store_true",
        default=False,
        help="静默运行 convert.py，终端仅输出最终数据目录；默认关闭，便于调试。",
    )
    parser.add_argument(
        "--no_convert_output_path_only",
        "--no-convert_output_path_only",
        dest="convert_output_path_only",
        action="store_false",
        help="恢复 convert.py 详细终端日志（默认行为）。",
    )
    parser.add_argument(
        "--skip_matching",
        action="store_true",
        help="跳过特征提取、匹配与 SfM 重建，复用 sparse/0 点云，仅重跑比例盒/圆环外参/post_sfm",
    )
    parser.add_argument(
        "--skip_sfm",
        action="store_true",
        help="跳过 SfM（SO-RCG）重建，复用上次 sparse/0 点云；仍运行特征匹配与后续几何步骤",
    )
    parser.add_argument("--source_path", "-s", required=True, type=str, help="源数据路径")
    # SAR图像专用：强制使用SAR-SIFT特征
    parser.add_argument("--camera", default="OPENCV", type=str, help="相机模型（SAR图像使用SAR几何模型）")

    # 外部工具路径（仅保留ImageMagick用于图像缩放）
    parser.add_argument("--magick_executable", default="", type=str, help="ImageMagick可执行文件路径（用于图像缩放）")

    # 图像处理参数
    parser.add_argument("--resize", action="store_true", help="调整图像大小")
    parser.add_argument("--image_scale", type=float, default=1.0, help="图像缩放因子")
    parser.add_argument("--preprocess_sar", action="store_true", help="SAR图像预处理")
    parser.add_argument("--speckle_filter", action="store_true", help="应用斑点噪声滤波")
    parser.add_argument("--contrast_enhance", action="store_true", help="应用对比度增强")

    # SAR几何参数
    parser.add_argument("--sar_height", type=float, default=10.0, help="SAR图像飞行高度")
    parser.add_argument("--target_height", type=float, default=3.0, help="目标高度")
    parser.add_argument("--target_base_height", type=float, default=0.0, help="目标基础高度")
    parser.add_argument("--shadow_height", type=float, default=0.0, help="阴影高度")
    parser.add_argument("--background_height", type=float, default=0.0, help="背景高度")
    parser.add_argument("--point_cloud_depth_factor", type=float, default=2.0, help="点云深度范围因子（相对于camera_height的倍数，默认2.0，增大可增加点云厚度）")
    parser.add_argument("--target_density", type=float, default=1000, help="目标点云密度")
    parser.add_argument("--background_density", type=float, default=0.5, help="背景点云密度")
    parser.add_argument("--point_cloud_scale", type=float, default=3.0, help="点云缩放因子")
    parser.add_argument("--depression_angle_scale", type=float, default=1.0, help="俯视角缩放因子")
    parser.add_argument(
        "--sar_mstar_filename_depression_convention",
        type=str,
        default="horizon",
        choices=["horizon", "zenith"],
        help=(
            "MSTAR 式文件名第二段俯视角语义：**horizon（默认）**=相对水平面向下的**正常俯角 φ**（OSU/MSTAR 惯例），"
            "内部再折合为与天底夹角 θ=90°-φ 供球面/斜距公式使用。"
            "**zenith**：文件名第二段**已是**与天底入射角 θ，不再做 90°−φ 换算（仅当数据实为 θ 时使用）。"
        ),
    )
    parser.add_argument("--disable_cyclic_depression", action='store_true', default=False,
                       help="禁用循环俯视角模式（默认启用，使用15°→45°→15°循环，用于解决三角化退化问题）")
    parser.add_argument("--cyclic_depression_start", type=float, default=15.0,
                       help="循环俯视角起始角度（度），默认15°（MSTAR数据集主要俯视角）")
    parser.add_argument("--cyclic_depression_end", type=float, default=45.0,
                       help="循环俯视角结束角度（度），默认45°（MSTAR数据集扩展俯视角）")
    parser.add_argument("--cyclic_depression_step", type=float, default=15.0,
                       help="循环俯视角角度步长（度），默认15°（MSTAR数据集：15°、30°、45°）")
    parser.add_argument(
        "--sar_camera_distribution_mode",
        type=str,
        default="ring",
        choices=["ring", "sphere"],
        help="SAR 相机空间分布：ring=水平圆环（固定高度绕目标，360° 方位/卫星轨迹推荐）；"
        "sphere=球面分布（多俯角或启用循环俯视角时用）。",
    )

    parser.add_argument(
        "--sfm_use_optical_equivalent_depression",
        action="store_true",
        default=False,
        help="SO-RCG/SAR BA 前：用 SAR 文件名俯角换算光学等效俯角作初始俯仰，并强制 ring 分布、禁用循环俯视角",
    )
    parser.add_argument(
        "--sfm_optical_equivalent_depression_method",
        type=str,
        default="ring_look_direction",
        choices=[
            "complement_overhead",
            "ring_look_direction",
            "pixel_spacing_anisotropy",
            "slant_range_view_distance",
            "empirical_scale",
            "sphere_to_ring_elevation",
            "identity",
        ],
        help=(
            "光学/COLMAP 圆环位姿俯角（horizon 惯例，度）。"
            "complement_overhead：φ_render=90°−φ_sar（15° SAR 侧视→75° 光学俯视）；"
            "ring_look_direction：与文件名一致（侧视）。"
            "见 sar/optical_equivalent_depression.py"
        ),
    )
    parser.add_argument(
        "--sfm_optical_equivalent_depression_scale",
        type=float,
        default=1.0,
        help="empirical_scale / pixel_spacing_anisotropy 方法的 tan 缩放因子",
    )
    parser.add_argument(
        "--sfm_gs_optical_depression_horizon_deg",
        type=float,
        default=None,
        help=(
            "指定 COLMAP/GS 渲染用 horizon 俯角（度），覆盖文件名与 complement_overhead。"
            "MSTAR BTR70-15 常用 75（俯视）；勿与天底入射角混淆（天底 75°=horizon 15° 侧视）。"
            "需配合 --sfm_use_optical_equivalent_depression。"
        ),
    )

    # 特征提取参数
    parser.add_argument("--feature_type", type=str, default="SAR_SIFT", choices=["SAR_SIFT"], help="特征提取器类型")
    parser.add_argument("--max_features", type=int, default=200000, help="最大特征点数量（SAR图像建议200000，增加匹配点数量）")
    parser.add_argument("--sar_sift_sigma", type=float, default=1.5, help="SAR-SIFT初始尺度")
    parser.add_argument("--sar_sift_ratio", type=float, default=2 ** (1 / 3.), help="SAR-SIFT尺度比率")
    parser.add_argument("--sar_sift_layers", type=int, default=16, help="SAR-SIFT尺度层数")
    parser.add_argument("--sar_sift_d", type=float, default=0.01, help="SAR-SIFT Harris参数（降低以提取更多特征点）")
    parser.add_argument("--sar_sift_harris_threshold", type=float, default=0.05, help="SAR-SIFT Harris函数阈值（SAR图像建议0.2，降低以提取更多特征点）")

    # 匹配参数（SAR图像匹配困难，使用更宽松的参数）
    parser.add_argument("--max_ratio", type=float, default=0.99, help="匹配最大比率（SAR图像建议0.98，保留更多候选匹配）")
    parser.add_argument("--ransac_iterations", type=int, default=15000, help="RANSAC迭代次数（SAR图像建议15000，增加以找到更多内点）")
    parser.add_argument("--ransac_error_threshold", type=float, default=12.0, help="RANSAC误差阈值（SAR图像建议12.0像素，放宽以保留更多匹配点）")
    parser.add_argument("--min_matches_required", type=int, default=2, help="最小匹配点数要求（降低以保留更多匹配对，默认2）")
    parser.add_argument("--use_relaxed_matching", action='store_true', default=True, help="使用宽松匹配模式（默认启用，RANSAC失败时保留原始匹配）")
    parser.add_argument("--max_pairs_per_image", type=int, default=0.0, help="每张图像最多匹配的图像数（0表示全局匹配所有图像对，推荐用于SAR图像）")
    parser.add_argument("--max_azimuth_diff", type=float, default=0.0, help="最大方位角差异（0表示禁用几何约束，推荐用于SAR图像以增加匹配点）")
    parser.add_argument("--use_bidirectional_matching", action='store_true', default=True, help="使用双向匹配（默认启用，可提高匹配点数量和质量）")
    parser.add_argument("--matching_method", type=str, default="bidirectional",
                       choices=["bidirectional", "symmetry"], help="双向匹配策略")

    # 并行处理参数
    parser.add_argument("--num_threads", type=int, default=-1, help="特征提取线程数")
    # 注意：action='store_true'的参数默认是False，需要通过代码逻辑实现默认True
    parser.add_argument("--use_parallel_match", action='store_true', help="使用并行匹配（默认启用，可通过--disable_parallel_match禁用）")
    parser.add_argument("--disable_parallel_match", action='store_true', help="禁用并行匹配（使用单进程模式）")
    parser.add_argument("--parallel_workers", type=int, default=4, help="并行匹配工作进程数（默认4）")
    parser.add_argument("--use_processes", action='store_true', help="使用进程而不是线程")

    # 可视化参数
    parser.add_argument('--save_feature_images', action='store_true', default=True,
                       help='保存特征点图像')
    parser.add_argument('--save_match_images', action='store_true', default=True,
                       help='保存匹配图像')
    parser.add_argument("--no_save_feature_images", action='store_true', help="不保存特征点可视化图像")
    parser.add_argument("--no_save_match_images", action='store_true', help="不保存特征点匹配可视化图像")
    parser.add_argument("--max_feature_images", type=int, default=200, help="最多保存的特征点图像数量")
    parser.add_argument("--max_match_pairs", type=int, default=200, help="最多保存的匹配图像对数")
    parser.add_argument("--save_visualization", action='store_true', default=True, help="保存阈值/增强可视化")
    parser.add_argument("--visualization_dir", type=str, default="adaptive_threshold_visualization",
                       help="自适应阈值可视化图像保存目录")
    parser.add_argument(
        "--shadow_geometry_azimuth_mode",
        type=str,
        default="oblique",
        choices=["oblique", "cardinal", "mstar_axis_split", "any"],
        help="单张 SAR 阴影几何 target_dimensions：oblique=非正侧视(15°步进方位、排除0/90/180/270附近)，"
        "阴影长度按方位在距离向/方位向包络上混合，γ 由目标方位角推导（减轻叠掩—阴影粘连）；"
        "cardinal=沿用原 0/90/180/270 门控与分档阴影，并用 Tr/Taz 解 L/W；"
        "mstar_axis_split=MSTAR：仅 **90°/270°**（±容差）；**L=Tr、W=Taz**（包围盒两轴向跨度）；"
        "H 仍由阴影（若掩码含阴影）；any=不做方位门控。",
    )
    parser.add_argument(
        "--target_dimension_summary_azimuth_mode",
        type=str,
        default="oblique",
        choices=["near_zero", "cardinal", "mstar_axis_split", "diagonal", "oblique"],
        help="target_dimension_constraints_summary（L/W/H 像素中位数汇总）所用方位角筛选："
        "oblique=与 shadow_geometry 斜视门控一致(15°步进、排除 cardinal)；"
        "near_zero=仅 0°/360°±容差；cardinal=0/90/180/270/360°±容差；"
        "mstar_axis_split=与 mstar_axis_split 配套（**仅 90°/270°±容差**，L=Tr、W=Taz）；"
        "diagonal=45/135/225/315°±容差。",
    )
    parser.add_argument(
        "--target_dimension_summary_azimuth_tolerance_deg",
        type=float,
        default=5.0,
        help="与 --target_dimension_summary_azimuth_mode 配合的方位角容差（度）",
    )
    parser.add_argument(
        "--target_dimension_filter_percentile",
        type=float,
        default=99.0,
        help=(
            "standalone target_dimension 汇总在提取 L/W 前使用的背景过滤分位数；"
            "复用 utils/filter_background.py 的 percentile+largest_cc 目标过滤逻辑。"
        ),
    )
    parser.add_argument(
        "--convert_use_standalone_shadow_target_dimensions",
        action="store_true",
        default=False,
        help=        "convert 特征阶段不逐张计算 target_dimensions（加快多进程、避免与汇总逻辑重复）；"
        "在「特征提取尚未开始时」先完成 standalone 批处理并写入 "
        "`visualization_dir/target_dimension_constraints_summary.txt`（含 H/L/W 像素中位数行，供 "
        "`sfm_summary_box_mesh_prior` before_filter 解析），随后再进行 SAR-SIFT 与 SfM。"
        "单张几何与筛选仍由 --shadow_geometry_azimuth_mode 与 --target_dimension_summary_azimuth_mode 控制。",
    )
    parser.add_argument(
        "--no_standalone_shadow_robust_filter",
        action="store_true",
        default=False,
        help="与 --convert_use_standalone_shadow_target_dimensions 配合：关闭批处理内的 MAD 等鲁棒剔除，"
        "仅用「几何成功 + 方位筛选」样本的中位数。",
    )
    parser.add_argument(
        "--standalone_shadow_mad_k",
        type=float,
        default=4.0,
        help="standalone 批处理鲁棒剔除：像素 L/W/H 的 MAD 倍数；≤0 关闭软门控；仅在与上项未 --no_standalone_shadow_robust_filter 时生效。",
    )

    # SAR SfM参数（此代码专用于SAR图像，使用自定义sfm.py）
    parser.add_argument("--save_features_for_sfm", action='store_true', default=True, help="保存特征为pickle格式（用于sfm.py）")
    parser.add_argument("--save_matches_for_sfm", action='store_true', default=True, help="保存匹配为pickle格式（用于sfm.py）")
    
    # sfm.py相关参数
    parser.add_argument("--sfm_data_dir", type=str, default="", help="sfm.py的数据根目录（默认使用source_path）")
    parser.add_argument("--sfm_dataset", type=str, default="", help="sfm.py的数据集名称（默认使用source_path的basename）")
    parser.add_argument("--sfm_features", type=str, default="SAR_SIFT", help="sfm.py的特征名称")
    parser.add_argument("--sfm_matcher", type=str, default="SAR_SIFT", help="sfm.py的匹配器名称")
    parser.add_argument("--sfm_ext", type=str, default="jpg,jpeg,png,tif,tiff,bmp", help="sfm.py支持的图像扩展名（逗号分隔）")
    parser.add_argument("--sfm_out_dir", type=str, default="", help="sfm.py的输出目录（默认使用source_path/sfm_results）")
    parser.add_argument("--sfm_cross_check", action='store_true', default=True, help="sfm.py匹配交叉检查")
    parser.add_argument("--use_sar_geometry", action='store_true', default=True, help="使用SAR几何模型进行SfM重建（SAR图像默认启用）")
    
    # SAR系统参数（可选，使用默认值如果未指定）
    # 注意：sar_camera_height是实际SAR系统参数，sar_height是用于合成点云的参数，两者用途不同
    parser.add_argument("--sar_camera_height", type=float, default=12000.0, 
                       help="MSTAR **物理**平台高度 (m)，仅作对照/非 MSTAR 路径；"
                            "启用 --geometry_prior_gs_model 时场景 H 由像素分辨率公式闭合，勿与 COLMAP 场景 H 混用。")
    parser.add_argument("--sar_scene_scale", type=float, default=1,
                       help="SAR场景缩放因子 - 控制相机到目标的距离。"
                            "值越小相机越近：0.1适合高斯泼溅（相机距离~12.8），1.0中等，10.0较远。"
                            "默认0.1，适合128x128图像的高斯泼溅训练。")
    parser.add_argument("--sar_radius_scale", type=float, default=None,
                       help="SAR相机半径缩放因子 - 如果未指定（None），将自动根据图像尺寸计算。"
                            "自动计算公式: radius_scale = (图像尺寸 × scene_scale) / R0")
    parser.add_argument("--sar_platform_velocity", type=float, default=250.0)
    parser.add_argument("--sar_prf", type=float, default=2500.0, 
                       help="SAR脉冲重复频率 (Hz) - 典型范围500-5000 Hz。"
                            "低分辨率500-1500 Hz，中等分辨率1500-3000 Hz，高分辨率3000-5000 Hz。"
                            "默认2500.0 Hz（MSTAR 数据集典型值）")
    parser.add_argument("--sar_bandwidth", type=float, default=500e6, 
                       help="SAR带宽 (Hz) - 典型范围50-500 MHz。"
                            "X波段100-500 MHz，C波段50-200 MHz，L波段50-150 MHz。"
                            "星载SAR通常20-100 MHz。"
                            "默认500 MHz（MSTAR数据集，基于距离向分辨率0.3m计算）")
    parser.add_argument("--sar_image_size_azimuth", type=int, default=512, help="SAR图像方位向采样点数")
    parser.add_argument("--sar_image_size_range", type=int, default=512, help="SAR图像距离向采样点数")
    parser.add_argument("--sfm_fund_method", type=str, default="FM_RANSAC", help="sfm.py基础矩阵估计方法")
    parser.add_argument("--sfm_outlier_thres", type=float, default=3.0, help="sfm.py异常值阈值（SAR图像建议3.0，增大可保留更多匹配点）")
    parser.add_argument("--sfm_fund_prob", type=float, default=0.3, help="sfm.py基础矩阵估计置信度（SAR图像建议0.3，降低可保留更多匹配点）")
    parser.add_argument("--sfm_pnp_method", type=str, default="SOLVEPNP_DLS", help="sfm.py PnP方法")
    parser.add_argument("--sfm_pnp_prob", type=float, default=0.9, help="sfm.py PnP置信度")
    parser.add_argument("--sfm_reprojection_thres", type=float, default=12.0, help="sfm.py重投影误差阈值（增大可保留更多点）")
    parser.add_argument("--sfm_plot_error", action='store_true', default=False, help="sfm.py绘制重投影误差图")
    
    # 点云密度控制参数
    parser.add_argument("--sfm_min_matches_for_triangulation", type=int, default=1, help="三角化所需的最小匹配点数（SAR图像建议1，降低可增加点云密度）")
    parser.add_argument("--sfm_min_inliers_for_triangulation", type=int, default=1, help="三角化所需的最小内点数（SAR图像建议1，降低可增加点云密度）")
    parser.add_argument("--sfm_dense_init", action='store_true', default=False,
                       help="使用密集初始化（在稀疏点云基础上添加规则网格点）")
    parser.add_argument("--sfm_dense_grid_size", type=int, default=16,
                       help="密集初始化网格大小（默认16，即16x16=256个额外点）")
    parser.add_argument("--sfm_skip_fundamental_matrix", action='store_true', default=True, help="跳过基础矩阵验证（SAR图像匹配困难，默认启用）")
    parser.add_argument("--sfm_filter_behind_camera", action='store_true', default=True, help="是否过滤相机后方的点（默认启用，符合COLMAP要求：深度必须为正）")
    parser.add_argument("--sfm_merge_point_threshold", type=float, default=1.0, help="合并相似3D点的距离阈值（米），小于此距离的点会被合并")
    parser.add_argument("--sfm_build_complete_tracks", action='store_true', default=True, help="从所有匹配数据构建完整的track信息（默认启用，提高track长度）")
    parser.add_argument("--sfm_default_point_color_r", type=int, default=50, help="默认点云颜色R分量（0-255），当无法从图像提取颜色时使用")
    parser.add_argument("--sfm_default_point_color_g", type=int, default=50, help="默认点云颜色G分量（0-255）")
    parser.add_argument("--sfm_default_point_color_b", type=int, default=200, help="默认点云颜色B分量（0-255），深蓝色在白色背景上更易观察")

    parser.add_argument(
        "--sfm_mesh_prior_obj",
        type=str,
        default="",
        help="几何先验网格（.obj/.ply/.stl/.off），相对路径相对 --source_path。非空且 mode≠off 时在 SfM 内对齐并表面采样",
    )
    parser.add_argument(
        "--sfm_mesh_prior_mode",
        type=str,
        default="off",
        choices=["off", "replace", "merge"],
        help="replace：SO-RCG/可选 SAR BA 之后用网格表面点替换点云（稀疏三角化差时推荐）；merge：重投影过滤后追加表面点，保留稀疏 track",
    )
    parser.add_argument(
        "--sfm_mesh_prior_sample_count",
        type=int,
        default=120_000,
        help="SfM 网格先验：写入 COLMAP 的表面采样点数（裁剪后可能更少）；"
        "--geometry_prior_gs_model / --geometry_prior_rough_model 预设默认 10000；复杂 CAD 可 8e4~2e5",
    )
    parser.add_argument(
        "--sfm_mesh_prior_icp_source_samples",
        type=int,
        default=50_000,
        help="SfM 网格先验：用于对齐/ICP 的网格表面采样数",
    )
    parser.add_argument(
        "--sfm_mesh_prior_skip_icp",
        action="store_true",
        default=True,
        help="SfM 网格先验：默认跳过 ICP，仅用 AABB 粗相似对齐（稀疏点极少时更稳）。指定 --sfm_mesh_prior_use_icp 可开启ICP",
    )
    parser.add_argument(
        "--sfm_mesh_prior_use_icp",
        dest="sfm_mesh_prior_skip_icp",
        action="store_false",
        help="SfM 网格先验：启用多尺度 ICP 细化（需足够可信的稀疏点）",
    )
    parser.add_argument(
        "--sfm_mesh_prior_poisson_disk",
        action="store_true",
        default=True,
        help="SfM 网格先验：默认泊松盘表面采样；指定 --sfm_mesh_prior_uniform_sample 改为均匀采样",
    )
    parser.add_argument(
        "--sfm_mesh_prior_uniform_sample",
        dest="sfm_mesh_prior_poisson_disk",
        action="store_false",
        help="SfM 网格先验：三角形上均匀采样而非泊松盘",
    )
    parser.add_argument(
        "--sfm_mesh_prior_scale_override",
        type=float,
        default=0.0,
        help="SfM 网格先验：**粗对齐阶段**对顶点做**各向同性等比缩放** s（相似变换），把整个 CAD 按比例缩小/放大，**不是**按盒截断或删掉部分三角面。"
        "0 表示不使用。与 --sfm_mesh_prior_final_uniform_scale 不同（后者在粗对齐+ICP+yaw 之后才缩放）。",
    )
    parser.add_argument(
        "--sfm_mesh_prior_extra_world_yaw_deg",
        type=float,
        default=90.0,
        help="SfM/COLMAP世界系绕 +Y（竖直轴）右转角（度，右手法则）。对齐与ICP之后在写入采样的点之前施加，用于 CAD 轴线与 SAR 方位 0° / "
        "\"自 -Y 俯视\"语义约定相差固定偏航（常见 ±90）。若朝向已正确可设为 0。",
    )
    parser.add_argument(
        "--sfm_mesh_prior_final_uniform_scale",
        type=float,
        default=1.0,
        help="SfM网格先验：粗对齐+yaw 之后在**对齐目标质心**上做**相似缩放**（各向等比），**不是**裁掉网格一部分。"
        "1.0=不变；小于 1 则整体缩小，大于 1 则放大；≤0 或无效视为 1.0。（相机外参不变，只改写入点云尺度的网格顶点位置）",
    )
    parser.add_argument(
        "--sfm_mesh_prior_sar_imaging_scale_match",
        action="store_true",
        default=False,
        help=(
            "网格/阴影比例盒先验（SfM 内）：在 yaw 与 --sfm_mesh_prior_final_uniform_scale 之后，"
            "在参考质心处**各向等乘一个均匀比例 γ**（相似缩放），**不是**截断三角网格。"
            "用与 SAR 训练一致的近似斜距成像比较稀疏 vs 模型在参考视角下的像素包络。"
            " 默认参考方位见 --sfm_mesh_prior_sar_imaging_ref_azimuth_deg （默认90°）；俯角默认取训练视图俯角中位数。"
            " （需 sar.geometry；与 COLMAP/SfM 同尺度时请保证数据目录下 sar_scale_params.json，mstar_nosfm 会自动跳过合并）"
        ),
    )
    parser.add_argument(
        "--sfm_mesh_prior_sar_imaging_ref_azimuth_deg",
        type=float,
        default=90.0,
        help="上项 SAR 成像尺度匹配使用的参考方位角（度），默认 90。",
    )
    parser.add_argument(
        "--sfm_mesh_prior_sar_imaging_ref_depression_deg",
        type=float,
        default=None,
        help=(
            "SAR 成像尺度匹配的参考俯角（度）；默认省略则由 convert/SfM 用当前训练视图俯仰角序列的中位数。"
        ),
    )
    parser.add_argument(
        "--sfm_mesh_prior_use_mstar_pixel_resolution",
        action="store_true",
        default=False,
        help=(
            "阴影比例盒：用 MSTAR 像素分辨率（默认方位/距离均为 0.3 m/px，可改 --sfm_mstar_*_resolution_m）"
            "直接把汇总 L/W/H 换到世界米制，不再用稀疏点估计 depth_scale。"
        ),
    )
    parser.add_argument(
        "--sfm_mstar_init_from_pixel_resolution",
        action="store_true",
        default=False,
        help=(
            "SfM/SO-RCG 初始化：由 128px MSTAR 的 f、ρ、俯角公式闭合 camera_height 与 scene_scale，"
            "作为共享高度初值；SO-RCG 仅在 ±sfm_sorc_height_only_search_ratio 内微调（非事后改 COLMAP）。"
        ),
    )
    parser.add_argument(
        "--sfm_mstar_focal_scale",
        type=float,
        default=None,
        help="MSTAR 初值闭合用的 f = Na×该值（默认 0.5，即 128px→f=64）",
    )
    parser.add_argument(
        "--sfm_mstar_default_length_px",
        type=float,
        default=17.0,
        help="无阴影汇总时默认目标长度（px，方位向）",
    )
    parser.add_argument(
        "--sfm_mstar_default_width_px",
        type=float,
        default=38.0,
        help="无阴影汇总时默认目标宽度（px，距离向）",
    )
    parser.add_argument(
        "--sfm_sorc_height_only_search_ratio",
        type=float,
        default=0.35,
        help="固定角度仅优化高度时，在初值 H 的 ±该比例内搜索（默认 0.35 → [0.65H, 1.35H]）",
    )
    parser.add_argument(
        "--sfm_mstar_range_resolution_m",
        type=float,
        default=0.3,
        help="MSTAR 距离向像素分辨率 (m/px)，默认 0.3",
    )
    parser.add_argument(
        "--sfm_mstar_azimuth_resolution_m",
        type=float,
        default=0.3,
        help="MSTAR 方位向像素分辨率 (m/px)，默认 0.3（与距离向一致）",
    )
    parser.add_argument(
        "--sfm_mstar_camera_height_scale",
        type=float,
        default=None,
        help=(
            "MSTAR 闭合 H 之后再乘该因子（>1 拉远相机）。"
            "默认 None：在 --sfm_mstar_auto_image_fill_match 下由训练图 L/W 像素与 SAR 成像投影自动计算；"
            "显式指定则覆盖自动值及 optical verify 自动闭合。"
        ),
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_auto_scale",
        action="store_true",
        default=False,
        help=(
            "gs_satellite_poses 后迭代调整 camera_height_scale，使 optical verify 全角度 ru/rv≈1 "
            "（--geometry_prior_gs_model 默认开启）。"
        ),
    )
    parser.add_argument(
        "--no_sfm_mstar_optical_envelope_auto_scale",
        dest="sfm_mstar_optical_envelope_auto_scale",
        action="store_false",
        help="禁用 optical verify 自动尺度闭合，仅用 SAR auto_image_fill 或显式 --sfm_mstar_camera_height_scale。",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_scale_tol",
        type=float,
        default=0.05,
        help="optical verify 闭合目标：|ru−1|、|rv−1| 约 ±该比例（默认 0.05 → ±5%%）。",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_scale_min",
        type=float,
        default=0.35,
        help="optical 闭合搜索 sfm_mstar_camera_height_scale 下界（默认 0.35）。",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_scale_max",
        type=float,
        default=1.5,
        help="optical 闭合搜索 sfm_mstar_camera_height_scale 上界（默认 1.5）。",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_scale_max_iter",
        type=int,
        default=12,
        help="optical 闭合最大迭代次数（默认 12）。",
    )
    parser.add_argument(
        "--sfm_gs_optical_perspective_only",
        action="store_true",
        default=False,
        help=(
            "MSTAR 按俯视光学透视处理（针孔圆环），禁用 SAR 斜距 auto_fill。"
            "（--geometry_prior_gs_model 默认开启）"
        ),
    )
    parser.add_argument(
        "--no_sfm_gs_optical_perspective_only",
        dest="sfm_gs_optical_perspective_only",
        action="store_false",
        help="在 gs_model 下仍启用 SAR 斜距 auto_image_fill（与光学 verify 混用，不推荐）。",
    )
    parser.add_argument(
        "--sfm_gs_sar_slant_imaging",
        action="store_true",
        default=False,
        help="等价于 --no_sfm_gs_optical_perspective_only。",
    )
    parser.add_argument(
        "--sfm_gs_optical_ring_azimuth_sign",
        type=int,
        default=None,
        choices=(-1, 1),
        help=(
            "光学圆环外参方位乘子（±1）。默认 None：gs 光学透视为 -1（与 MSTAR 图旋转一致），"
            "SAR 斜距路径为 +1。"
        ),
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_tune_mesh",
        action="store_true",
        default=False,
        help="H scale 闭合后若 ru/rv 仍未达标，再扫描 mesh_uniform_scale（--geometry_prior_gs_model 默认开启）。",
    )
    parser.add_argument(
        "--no_sfm_mstar_optical_envelope_tune_mesh",
        dest="sfm_mstar_optical_envelope_tune_mesh",
        action="store_false",
        help="仅调 camera_height_scale，不扫描比例盒 mesh_uniform_scale。",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_target_ratio",
        type=float,
        default=1.02,
        help="optical envelope target ratio; values above 1 make generated targets slightly larger.",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_min_ratio",
        type=float,
        default=1.0,
        help="Minimum acceptable optical envelope ratio; default rejects undersized targets.",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_mesh_min",
        type=float,
        default=0.85,
        help="Lower bound for optical envelope mesh_uniform_scale search.",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_mesh_max",
        type=float,
        default=2.5,
        help="Upper bound for optical envelope mesh_uniform_scale search.",
    )
    parser.add_argument(
        "--sfm_mstar_optical_envelope_undersize_weight",
        type=float,
        default=2.5,
        help="Extra cost weight when simulated target envelope is smaller than the training target.",
    )
    parser.add_argument(
        "--sfm_mstar_optical_side_u_refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="optical 闭合后用侧视 median ratio_u 做一次精修，并把细小平面尺度修正烘焙到 points3D。"
        "--geometry_prior_gs_model 默认开启。",
    )
    parser.add_argument(
        "--sfm_mstar_optical_side_u_tol",
        type=float,
        default=0.01,
        help="侧视 median ratio_u 精修目标容差，默认 0.01，即 |ratio_u-1|<0.01。",
    )
    parser.add_argument(
        "--sfm_mstar_fill_ref_azimuth_deg",
        type=float,
        default=None,
        help=(
            "MSTAR 训练图像素占比自动闭合时的 SAR 成像参考方位（度）。"
            "默认 None：从 summary 有效侧视图朝向吸附 90°/270°，与 mstar_axis_split 汇总 L/W 一致。"
        ),
    )
    parser.add_argument(
        "--sfm_mstar_auto_image_fill_match",
        action="store_true",
        default=False,
        help=(
            "根据 target_dimension_constraints_summary 中 L/W/H 像素中位数，"
            "用 SAR 成像投影闭合 camera_height_scale 与 sfm_mesh_prior_final_uniform_scale，"
            "使比例盒在参考视角下的成像占比与训练图一致（--geometry_prior_gs_model 默认开启）。"
        ),
    )
    parser.add_argument(
        "--sfm_mstar_auto_image_fill_all_views",
        action="store_true",
        default=False,
        help=(
            "在 auto_image_fill_match 下使用全部 shadow_diagnostic 逐视角 bbox 闭合尺度"
            "（≥3 张有效时启用；--geometry_prior_gs_model 默认开启）。"
        ),
    )
    parser.add_argument(
        "--no_sfm_mstar_auto_image_fill_all_views",
        dest="sfm_mstar_auto_image_fill_all_views",
        action="store_false",
        help="禁用全视角 bbox 闭合，回退到侧视 summary L/W 中位数。",
    )
    parser.add_argument(
        "--sfm_mstar_fixed_physical_target_dims",
        action="store_true",
        default=False,
        help=(
            "比例盒使用 MSTAR 典型米制 W/H（默认 BTR70 W=4.95 m, H≈5.25 m）；"
            "L 默认 W×summary(L/W)（与 shadow 比例盒同源，非硬编码 11.4 m）；"
            "auto_fill / optical verify 仅调 camera_height。"
            "（--geometry_prior_gs_model 默认开启）"
        ),
    )
    parser.add_argument(
        "--no_sfm_mstar_fixed_physical_target_dims",
        dest="sfm_mstar_fixed_physical_target_dims",
        action="store_false",
        help="回退为 shadow summary L/W×ρ 推导比例盒米制。",
    )
    parser.add_argument(
        "--sfm_mstar_fixed_physical_snap_lw_from_summary",
        action="store_true",
        default=False,
        help=(
            "固定米制模式下用 target_dimension_constraints_summary 的 L/W 中位数校正 L"
            "（W 保持典型米制；与 mstar_physical_semiaxes 侧视推理一致；"
            "--geometry_prior_gs_model 默认开启）。"
        ),
    )
    parser.add_argument(
        "--no_sfm_mstar_fixed_physical_snap_lw_from_summary",
        dest="sfm_mstar_fixed_physical_snap_lw_from_summary",
        action="store_false",
        help="禁用 summary L/W 校正，L 固定为目录典型值 11.4 m。",
    )
    parser.add_argument(
        "--sfm_mstar_fixed_physical_snap_lw_min",
        type=float,
        default=1.75,
        help="summary L/W 低于该值时不校正 L（排除 oblique 误汇总，默认 1.75）。",
    )
    parser.add_argument(
        "--sfm_mstar_fixed_physical_snap_lw_max",
        type=float,
        default=3.75,
        help="summary L/W 高于该值时不校正 L（默认 3.75）。",
    )
    parser.add_argument(
        "--sfm_mstar_vehicle_preset",
        type=str,
        default="generic",
        choices=["auto", "generic", "t72_body", "t72_full", "custom"],
        help="固定米制目标尺寸预设。auto 会从文件名前缀识别：T72 默认用 t72_body，"
        "即只用车体主体长度，避免炮管把目标长度判定拉长；generic 保持旧 BTR70 风格默认值；"
        "custom 表示完全使用 --sfm_mstar_target_length_m/width_m/height_m。",
    )
    parser.add_argument(
        "--sfm_mstar_target_length_m",
        type=float,
        default=None,
        help="固定目标全长 L（米，默认 11.4，需 --sfm_mstar_fixed_physical_target_dims）。",
    )
    parser.add_argument(
        "--sfm_mstar_target_width_m",
        type=float,
        default=None,
        help="固定目标全宽 W（米，默认 4.95 = 16.5 px × 0.3 m/px）。",
    )
    parser.add_argument(
        "--sfm_mstar_target_height_m",
        type=float,
        default=None,
        help="固定目标竖直高度 H（米）；未指定时 max(5.25, shadow_H×ρ×boost)。",
    )
    parser.add_argument(
        "--sfm_mstar_target_height_boost",
        type=float,
        default=1.08,
        help="未指定 target_height_m 时，相对 shadow H_px×ρ 的抬高系数（默认 1.08）。",
    )
    parser.add_argument(
        "--sfm_mstar_scene_scale_skip_if_near",
        type=float,
        default=0.92,
        help="若统一缩放因子 s≥该阈值则跳过归一化（已接近目标尺度时）",
    )
    parser.add_argument(
        "--sfm_mesh_prior_vertex_axis_sign",
        type=str,
        default="1,-1,1",
        help="SfM OBJ 顶点在采样对齐前乘以 (sx,sy,sz)，每项为 ±1。默认 1,-1,1：将常见 CAD 「Y向上」翻到 COLMAP 「Y≈向下」的一侧，"
        "配合俯视语义；若网格已在 COLMAP 轴向可改为 1,1,1。",
    )
    parser.add_argument(
        "--sfm_mesh_prior_crop_margin",
        type=float,
        default=0.0,
        help="SfM 网格先验：相对稀疏点 AABB 外扩倍率后**丢弃框外表面采样点**（≠ 等比缩放）。"
        "默认 0：不裁，靠 --sfm_mesh_prior_final_uniform_scale / --sfm_mesh_prior_sar_imaging_scale_match 等做相似缩放。"
        "需要裁掉稀疏盒外伸出的网格部分时再设 1.35 等；≤0 均视为关闭裁剪。",
    )
    parser.add_argument(
        "--sfm_mesh_prior_icp_max_iters",
        type=int,
        default=40,
        help="SfM 网格先验：每级 ICP 最大迭代",
    )
    parser.add_argument(
        "--sfm_mesh_prior_seed",
        type=int,
        default=0,
        help="SfM 网格先验：采样与子采样随机种子",
    )
    parser.add_argument(
        "--sfm_mesh_prior_max_sparse_read",
        type=int,
        default=500_000,
        help="SfM 网格先验：用于对齐的稀疏点上限（子采样）",
    )
    parser.add_argument(
        "--sfm_mesh_prior_save_transform_json",
        nargs="?",
        const="geometry_prior/sfm_mesh_prior_transform.json",
        default="",
        help="SfM 网格/比例盒先验：将 4×4 变换写入 JSON；"
        "仅写开关不写路径时默认为 geometry_prior/sfm_mesh_prior_transform.json（相对 source_path）",
    )
    parser.add_argument(
        "--sfm_mesh_prior_pseudocolor_normals",
        action="store_true",
        default=False,
        help="SfM 网格先验：写入 points3D 的 RGB 用法向伪彩（默认关闭：统一中性灰 128，与 SAR 散射无关，GS 从图像学色）",
    )
    parser.add_argument(
        "--sfm_mesh_prior_nominal_match_sparse_diagonal",
        action="store_true",
        default=False,
        help="SfM：阴影汇总比例盒（名义半轴）粗对齐时使用稀疏对角缩放（旧行为）。"
        " 默认改为不缩放网格，仅用汇总给出的盒尺寸并向稀疏几何中心平移后采样。",
    )
    parser.add_argument(
        "--geometry_prior_rough_model",
        "--rough_model",
        action="store_true",
        default=False,
        help="一键生成大致几何模型（阴影比例盒 replace 稀疏点，供 GS 初始化）。"
        " 自动启用：ring 圆环位姿、90°/270° 侧视 L/W 汇总、standalone 阴影汇总、"
        "SAR 成像尺度对齐、相机光心对齐、50k Poisson 采样、geometry_prior/sfm_summary_transform.json 等；"
        " 等价于原先一长串 --sfm_summary_box_* / --shadow_geometry_* / --disable_cyclic_depression 参数。"
        " 360° 圆环 MSTAR 数据推荐：python convert.py -s <数据> --geometry_prior_rough_model",
    )
    parser.add_argument(
        "--geometry_prior_gs_model",
        "--gs_model",
        action="store_true",
        default=False,
        help="3DGS 训练适配版大致模型：等价 --geometry_prior_rough_model，并额外启用光学等效俯角 SfM 初始化、"
        "**固定 MSTAR 文件名俯仰/方位（仅优化共享 camera_height，不跑全自由度 SO-RCG）**、"
        "SfM 后将 sparse/0 外参替换为 SAR 水平圆环理论位姿。"
        " 推荐流程：convert.py -s <数据> --geometry_prior_gs_model → train_sar_complete.py -s <数据> -m <输出>",
    )
    parser.add_argument(
        "--convert_auto_verify_optical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="convert 完成后导入 scripts/verify_sar_imaging_envelope_match.py 自动运行 optical 评测并生成报告。"
        "--geometry_prior_gs_model 默认开启；可用 --no_convert_auto_verify_optical 关闭。",
    )
    parser.add_argument(
        "--sfm_export_gs_satellite_poses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="SfM 完成后：用 75° 光学圆环理论位姿重写 sparse/0 外参（--geometry_prior_gs_model 时默认开启）。"
        " 训练角拟合 MSTAR 侧视图像较差时，改用 --no-sfm_export_gs_satellite_poses 保留 SO-RCG 外参。",
    )
    parser.add_argument(
        "--sfm_summary_box_mesh_prior_mode",
        type=str,
        default="off",
        choices=["off", "replace", "merge"],
        help="阴影汇总比例盒：与 --sfm_mesh_prior_mode 同一时机（replace=重投影过滤前；merge=过滤后）。"
        " 参数与 --sfm_mesh_prior_* 对齐采样/ICP；盒子尺寸仍读 post_sfm_scheme_a_* 与 target_dimension_constraints_summary.txt。"
        " 若与 --sfm_mesh_prior_obj 同时启用，同一阶段优先 OBJ。"
        " 启用 replace/merge 后勿再在 post_sfm_modules 中重复 merge_summary_box_mesh_prior（将自动跳过）。",
    )
    parser.add_argument(
        "--sfm_summary_box_surface_mode",
        type=str,
        default="box",
        choices=["box", "plane"],
        help="阴影汇总比例盒写入 points3D 的表面模式：box=六面体表面采样；"
        "plane=仅采样 XZ 接地/成像平面，保留 L/W 但不把目标高度侧面写入稀疏点。"
        "--geometry_prior_gs_model 默认 plane，用于 SAR-like 目标散射图减少膜状泄漏。",
    )
    parser.add_argument(
        "--geometry_prior_versioned_run_dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="阴影汇总比例盒（merge_summary_box_mesh_prior / SfM 内 sfm_summary_box_mesh_prior）："
        " 默认每次运行写入 geometry_prior/run_<时间戳>/ 下的 OBJ 与相对路径的 --sfm_mesh_prior_save_transform_json，避免覆盖旧模型；"
        " 固定路径请加 --no-geometry_prior_versioned_run_dir。"
        " 若 --post_sfm_summary_box_mesh_out 为绝对路径则不改动（完全自定义）。",
    )
    parser.add_argument(
        "--sfm_center_scene_after_prior_replace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SfM/convert：mesh prior replace 后平移世界系使 SAR 接地点（默认 footprint 底面中心，见 --sfm_center_scene_anchor）"
        "落在原点，并同步更新相机外参 t′=t+Rμ（COLMAP/OpenCV：x_cam=R·x_world+t）。"
        "若必须与雷达绝对坐标对齐请加 --no-sfm_center_scene_after_prior_replace",
    )
    parser.add_argument(
        "--sfm_center_scene_anchor",
        type=str,
        default="ground_contact",
        choices=("ground_contact", "mean"),
        help="replace 后场景居中锚点：ground_contact=XZ 中位+底面 Y 分位（SAR 默认）；mean=点云均值（旧行为）。",
    )
    parser.add_argument(
        "--sfm_center_scene_ground_y_percentile",
        type=float,
        default=90.0,
        help="ground_contact 锚点的 Y 分位数（COLMAP Y 向下，默认 90=底面侧）。",
    )
    parser.add_argument(
        "--sfm_mesh_prior_nominal_coarse_mode",
        type=str,
        default="ground_contact",
        choices=("ground_contact", "origin_ground_contact", "translate_centroid", "sparse_diagonal"),
        help="阴影比例盒粗对齐：ground_contact=底面中心对齐 sparse footprint；"
        "origin_ground_contact=底面中心固定世界原点（gs_model 默认，不跟随 sparse）；"
        "translate_centroid=盒心对稀疏 AABB 中心；sparse_diagonal=底面+稀疏对角缩放。",
    )

    # SAR Bundle Adjustment参数（默认禁用）
    parser.add_argument("--sfm_use_sar_ba", action='store_true', default=False, help="使用SAR专用的Bundle Adjustment（基于斜距误差，默认禁用）")
    parser.add_argument("--sfm_no_sar_ba", dest='sfm_use_sar_ba', action='store_false', help="禁用SAR Bundle Adjustment")
    parser.add_argument("--sfm_sar_ba_max_iterations", type=int, default=150, help="SAR BA最大迭代次数（默认150，增加以改善点云形状）")
    parser.add_argument("--sfm_sar_ba_ftol", type=float, default=1e-6, help="SAR BA函数容差")
    parser.add_argument("--sfm_sar_ba_xtol", type=float, default=1e-8, help="SAR BA参数容差")
    parser.add_argument("--sfm_ba_fix_angles_height_only", action='store_true', default=False,
                       help="俯仰角与方位角固定为文件名/数据解析结果，仅优化共享平台高度；SO-RCG 与 SAR BA 均适用，并自动禁用循环俯视角")
    parser.add_argument("--sfm_disable_auto_height_only", action='store_true', default=False,
                       help="禁用自动「固定角度+仅优化高度」：当所有图像文件名均含 MSTAR 俯仰/方位时默认会启用该模式；若需完整 SO-RCG 与循环俯视角请指定此项")
    
    # 重投影误差过滤参数（优化点云质量）
    parser.add_argument("--sfm_max_reprojection_error", type=float, default=50.0, 
                       help="最大允许的重投影误差（像素），超过此值的3D点将被过滤。默认50.0，适合128x128图像")
    parser.add_argument("--sfm_filter_outlier_points", action='store_true', default=True,
                       help="是否过滤重投影误差过大的异常点（默认启用）")
    
    # SO-RCG约束参数
    parser.add_argument("--sfm_enable_height_constraint", action='store_true', default=True,
                       help="启用高度约束（默认启用，确保相机在场景上方）")
    parser.add_argument("--sfm_min_z_ratio", type=float, default=0.1,
                       help="最小高度比例（相对于球面半径，COLMAP坐标系：Y轴负方向为高度）。"
                            "默认0.1，即|Y| >= 球面半径×0.1（高度在Y轴负方向）。"
                            "对于球面半径12.8米，min_height=1.28米")
    parser.add_argument("--sfm_enable_azimuth_constraint", action='store_true', default=True,
                       help="启用方位角约束（默认启用，确保相机方位角与图像一致）")
    parser.add_argument("--sfm_azimuth_tolerance", type=float, default=15.0,
                       help="方位角容差（度），超过此值的相机位置将被校正。默认15度")
    parser.add_argument("--sfm_enable_position_constraint", action='store_true', default=True,
                       help="启用位置约束（默认启用，限制相机位置不能偏离初始位置太远）")
    parser.add_argument("--sfm_position_tolerance_ratio", type=float, default=0.3,
                       help="位置偏离容差比例（相对于初始位置的距离）。默认0.3，即相机位置不能偏离初始位置超过30%%")
    parser.add_argument("--sfm_enable_depression_constraint", action='store_true', default=True,
                       help="启用俯视角约束（默认启用，限制俯视角在合理范围内）")
    parser.add_argument("--sfm_depression_min", type=float, default=10.0,
                       help="最小俯视角（度），默认10°")
    parser.add_argument("--sfm_depression_max", type=float, default=80.0,
                       help="最大俯视角（度），默认80°")
    parser.add_argument("--sfm_enable_point_height_constraint", action='store_true', default=True,
                       help="启用点云高度约束（默认启用，限制点云高度/长度比例）")
    parser.add_argument("--sfm_point_height_length_ratio", type=float, default=0.3,
                       help="点云高度/长度比例阈值（默认0.4，即高度范围不应超过长度范围的50%%）")
    parser.add_argument("--sfm_point_height_constraint_mode", type=str, default='scale',
                       choices=['filter', 'scale'],
                       help="点云高度约束模式：'scale'=缩放所有点的高度以符合约束（默认），'filter'=过滤超出范围的点")
    parser.add_argument(
        "--sfm_ignore_target_dimension_summary_ratios",
        action="store_true",
        default=False,
        help="为 True 时不从 adaptive_threshold_visualization/target_dimension_constraints_summary.txt 注入 H/L、(H/W)："
             "仅用 --sfm_point_height_length_ratio 等固定值（默认 False，即优先用汇总中位数驱动 SO-RCG 等比例约束）",
    )
    parser.add_argument("--sfm_enable_point_width_constraint", action='store_true', default=True,
                       help="启用点云宽度约束（默认启用，限制点云宽度/长度或高度/宽度比例）")
    
    # SO-RCG方法参数（基于论文"Keypoint-Based SAR Structure From Motion via Riemannian Optimization"）
    parser.add_argument("--sfm_use_sorc", action='store_true', default=True, help="使用SO-RCG方法进行SAR SfM重建（在SO(3)群上优化）")
    parser.add_argument("--sfm_sorc_max_iterations", type=int, default=100, help="SO-RCG最大迭代次数")
    parser.add_argument(
        "--sfm_sorc_height_only_max_iterations",
        type=int,
        default=20,
        help="固定俯仰/方位、仅优化共享高度时 SO-RCG 外层迭代上限（默认 20；"
        "高度在第 1 轮一维搜索后即固定，后续轮只优化点云）",
    )
    parser.add_argument("--sfm_sorc_tolerance", type=float, default=1e-6, help="SO-RCG收敛容差")
    parser.add_argument("--sfm_sorc_use_5dof", action='store_true', default=True, help="使用5个自由度约束（SAR成像只依赖5个DoF）")
    parser.add_argument("--sfm_sorc_use_gpu", action='store_true', default=False, help="使用GPU加速SO-RCG优化（需要PyTorch和CUDA，默认禁用，使用CPU以确保参数计算的规范性）")
    parser.add_argument("--sfm_sorc_gpu_batch_size", type=int, default=1000, help="SO-RCG GPU批量大小（增大可提高GPU利用率，但需要更多GPU内存，默认1000）")
    parser.add_argument("--sfm_sorc_num_threads", type=int, default=None, help="SO-RCG CPU多线程数量（None表示自动检测CPU核心数，0表示禁用多线程，默认None）")
    
    # 相机间距离惩罚项参数（用于SOSAR优化，减小相机间距离）
    parser.add_argument("--sfm_enable_camera_distance_penalty", action='store_true', default=False,
                       help="启用相机间距离惩罚项（默认禁用，启用后会在SOSAR优化中惩罚相机间距离偏离目标值）")
    parser.add_argument("--sfm_camera_distance_penalty_weight", type=float, default=0.001,
                       help="相机间距离惩罚项权重（默认0.001，值越大惩罚越强，建议范围0.0001-0.01）")
    parser.add_argument("--sfm_camera_target_distance", type=float, default=10.0,
                       help="相机间目标距离（米），用于惩罚项计算（默认10.0米，相机间距离越接近此值惩罚越小）")
    
    # 矩阵分解方法参数（现在这是唯一支持的方法）
    parser.add_argument("--sfm_matrix_factorization_max_iterations", type=int, default=100,
                        help="矩阵分解方法的最大迭代次数（默认: 100）")
    parser.add_argument("--sfm_matrix_factorization_tolerance", type=float, default=1e-6,
                        help="矩阵分解方法的收敛容差（默认: 1e-6）")

    from sar.semantic_mask_config import register_semantic_mask_cli_args
    register_semantic_mask_cli_args(parser)

    parser.add_argument(
        "--no_sar_sift_target_keypoint_filter",
        action="store_false",
        dest="sar_sift_target_keypoint_filter",
        help="关闭 SAR-SIFT 检测后过滤（默认开启：仅保留「目标 mask==1 再经 ROI 膨胀」内的特征点，避免极值落在轮廓外黑底）",
    )
    parser.add_argument(
        "--target_keypoint_roi_dilate_px",
        type=int,
        default=8,
        help="目标特征 ROI：在目标区域（mask 1∪2）上再 3×3 膨胀迭代次数，仅用于特征保留/分布图；与 --target_mask_dilate_px 独立。128 小图默认 8",
    )
    parser.add_argument("--target_threshold_factor", type=float, default=0.2, help="目标区域Harris阈值因子")
    parser.add_argument("--background_threshold_factor", type=float, default=200.0, help="背景区域Harris阈值因子")
    parser.add_argument("--transition_threshold_factor", type=float, default=0.5, help="过渡区域Harris阈值因子")

    # ---------- SfM 完成后可插拔后处理（convert.py → post_sfm，方案 A 体积先验等）----------
    parser.add_argument(
        "--post_sfm_modules",
        type=str,
        default="",
        help="SfM 成功后按顺序运行的模块，逗号分隔。内置: backup_sparse, merge_external_ply, "
        "merge_mesh_obj_prior（OBJ 对齐 sparse 后表面采样合并）, "
        "merge_summary_box_mesh_prior（由阴影汇总 L/W/H 生成长方体 OBJ，再走与 OBJ 完全相同对齐/采样/合并）, "
        "build_cardinal_box_obj（仅由汇总 txt 写出优选方位角长方体 OBJ，不写 sparse；见 --post_sfm_cardinal_box_*）, "
        "scheme_a_geometry_prior（多视剪影 hull + 射线锥 + 阴影高度带等体积先验）, "
        "scheme_a_height_band（仅均匀体素）。空=禁用。"
        " 若已启用网格几何先验（--sfm_mesh_prior_obj 且 mode≠off，或 --post_sfm_mesh_obj，或在列表中包含 "
        "merge_summary_box_mesh_prior），"
        "将自动从本列表中移除 scheme_a_*（可用 --post_sfm_force_scheme_a 强制保留）。",
    )
    parser.add_argument(
        "--post_sfm_strict",
        action="store_true",
        default=False,
        help="post_sfm 任一模块失败则中止 convert（默认失败仅打印堆栈并继续后续步骤）",
    )
    parser.add_argument(
        "--post_sfm_force_scheme_a",
        action="store_true",
        default=False,
        help="强制运行 scheme_a_geometry_prior / scheme_a_height_band。"
        " 默认在已配置网格先验（SfM/post_sfm OBJ，或 post_sfm_modules 中含 merge_summary_box_mesh_prior）时会自动跳过这两类模块，加此项可覆盖。",
    )
    parser.add_argument(
        "--post_sfm_external_ply",
        type=str,
        default="",
        help="merge_external_ply 模块：合并到 sparse/0 的外部 PLY（需含 x,y,z；无 RGB 则用灰色）",
    )
    parser.add_argument(
        "--post_sfm_merge_min_dist",
        type=float,
        default=0.0,
        help="merge_external_ply / merge_mesh_obj_prior / merge_summary_box_mesh_prior："
        "与现有点最近距离低于此值（米）的新点不添加；0=不过滤",
    )
    parser.add_argument(
        "--post_sfm_mesh_obj",
        type=str,
        default="",
        help="mesh 先验：目标网格路径（.obj/.ply/.stl/.off），相对路径时相对 --source_path 解析；"
        "merge_summary_box_mesh_prior 可独立使用无需此项",
    )
    parser.add_argument(
        "--post_sfm_mesh_nominal_match_sparse_diagonal",
        action="store_true",
        default=False,
        help="merge_summary_box_mesh_prior：名义盒按稀疏对角线缩放对齐（旧行为）。默认仅平移至稀疏几何中心并保持汇总名义尺寸。",
    )
    parser.add_argument(
        "--post_sfm_mesh_sample_count",
        type=int,
        default=200_000,
        help="mesh 合并：写入 sparse 前表面采样点数上限（裁剪后可能更少）；merge_mesh_obj_prior "
        "与 merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_icp_source_samples",
        type=int,
        default=80_000,
        help="mesh 合并：用于 ICP 与粗对齐的网格表面采样数；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_skip_icp",
        action="store_true",
        default=False,
        help="mesh 合并：跳过 Open3D 多尺度 ICP，仅包络盒粗相似对齐；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_poisson_disk",
        action="store_true",
        default=False,
        help="mesh 合并：泊松盘表面采样（否则均匀）；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_scale_override",
        type=float,
        default=0.0,
        help="mesh 合并：>0 时覆盖粗对齐尺度；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_crop_margin",
        type=float,
        default=1.45,
        help="mesh 合并：相对 sparse AABB 外扩倍率后裁剪采样；≤0 关闭；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_icp_max_iters",
        type=int,
        default=40,
        help="mesh 合并：ICP 每级最大迭代；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_seed",
        type=int,
        default=0,
        help="mesh 合并：采样种子；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_max_sparse_read",
        type=int,
        default=500_000,
        help="mesh 合并：sparse 对齐子采样点数上限；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_save_transform_json",
        type=str,
        default="",
        help="mesh 合并：4×4 变换 JSON；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_save_aligned_ply",
        type=str,
        default="",
        help="mesh 合并：对齐后采样 PLY（调试）；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_mesh_no_align_camera_centers",
        action="store_true",
        default=False,
        help="mesh 合并：关闭「相机光心 + sparse」联合估计包围盒与 ICP；默认合并光心以更稳地与环视相机配准",
    )
    parser.add_argument(
        "--post_sfm_mesh_pseudocolor_normals",
        action="store_true",
        default=False,
        help="mesh 合并：法向伪彩；merge_summary_box_mesh_prior 共用",
    )
    parser.add_argument(
        "--post_sfm_summary_box_mesh_out",
        type=str,
        default="geometry_prior/shadow_proportional_box.obj",
        help="merge_summary_box_mesh_prior：写出阴影比例轴对齐长方体网格的路径（相对 --source_path）；"
        "之后与 merge_mesh_obj_prior 相同：build_mesh_prior_pointcloud（ICP、表面采样、写回 sparse）",
    )
    parser.add_argument(
        "--post_sfm_summary_box_orient_mode",
        type=str,
        default="none",
        choices=["auto", "aligned", "pca_only", "summary_gamma_only", "none", "off"],
        help="【已弃用保留】比例盒现固定为轴对齐 + 名义粗尺度 + ICP。设非 none 时仅打印提示，不改变几何。",
    )
    parser.add_argument(
        "--post_sfm_summary_box_pca_min_ratio",
        type=float,
        default=1.0,
        help="merge_summary_box：XZ 伸长轴 PCA 阈值 √λ_max/√λ_min，低于则用汇总 γ（auto 模式下）",
    )
    parser.add_argument(
        "--post_sfm_summary_box_yaw_offset_deg",
        type=float,
        default=0.0,
        help="merge_summary_box：在已定朝向上再绕世界 +Y（竖直轴）外加角度（度），用于手工微调车体偏航",
    )
    parser.add_argument(
        "--post_sfm_summary_box_align_camera_centers",
        action="store_true",
        default=False,
        help="merge_summary_box_mesh_prior：对齐时把相机光心与 sparse 拼入 ICP/粗尺度目标。"
        "默认关闭，与 SfM 内 --sfm_mesh_prior_mode replace（仅稀疏点）一致；"
        "需环视相机牵引时再开启。",
    )
    parser.add_argument(
        "--post_sfm_summary_box_no_crop_on_replace",
        action="store_true",
        default=False,
        help="merge_summary_box_mesh_prior：在 replace_sparse 下仍关闭按 sparse AABB×margin 的裁剪（保留全量表面点）。"
        "默认不指定本项时：replace 后仍裁剪，与 SfM 内网格 replace 一致；"
        "merge_mesh_obj_prior 在 replace 时默认 margin=0 的旧行为若需比例盒也全保留，请加本项。",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_out",
        type=str,
        default="geometry_prior/box_cardinal.obj",
        help="build_cardinal_box_obj：输出轴对齐长方体 OBJ（相对 --source_path）；"
        "读 adaptive_threshold_visualization/target_dimension_constraints_summary.txt，"
        "优选方位角聚合 L/W/H，不经 ICP、不写 sparse",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_depth_scale",
        type=float,
        default=1.0,
        help="build_cardinal_box_obj：像素→世界长度倍数（例如 COLMAP 尺度下米/像素 0.002）",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_azimuth_tolerance",
        type=float,
        default=5.0,
        help="build_cardinal_box_obj：与 post_sfm_cardinal_box_azimuth_lane 一致的主参考角族偏差（度）；"
        "cardinal 时为 0/90/180/270；diagonal 时为 45/135/225/315",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_azimuth_lane",
        type=str,
        default="cardinal",
        choices=["cardinal", "diagonal"],
        help="build_cardinal_box_obj：汇总 txt 中筛选方位的参考族。"
        "cardinal=0/90/180/270°±容差；diagonal=45/135/225/315°±容差（叠掩—阴影邻带分割不稳时可试）。",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_aggregate",
        type=str,
        default="weighted_mean",
        choices=["median", "mean", "weighted_mean"],
        help="build_cardinal_box_obj：L/W/H 聚合；weighted_mean=按到 lane 参考角角距线性降权",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_azimuth_weight_floor",
        type=float,
        default=0.05,
        help="build_cardinal_box_obj：加权聚合时最小权重（角距≥容差时）；默认 0.05",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_height_mode",
        type=str,
        default="cardinal",
        choices=["cardinal", "all_images"],
        help="build_cardinal_box_obj：H 取自优选方位样本(cardinal)或全部解析图像(all_images)",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_include_all_for_lw",
        action="store_true",
        default=False,
        help="build_cardinal_box_obj：不筛方位角，用全部汇总图像估计 L/W（仅缺数据时调试）",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_json",
        type=str,
        default="",
        help="build_cardinal_box_obj：可选诊断 JSON 路径（相对 source_path 或绝对路径）",
    )
    parser.add_argument(
        "--post_sfm_cardinal_box_ascii_only",
        action="store_true",
        default=False,
        help="build_cardinal_box_obj：不尝试 Open3D，仅写 ASCII 三角网格 OBJ",
    )
    parser.add_argument(
        "--post_sfm_mesh_replace_sparse",
        action="store_true",
        default=False,
        help="merge_mesh_obj_prior / merge_summary_box_mesh_prior：ICP 与表面采样后轮替 points3D，"
        "仅保留采样点（无 track）；推荐与比例盒联用以避免与原 SfM 点混叠。"
        "merge_mesh_obj_prior：replace 时默认 margin=0（不裁）。"
        "merge_summary_box_mesh_prior：默认仍按 --post_sfm_mesh_crop_margin 相对 sparse 裁（与 SfM 内 replace 一致），"
        "全保留请加 --post_sfm_summary_box_no_crop_on_replace。",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_grid",
        type=int,
        default=12,
        help="scheme_a_height_band：AABB 每条轴划分数，总点数约 n^3（建议 8～16）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_aabb_margin_pct",
        type=float,
        default=5.0,
        help="scheme_a_height_band：包络盒相对各轴跨度的外扩百分比",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_min_dist_to_existing",
        type=float,
        default=0.02,
        help="scheme_a_height_band：新网格点与已有 sparse 点的最小间距（米），避免重复堆点",
    )
    parser.add_argument(
        "--post_sfm_mask_dirs",
        type=str,
        default="",
        help="额外查找 *_mask.npy 的目录（逗号分隔），默认已搜 source_path 下 scattering_masks 与 adaptive_threshold_visualization",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_no_visual_hull",
        dest="post_sfm_scheme_a_enable_visual_hull",
        action="store_false",
        help="关闭 scheme_a_geometry_prior 的多视剪影一致 (visual hull)；默认启用",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_no_cone_samples",
        dest="post_sfm_scheme_a_enable_cone_samples",
        action="store_false",
        help="关闭剪影锥射线稠密采样；默认启用",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_no_height_slab",
        dest="post_sfm_scheme_a_enable_height_slab",
        action="store_false",
        help="关闭基于汇总报告的阴影高度带；默认启用",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_no_planform_from_summary",
        dest="post_sfm_scheme_a_enable_planform_from_summary",
        action="store_false",
        help="关闭用汇总 L/W（像素×深度尺度）在 XZ 平面裁剪先验点；默认启用",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_planform_scale",
        type=float,
        default=1.0,
        help="平面裁剪：L/W 换算为米后再乘以此系数；默认 1.0，直接按训练图像统计闭合尺寸",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_no_planform_clip_uniform",
        dest="post_sfm_scheme_a_planform_clip_uniform",
        action="store_false",
        help="scheme_a_height_band 均匀模式也关闭 L/W 平面裁剪；默认均匀模式会裁剪",
    )

    parser.add_argument(
        "--post_sfm_scheme_a_min_views",
        type=int,
        default=2,
        help="visual hull：至少多少张视图剪影同时覆盖该体素",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_surface_shell_only",
        action="store_true",
        default=False,
        help="scheme_a_geometry_prior：visual hull 仅保留外壳一层体素（scipy 形态学边界），"
        "更接近「模型表面采样」而无 CAD；自动关闭剪影锥与均匀退火，并可能缩小 voxel_div 以满足 max_voxel_eval",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_fallback_summary_box_shell_points",
        type=int,
        default=0,
        help="无掩码时 scheme_a：用汇总 L/W/H 在椭球（默认）或长方体表面采样目标点数；0=max(8000, max_new capped)；"
        "密表层可调大并配合较小的 --post_sfm_scheme_a_min_dist_new",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_disable_summary_box_shell",
        action="store_true",
        default=False,
        help="禁用上述无掩码长方体表面兜底（仅保留旧均匀 AABB）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_summary_prior_surface_mode",
        type=str,
        default="ellipsoid",
        choices=["ellipsoid", "box"],
        help="无掩码且可用阴影汇总 L/W/H 时：ellipsoid=Fibonacci 近似均匀布满椭球表面（单层密点）；"
        "box=长方体六个面网格（与原行为一致）；配合 post_sfm_scheme_a_fallback_summary_box_shell_points",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_voxel_div",
        type=int,
        default=36,
        help="每条轴体素划分数（+1 个节点），总评估量约 n^3；过大时按 post_sfm_scheme_a_max_voxel_eval 子采样",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_max_voxel_eval",
        type=int,
        default=350000,
        help="visual hull 最多评估的体素中心数（防 OOM）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_eval_chunk",
        type=int,
        default=65536,
        help="向 GPU/CPU 投影批大小（一次处理的体素数）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_mask_dilate",
        type=int,
        default=1,
        help="目标掩码 3×3 膨胀迭代次数（容错配准误差）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_mask_labels",
        type=str,
        default="1,2",
        help="视为目标的散射标签（逗号分隔），默认 1,2 即内部+边缘",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_cone_stride",
        type=int,
        default=3,
        help="剪影锥：掩码像素步长（越大射线越少）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_cone_depth_steps",
        type=int,
        default=16,
        help="每条射线在相机 Z 方向的采样层数",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_cone_max_rays_per_view",
        type=int,
        default=4000,
        help="每视图最多抽多少条掩码射线（子采样）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_shadow_height_scale",
        type=float,
        default=1.0,
        help="H_world = H_px×depth_scale×该因子（阴影高度偏保守时可 <1，偏松可 >1）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_height_slab_upper_scale",
        type=float,
        default=1.25,
        help="相对参考高度，向上（-Y 向）保留 H_world 的倍数",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_height_slab_lower_scale",
        type=float,
        default=0.35,
        help="相对参考高度，向下允许多大 H_world 比例（地面侧放宽）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_max_new_points",
        type=int,
        default=250000,
        help="写入 sparse 的先验点上限（去冗后）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_min_dist_new",
        type=float,
        default=0.015,
        help="新先验点彼此及与现 sparse 的最小距离（米）",
    )
    parser.add_argument(
        "--post_sfm_scheme_a_no_fallback_uniform",
        dest="post_sfm_scheme_a_fallback_uniform_fill",
        action="store_false",
        help="禁止在掩码不足时做 AABB 均匀退化填补；默认允许",
    )

    parser.add_argument("--target_boost", type=float, default=1.0, help="目标区域对比度增强因子")
    parser.add_argument("--background_suppress", type=float, default=1.0, help="背景区域对比度抑制因子")
    parser.add_argument("--target_dark_suppress", type=float, default=1.0, help="目标区域内部暗部抑制因子")
    parser.add_argument("--target_bright_boost", type=float, default=1.0, help="目标区域内部亮部增强因子")

    args = parser.parse_args()
    try:
        import sys
        args._argv = list(sys.argv)
    except Exception:
        pass

    # 设置正向标志
    args.save_feature_images = not args.no_save_feature_images
    args.save_match_images = not args.no_save_match_images

    # 自动GPU后端设置
    if args.gpu_backend == "cupy":
        print("⚠️ 检测到CuPy后端，由于已知的内存问题，自动切换到PyTorch后端")
        args.gpu_backend = "pytorch"

    # 统一GPU/CPU控制逻辑（SAR图像匹配使用）
    args.force_cpu = args.no_gpu or args.use_cpu
    args.enable_gpu = not args.force_cpu
    args.compute_device = args.gpu_device if args.enable_gpu else "cpu"

    # 将解析后的阈值倍率因子更新到全局变量，供其他模块使用
    global TARGET_THRESHOLD_FACTOR, BACKGROUND_THRESHOLD_FACTOR, TRANSITION_THRESHOLD_FACTOR
    TARGET_THRESHOLD_FACTOR = args.target_threshold_factor
    BACKGROUND_THRESHOLD_FACTOR = args.background_threshold_factor
    TRANSITION_THRESHOLD_FACTOR = args.transition_threshold_factor

    try:
        apply_config_presets(args)
    except Exception as e:
        print(f"⚠️ config 预设应用失败: {e}")

    return args
