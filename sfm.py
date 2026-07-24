import numpy as np 
import cv2 
import math
import argparse
import pickle
import os 
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot

from sfm_utils import * 
try:
    from sar_geometry import (
        compute_sar_camera_pose, sar_stereo_triangulation,
        compute_sar_fundamental_matrix, sar_pose_from_angles_and_matches,
        DEFAULT_SAR_PARAMS
    )
    from sar_utils import parse_mstar_filename
    SAR_GEOMETRY_AVAILABLE = True
except ImportError:
    SAR_GEOMETRY_AVAILABLE = False
    print("⚠️ 警告: sar_geometry模块未找到，将使用标准光学相机模型")

try:
    from sar_matrix_factorization_sfm import SARMatrixFactorizationSfM
    MATRIX_FACTORIZATION_AVAILABLE = True
except ImportError:
    MATRIX_FACTORIZATION_AVAILABLE = False
    print("⚠️ 警告: sar_matrix_factorization_sfm模块未找到，矩阵分解方法不可用")

try:
    from sar_sorc_sfm import SORCG_SAR_SfM
    SORCG_AVAILABLE = True
except ImportError as e:
    SORCG_AVAILABLE = False
    import traceback
    print(f"⚠️ 警告: sar_sorc_sfm模块未找到，SO-RCG方法不可用: {e}")
    print(f"   详细错误信息:")
    traceback.print_exc()
except Exception as e:
    SORCG_AVAILABLE = False
    import traceback
    print(f"⚠️ 警告: 加载sar_sorc_sfm模块时出错，SO-RCG方法不可用: {e}")
    print(f"   详细错误信息:")
    traceback.print_exc()

import pdb 

class Camera(object): 
    def __init__(self, R, t, ref): 
        self.R = R 
        self.t = t 
        self.ref = ref

class Match(object): 
    def __init__(self, matches, img1pts, img2pts, img1idx, img2idx, mask): 
        self.matches = matches
        self.img1pts, self.img2pts = img1pts, img2pts 
        self.img1idx, self.img2idx = img1idx, img2idx
        self.mask = mask

class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = np.zeros((0,3))

        #setting up directory stuff..
        self.images_dir = os.path.join(opts.data_dir,opts.dataset, 'images')
        self.feat_dir = os.path.join(opts.data_dir, opts.dataset, 'features', opts.features)
        self.matches_dir = os.path.join(opts.data_dir, opts.dataset, 'matches', opts.matcher)
        self.out_cloud_dir = os.path.join(opts.out_dir, opts.dataset, 'point-clouds')
        # 修改sparse目录路径：直接使用data_dir（source_path）而不是out_dir/dataset
        self.out_sparse_dir = os.path.join(opts.data_dir, 'sparse', '0')
        self.out_err_dir = os.path.join(opts.out_dir, opts.dataset, 'errors')

        #output directories
        if not os.path.exists(self.out_cloud_dir): 
            os.makedirs(self.out_cloud_dir)
        if not os.path.exists(self.out_sparse_dir):
            os.makedirs(self.out_sparse_dir)

        if (opts.plot_error is True) and (not os.path.exists(self.out_err_dir)): 
            os.makedirs(self.out_err_dir)

        self.image_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir)) \
                            if x.split('.')[-1] in opts.ext]

        #setting up shared parameters for the pipeline
        self.image_data, self.matches_data, errors = {}, {}, {}
        self.image_K = {}  # 存储每张图像的K矩阵（SAR图像每张的K可能不同）
        self.image_sizes = {}  # 存储每张图像的尺寸
        
        # 统计信息：用于分析匹配点的用途
        self.stats_new_points = 0  # 创建新3D点的匹配数
        self.stats_merged_points = 0  # 合并/扩展已有3D点的匹配数
        
        # SAR模式：强制使用SAR几何模型（此代码专用于SAR图像）
        self.use_sar_geometry = True
        
        if not SAR_GEOMETRY_AVAILABLE:
            raise ImportError("❌ 错误: sar_geometry模块未找到，SAR重建需要此模块")
        
        print("🔧 使用SAR几何模型进行重建")
        # 获取SAR参数（如果有）
        if hasattr(opts, 'sar_params'):
            self.sar_params = opts.sar_params
        else:
            self.sar_params = DEFAULT_SAR_PARAMS.copy()
        # 配置中的物理平台高度（米），与 SfM 场景坐标下的 MSTAR 初值 H 区分
        _phys_h = getattr(opts, 'sar_camera_height', None)
        if _phys_h is None:
            full = getattr(opts, '_full_cli_args', None)
            if full is not None:
                _phys_h = getattr(full, 'sar_camera_height', None)
        self._physical_platform_height_m = float(
            _phys_h if _phys_h is not None else DEFAULT_SAR_PARAMS['camera_height']
        )
        
        # 读取实际图像尺寸并更新SAR参数
        if len(self.image_names) > 0 and os.path.exists(self.images_dir):
            image_files = [f for f in os.listdir(self.images_dir) 
                          if f.split('.')[-1].lower() in opts.ext]
            if image_files:
                first_image_path = os.path.join(self.images_dir, image_files[0])
                first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
                if first_image is not None:
                    height, width = first_image.shape
                    # 更新SAR参数中的图像尺寸
                    self.sar_params['Na'] = width
                    self.sar_params['Nr'] = height
                    print(f"📐 检测到图像尺寸: {width} x {height}，更新SAR参数")
        
        # 若全部图像文件名均含 MSTAR 式俯仰/方位元数据，默认启用「固定角度、仅优化共享 camera_height」
        self._angles_from_filename_metadata = False
        if len(self.image_names) > 0:
            try:
                from sar_utils import mstar_filename_has_angle_metadata
                if all(mstar_filename_has_angle_metadata(n) for n in self.image_names):
                    self._angles_from_filename_metadata = True
            except ImportError:
                pass
        disable_auto_ho = getattr(opts, 'sfm_disable_auto_height_only', False)
        if self._angles_from_filename_metadata and not disable_auto_ho:
            self.sar_params['ba_fix_angles_optimize_height_only'] = True
            print(
                "\n📌 检测到所有图像文件名均含俯仰角/方位角（MSTAR 模式）→ "
                "启用「固定俯仰/方位、仅优化共享 camera_height」；已关闭循环俯视角。\n"
                "   若需完整 SO-RCG 位姿优化，请添加 --sfm_disable_auto_height_only\n"
            )
        if getattr(opts, 'sfm_ba_fix_angles_height_only', False):
            self.sar_params['ba_fix_angles_optimize_height_only'] = True

        # 光学等效俯角（SO-RCG / SAR BA 初始位姿，见 sar/optical_equivalent_depression.py）
        self._use_optical_equiv_dep = bool(
            getattr(opts, "sfm_use_optical_equivalent_depression", False)
        )
        self._optical_equiv_dep_method = str(
            getattr(opts, "sfm_optical_equivalent_depression_method", "ring_look_direction")
        )
        self._optical_equiv_dep_scale = float(
            getattr(opts, "sfm_optical_equivalent_depression_scale", 1.0)
        )
        if self._use_optical_equiv_dep:
            self.sar_params["camera_distribution_mode"] = "ring"
            print(
                "\n📐 已启用光学等效俯视角初始化（SO-RCG/SAR BA 前）\n"
                f"   方法: {self._optical_equiv_dep_method}\n"
                f"   相机分布: ring（与光学透视 GS 一致）\n"
                "   循环俯视角: 强制关闭（保留文件名 SAR 俯角→换算 φ_opt）\n"
            )
        elif getattr(opts, "sar_camera_distribution_mode", None):
            mode = str(getattr(opts, "sar_camera_distribution_mode", "ring")).strip().lower()
            if mode in ("ring", "sphere"):
                self.sar_params["camera_distribution_mode"] = mode

        # 360° 方位圆环：文件名俯角一致、方位跨越大 → 禁用循环俯视角 + 水平圆环
        self._azimuth_ring_dataset = False
        if len(self.image_names) >= 8:
            try:
                _deps, _azis = [], []
                for _n in self.image_names:
                    _d, _a = parse_mstar_filename(_n)
                    _deps.append(float(_d))
                    _azis.append(float(_a))
                _dep_unique = len({round(x, 2) for x in _deps})
                _az_span = float(max(_azis) - min(_azis))
                if _dep_unique == 1 and _az_span >= 180.0:
                    self._azimuth_ring_dataset = True
            except Exception:
                pass
        if self._azimuth_ring_dataset:
            self.sar_params["camera_distribution_mode"] = "ring"
            print(
                "\n🛰️ 检测到 360° 方位圆环数据集（固定俯角 + 方位≥180°）→ "
                "禁用循环俯视角，相机分布 ring（水平圆环，与 SAR 卫星轨迹一致）\n"
            )

        # 解析所有图像的俯视角和方位角
        self.image_angles = {}
        depression_angles = []  # 收集所有俯视角，用于自动计算中心下视角
        
        # 检查是否启用循环俯视角模式（用于解决三角化退化问题）
        # 当所有图像的俯视角相同时，通过循环变化俯视角来避免相机在同一高度
        # 默认启用，可以通过 --disable_cyclic_depression 来禁用
        # 若启用「固定俯仰/方位、仅优化高度」模式，必须使用文件名/数据中的真实角度，强制关闭循环俯视角
        disable_cyclic_depression = getattr(opts, 'disable_cyclic_depression', False) or getattr(
            opts, 'sfm_ba_fix_angles_height_only', False
        )
        if self._use_optical_equiv_dep:
            disable_cyclic_depression = True
        if hasattr(self, 'sar_params') and self.sar_params.get('ba_fix_angles_optimize_height_only'):
            disable_cyclic_depression = True
        if getattr(self, "_azimuth_ring_dataset", False):
            disable_cyclic_depression = True
        enable_cyclic_depression = not disable_cyclic_depression  # 默认启用
        
        # 明确显示循环俯视角模式状态
        print(f"\n{'='*60}")
        print(f"循环俯视角模式状态检查")
        print(f"{'='*60}")
        print(f"  enable_cyclic_depression = {enable_cyclic_depression}")
        if enable_cyclic_depression:
            print(f"  ✅ 循环俯视角模式: 已启用（默认）")
        else:
            print(f"  ℹ️ 循环俯视角模式: 已禁用（使用实际俯视角）")
            print(f"  💡 提示: 循环俯视角模式默认启用，如需禁用，请在命令行中添加 --disable_cyclic_depression")
        print(f"{'='*60}\n")
        
        # 生成循环俯视角序列
        # 注意：默认值已移除，统一从config.py读取，确保单一配置源
        def generate_cyclic_depression_angles(num_images, start_angle, end_angle, step):
            """
            生成循环俯视角序列
            
            Args:
                num_images: 图像总数
                start_angle: 起始角度（度）
                end_angle: 结束角度（度）
                step: 角度步长（度）
            
            Returns:
                循环俯视角列表
                
            示例：
                - (15, 45, 15)：[15, 30, 45, 30, 15, 30, 45, ...]
                - (15, 45, 5)：[15, 20, 25, ..., 45, 40, 35, ..., 20, 15, ...]
                
            注意：
                所有默认值已从config.py统一管理，函数不再提供默认参数。
            """
            angles = []
            # 上升序列：start_angle -> start_angle+step -> ... -> end_angle
            ascending = list(range(start_angle, end_angle + 1, step))
            # 下降序列：end_angle-step -> ... -> start_angle+step
            # （不包括起始点start_angle和终点end_angle，避免与周期边界重复）
            descending = list(range(end_angle - step, start_angle, -step))
            
            # 组合成一个周期
            cycle = ascending + descending
            
            # 循环生成足够的角度
            for i in range(num_images):
                angles.append(cycle[i % len(cycle)])
            
            return angles
        
        # 如果启用循环俯视角模式，先生成循环序列
        cyclic_depression_angles = None  # 在外部定义，确保作用域正确
        if enable_cyclic_depression:
            # ✅ 统一配置源：所有参数都从config.py读取（opts对象）
            # 如果opts中没有这些属性（极少见情况），使用与config.py一致的默认值作为兜底
            start_angle = getattr(opts, 'cyclic_depression_start', 15.0)
            end_angle = getattr(opts, 'cyclic_depression_end', 45.0)
            step_angle = getattr(opts, 'cyclic_depression_step', 15.0)
            
            # 调用函数生成循环序列（参数来自config.py）
            cyclic_depression_angles = generate_cyclic_depression_angles(
                len(self.image_names), 
                start_angle=int(start_angle), 
                end_angle=int(end_angle), 
                step=int(step_angle)
            )
            print(f"🔄 启用循环俯视角模式: {len(self.image_names)} 张图像")
            print(f"   📝 配置来源: config.py（--cyclic_depression_* 参数）")
            print(f"   参数: 起始角度={start_angle}°, 结束角度={end_angle}°, 步长={step_angle}°")
            print(f"   俯视角序列: {cyclic_depression_angles[:10]}... (显示前10个)")
            print(f"   💡 提示: 修改参数请编辑 config.py 或命令行指定")
            print(f"   🔍 调试: 完整序列长度={len(cyclic_depression_angles)}, 序列={cyclic_depression_angles}")
        
        # 收集实际俯视角（用于计算统一相机半径）
        # 如果启用循环俯视角模式，使用循环俯视角；否则使用实际俯视角
        actual_depression_angles = []
        
        for idx, img_name in enumerate(self.image_names):
            # 尝试从文件名解析角度
            try:
                dep, azi = parse_mstar_filename(img_name)
                dep_sar = float(dep)

                # 如果启用循环俯视角模式，使用循环序列中的俯视角
                # 保留原始的方位角（光学等效模式下一律不用循环角）
                if (
                    enable_cyclic_depression
                    and cyclic_depression_angles is not None
                    and not self._use_optical_equiv_dep
                ):
                    dep_cyclic = cyclic_depression_angles[idx]
                    # 打印策略：前10个、每10个、最后5个 - 让用户看到循环模式贯穿始终
                    should_print = (idx < 10 or  # 前10个
                                   (idx + 1) % 10 == 0 or  # 每10个
                                   idx >= len(self.image_names) - 5)  # 最后5个
                    if should_print:
                        print(f"   图像 {idx+1}/{len(self.image_names)}: {img_name} -> 俯视角: {dep_cyclic}° (循环模式, 实际: {dep}°), 方位角: {azi}°")
                    # 使用循环俯视角（用于所有计算，包括统一相机半径）
                    dep = dep_cyclic
                    # 保存循环俯视角用于计算统一相机半径
                    actual_depression_angles.append(dep_cyclic)
                elif enable_cyclic_depression and cyclic_depression_angles is None:
                    # 如果启用了循环模式但序列未生成，给出警告
                    if idx == 0:
                        print(f"⚠️ 警告: 循环俯视角模式已启用，但循环序列未生成，使用实际俯视角")
                        print(f"   🔍 调试: enable_cyclic_depression={enable_cyclic_depression}, cyclic_depression_angles={cyclic_depression_angles}")
                else:
                    dep = dep_sar
                    if idx < 5 or (idx + 1) % 50 == 0:
                        if self._use_optical_equiv_dep:
                            print(
                                f"   图像 {idx+1}/{len(self.image_names)}: {img_name} -> "
                                f"SAR俯角: {dep_sar}°, 方位: {azi}°（待换算光学等效俯角）"
                            )
                        else:
                            print(f"   图像 {idx+1}/{len(self.image_names)}: {img_name} -> 俯视角: {dep}°, 方位角: {azi}°")
                    actual_depression_angles.append(dep)

                if self._use_optical_equiv_dep:
                    from sar.optical_equivalent_depression import (
                        compute_optical_equivalent_depression_horizon_deg,
                    )

                    dep_opt, dep_note = compute_optical_equivalent_depression_horizon_deg(
                        dep_sar,
                        self.sar_params,
                        self._optical_equiv_dep_method,
                        empirical_scale=self._optical_equiv_dep_scale,
                    )
                    dep = dep_opt
                    if idx < 3:
                        print(f"      → 光学等效俯角: {dep_opt:.2f}° ({dep_note})")

                self.image_angles[img_name] = {
                    'depression': dep,
                    'azimuth': azi,
                    'depression_sar': dep_sar,
                }
                depression_angles.append(dep)  # 收集俯视角（用于其他用途）
                
                # 读取该图像的实际尺寸
                image_files = [f for f in os.listdir(self.images_dir) 
                              if f.split('.')[-1].lower() in opts.ext and 
                              f.split('.')[0] == img_name]
                if image_files:
                    img_path = os.path.join(self.images_dir, image_files[0])
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        height, width = img.shape
                        self.image_sizes[img_name] = (width, height)
            except:
                # 如果解析失败，使用默认值或循环序列中的值
                if enable_cyclic_depression and cyclic_depression_angles is not None:
                    dep = cyclic_depression_angles[idx]
                    actual_depression_angles.append(dep)  # 保存循环俯视角
                else:
                    dep = 15
                    actual_depression_angles.append(dep)  # 保存默认俯视角
                self.image_angles[img_name] = {'depression': dep, 'azimuth': 0}
                depression_angles.append(dep)  # 使用默认俯视角或循环俯视角
                print(f"⚠️ 警告: 无法解析 {img_name} 的角度信息，使用{'循环' if enable_cyclic_depression and cyclic_depression_angles is not None else '默认'}俯视角 {dep}°")
        
        # 循环俯视角模式总结
        if enable_cyclic_depression and cyclic_depression_angles is not None:
            print(f"\n{'='*70}")
            print(f"✅ 循环俯视角模式应用总结")
            print(f"{'='*70}")
            print(f"  - 处理图像数: {len(self.image_names)}")
            print(f"  - 循环序列长度: {len(cyclic_depression_angles)}")
            print(f"  - 使用的俯视角: {sorted(set(cyclic_depression_angles))}")
            print(f"  - 俯视角统计:")
            from collections import Counter
            dep_counts = Counter(depression_angles)
            for dep_val in sorted(dep_counts.keys()):
                count = dep_counts[dep_val]
                percent = 100 * count / len(depression_angles)
                print(f"    • {dep_val}°: {count}张 ({percent:.1f}%)")
            print(f"  ℹ️ 注意: 为避免日志过长，只显示了部分图像的详细信息")
            print(f"  ✅ 但循环俯视角已应用到所有{len(self.image_names)}张图像！")
            print(f"{'='*70}\n")
        
        # 光学等效俯角：写 mapping（供 GS 与 render_novel 读取）及 SAR 原始俯角备份
        if self._use_optical_equiv_dep and len(self.image_angles) > 0:
            import json
            import glob

            print("\n" + "=" * 70)
            print("📝 保存光学等效俯视角映射（SO-RCG/SAR BA 初始角）")
            print("=" * 70)
            images_dir = os.path.join(opts.data_dir, "images")
            if not os.path.exists(images_dir):
                images_dir = os.path.join(opts.data_dir, "input")
            opt_mapping = {}
            sar_mapping = {}
            for img_name, ang in self.image_angles.items():
                stem = os.path.splitext(img_name)[0]
                opt_mapping[stem] = float(ang["depression"])
                sar_mapping[stem] = float(ang.get("depression_sar", ang["depression"]))
            mapping_file = os.path.normpath(
                os.path.join(opts.data_dir, "depression_angle_mapping.json")
            )
            sar_orig_file = os.path.normpath(
                os.path.join(opts.data_dir, "depression_angle_sar_original.json")
            )
            with open(mapping_file, "w", encoding="utf-8") as f:
                json.dump(opt_mapping, f, indent=2, ensure_ascii=False)
            with open(sar_orig_file, "w", encoding="utf-8") as f:
                json.dump(sar_mapping, f, indent=2, ensure_ascii=False)
            print(f"✅ 光学等效俯角: {mapping_file} ({len(opt_mapping)} 条)")
            print(f"✅ SAR 原始俯角: {sar_orig_file}")
            sample = list(opt_mapping.items())[:3]
            for k, v in sample:
                print(f"   示例 {k}: SAR={sar_mapping.get(k, v):.1f}° → opt={v:.2f}°")
            print("=" * 70 + "\n")

        # 自动计算统一的相机半径（用于保持相机位置不变）
        # 必须与 compute_sar_camera_pose 一致：unified_sita0_for_radius 为与天底入射角（弧度）
        if len(depression_angles) > 0:
            try:
                from sar.geometry import (
                    normalize_mstar_filename_depression_to_incidence_from_nadir_deg,
                )
            except ImportError:
                normalize_mstar_filename_depression_to_incidence_from_nadir_deg = None

            if normalize_mstar_filename_depression_to_incidence_from_nadir_deg is None:
                incidence_list = [float(x) for x in depression_angles]
            else:
                incidence_list = [
                    float(
                        normalize_mstar_filename_depression_to_incidence_from_nadir_deg(
                            float(x), self.sar_params
                        )
                    )
                    for x in depression_angles
                ]
            if len(set(incidence_list)) == 1:
                unified_inc_deg = incidence_list[0]
                print(
                    f"✅ 折合后与天底入射角相同 ({unified_inc_deg:.2f}°)，用于统一相机半径"
                )
            else:
                unified_inc_deg = float(np.mean(incidence_list))
                print(
                    f"✅ 折合入射角不同，均值 {unified_inc_deg:.2f}°（范围 "
                    f"{min(incidence_list):.2f}°–{max(incidence_list):.2f}°）"
                )
            self.unified_sita0_for_radius = np.deg2rad(unified_inc_deg)
            print(
                f"✅ unified_sita0_for_radius = {unified_inc_deg:.2f}°（与 compute_sar_camera_pose 一致）"
            )
        else:
            # 如果没有俯视角信息，抛出错误（不应该使用默认值）
            raise ValueError(f"❌ 错误: 无法从图像获取俯视角信息，请确保图像文件名包含角度信息或启用循环俯视角模式")
        
        # ========== 新增：保存俯视角映射文件（用于Gaussian Splatting训练） ==========
        if enable_cyclic_depression and cyclic_depression_angles is not None:
            print("\n" + "="*70)
            print("📝 保存俯视角映射文件（用于Gaussian Splatting训练）")
            print("="*70)
            
            # ========== 修复：只映射实际存在的图像 ==========
            # 1. 获取实际存在的图像文件
            import glob
            images_dir = os.path.join(opts.data_dir, 'images')
            if not os.path.exists(images_dir):
                images_dir = os.path.join(opts.data_dir, 'input')
            
            actual_image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                actual_image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            
            actual_image_names = set()
            for img_path in actual_image_files:
                img_name = os.path.basename(img_path)
                name_without_ext = os.path.splitext(img_name)[0]
                actual_image_names.add(img_name)
                actual_image_names.add(name_without_ext)
            
            print(f"  🔍 实际图像文件数: {len(actual_image_files)}")
            
            # 2. 创建映射字典：只包含实际存在的图像
            depression_angle_mapping = {}
            skipped = 0
            for idx, img_name in enumerate(self.image_names):
                # 检查图像是否实际存在
                name_without_ext = os.path.splitext(img_name)[0]
                
                if img_name in actual_image_names or name_without_ext in actual_image_names:
                    # 图像存在，添加映射
                    if idx < len(cyclic_depression_angles):
                        depression_angle_mapping[name_without_ext] = float(cyclic_depression_angles[idx])
                else:
                    # 图像不存在，跳过
                    skipped += 1
                    print(f"  ⚠️  跳过不存在的图像: {img_name}")
            
            if skipped > 0:
                print(f"  ℹ️  跳过了 {skipped} 个不存在的图像")
            # ========== 修复结束 ==========
            
            # 保存为JSON文件到数据集根目录
            import json
            mapping_file = os.path.join(opts.data_dir, 'depression_angle_mapping.json')
            mapping_file = os.path.normpath(mapping_file)  # 规范化路径
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(depression_angle_mapping, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 已保存俯视角映射: {mapping_file}")
            print(f"   映射数量: {len(depression_angle_mapping)} 条（只包含实际存在的图像）")
            print(f"   实际图像: {len(actual_image_files)} 张")
            if len(depression_angle_mapping) == len(actual_image_files):
                print(f"   ✅ 映射数量与实际图像数量一致！")
            else:
                print(f"   ⚠️  数量不一致，可能有图像扩展名不同")
            
            # 显示前5个映射示例
            sample_items = list(depression_angle_mapping.items())[:5]
            print(f"   示例映射:")
            for img, dep in sample_items:
                print(f"     - {img}: {dep}°")
            print("="*70 + "\n")
        # ========== 新增结束 ==========
        
        # 计算每张图像的K矩阵（在统一sita0设置之后）
        # 为了保持尺度一致性，使用统一的参考图像尺寸计算K矩阵
        # 使用第一张图像的实际尺寸作为参考（必须从图像读取，不能使用默认值）
        ref_width, ref_height = None, None
        if len(self.image_names) > 0:
            # 优先使用已读取的图像尺寸
            if self.image_names[0] in self.image_sizes:
                ref_width, ref_height = self.image_sizes[self.image_names[0]]
            else:
                # 如果还没有读取，立即读取第一张图像
                image_files = [f for f in os.listdir(self.images_dir) 
                              if f.split('.')[-1].lower() in opts.ext and 
                              f.split('.')[0] == self.image_names[0]]
                if image_files:
                    img_path = os.path.join(self.images_dir, image_files[0])
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        height, width = img.shape
                        ref_width, ref_height = width, height
                        self.image_sizes[self.image_names[0]] = (width, height)
        
        # 如果仍然无法获取图像尺寸，抛出错误（不应该使用默认值）
        if ref_width is None or ref_height is None:
            raise ValueError(f"❌ 错误: 无法读取参考图像尺寸，请确保图像文件存在且可读")

        # MSTAR 像素分辨率 → SO-RCG 初始 camera_height / scene_scale（公式闭合，非事后硬改）
        try:
            from sar.scene_scale_closure import ensure_mstar_initial_sar_params, mstar_init_enabled

            self._mstar_initial_scene = ensure_mstar_initial_sar_params(
                self.sar_params,
                getattr(opts, "_full_cli_args", None) or opts,
                ref_width=ref_width,
                ref_height=ref_height,
                image_angles=self.image_angles,
                source_path=opts.data_dir,
            )
            if mstar_init_enabled(getattr(opts, "_full_cli_args", None) or opts):
                if not self.sar_params.get("mstar_init_from_pixel_resolution"):
                    raise RuntimeError(
                        "MSTAR pixel-resolution init required but not applied "
                        "(check --geometry_prior_gs_model or --sfm_mstar_init_from_pixel_resolution)"
                    )
                ch = float(self.sar_params.get("camera_height", 0.0))
                if ch > 100.0:
                    raise RuntimeError(
                        f"MSTAR init produced invalid scene camera_height={ch:.2g} m (expected ~1-10 m)"
                    )
        except RuntimeError:
            raise
        except Exception as e:
            self._mstar_initial_scene = None
            try:
                from sar.scene_scale_closure import mstar_init_enabled
                _mstar_wanted = mstar_init_enabled(getattr(opts, "_full_cli_args", None) or opts)
            except Exception:
                _mstar_wanted = getattr(opts, "sfm_mstar_init_from_pixel_resolution", False)
            if _mstar_wanted:
                raise RuntimeError(f"MSTAR initial scale failed: {e}") from e

        # ========== 重要修改：计算统一参数供GS训练使用 ==========
        # 无论使用哪种SfM方法，都需要计算并保存统一的焦距参数
        # 因为GS训练时需要这些参数来确保相机内参与SfM一致
        
        # 统一焦距尺度：MSTAR 初值路径用 focal_scale=0.5（128→f=64）；否则默认 1.5
        unified_focal_scale = self.sar_params.get('focal_scale', 1.5)
        unified_focal_azimuth = ref_width * unified_focal_scale
        unified_focal_range = ref_height * unified_focal_scale
        # 统一K矩阵的中心点（使用参考图像尺寸）
        unified_cx = ref_width / 2.0
        unified_cy = ref_height / 2.0
        # 统一radius_scale的计算，使用参考图像尺寸
        camera_height = self.sar_params['camera_height']
        if hasattr(self, 'unified_sita0_for_radius'):
            sita0_for_radius = self.unified_sita0_for_radius
            sita0_deg = np.rad2deg(sita0_for_radius)
        else:
            ref_dep = 30.0
            if hasattr(self, 'image_angles') and len(self.image_angles) > 0:
                first_img_name = list(self.image_angles.keys())[0]
                ref_dep = self.image_angles[first_img_name].get('depression', 30.0)
            norm_fn = normalize_mstar_filename_depression_to_incidence_from_nadir_deg
            if norm_fn is not None:
                try:
                    sita0_deg = norm_fn(float(ref_dep), self.sar_params)
                except Exception:
                    sita0_deg = 90.0 - float(ref_dep)
            else:
                sita0_deg = 90.0 - float(ref_dep)
            sita0_for_radius = np.deg2rad(sita0_deg)

        # 斜距 R0 = H / cos(θ_nadir)，与 sar/geometry.py 一致
        R0 = camera_height / max(float(np.cos(sita0_for_radius)), 1e-9)
        scene_scale = self.sar_params.get('scene_scale', 0.1)
        image_size = max(ref_width, ref_height)
        if self.sar_params.get('radius_scale') is not None and self.sar_params.get('mstar_init_from_pixel_resolution'):
            unified_radius_scale = float(self.sar_params['radius_scale'])
        else:
            unified_radius_scale = (image_size * scene_scale) / R0 if R0 > 0 else 1.0
        self.sar_params['radius_scale'] = unified_radius_scale
        self.sar_params['default_radius_scale'] = unified_radius_scale
        
        print(f"\n{'='*70}")
        print(f"📐 计算统一radius_scale（基于实际数据）")
        print(f"{'='*70}")
        print(f"输入参数:")
        print(f"  - 实际图像尺寸: {ref_width} × {ref_height} 像素")
        print(f"  - 相机高度: {camera_height:.2f} m")
        print(f"  - 统一天底入射角 θ（用于 R0）: {sita0_deg:.2f}°")
        print(f"  - 场景倍数 scene_scale: {scene_scale:.5f}")
        print(f"\n计算过程:")
        print(f"  1. R0 = camera_height / cos(θ_nadir)")
        print(f"     R0 = {camera_height:.4f} / cos({sita0_deg:.2f}°)")
        print(f"     R0 = {camera_height:.4f} / {np.cos(sita0_for_radius):.4f}")
        print(f"     R0 = {R0:.2f} m")
        print(f"\n  2. image_size = max(width, height)")
        print(f"     image_size = max({ref_width}, {ref_height})")
        print(f"     image_size = {image_size} 像素")
        print(f"\n  3. radius_scale = (image_size × scene_scale) / R0")
        print(f"     radius_scale = ({image_size} × {scene_scale}) / {R0:.2f}")
        print(f"     radius_scale = {image_size * scene_scale} / {R0:.2f}")
        print(f"     radius_scale = {unified_radius_scale:.6f}")
        print(f"\n结果:")
        print(f"  ✅ radius_scale = {unified_radius_scale:.4f}")
        print(f"  ✅ 球面半径 = R0 × radius_scale = {R0:.2f} × {unified_radius_scale:.4f} = {R0 * unified_radius_scale:.2f} m")
        print(f"{'='*70}\n")
        
        print(f"📐 统一参数汇总（供GS训练使用）:")
        print(f"   图像尺寸: {ref_width} × {ref_height}")
        print(f"   焦距: 方位向={unified_focal_azimuth:.2f}, 距离向={unified_focal_range:.2f}")
        print(f"   中心点: cx={unified_cx:.2f}, cy={unified_cy:.2f}")
        print(f"   半径缩放: {unified_radius_scale:.4f}")
        print(f"   参考斜距R0: {R0:.2f} m")
        print(f"   缩放后球面半径: {R0 * unified_radius_scale:.2f} m")
        
        # 保存为SAR尺度参数供GS训练使用
        from sar.sfm_scale_params import build_initial_scale_params, merge_scale_params_file

        scale_params_file = os.path.join(opts.data_dir, 'sar_scale_params.json')
        scale_params = build_initial_scale_params(
            unified_radius_scale=unified_radius_scale,
            scene_scale=scene_scale,
            unified_focal_azimuth=unified_focal_azimuth,
            unified_focal_range=unified_focal_range,
            unified_cx=unified_cx,
            unified_cy=unified_cy,
            camera_height=camera_height,
            ref_width=ref_width,
            ref_height=ref_height,
            R0=R0,
            sita0_deg=sita0_deg,
            sar_params=self.sar_params,
        )
        
        # 添加所有相机数据
        if hasattr(self, 'image_data') and len(self.image_data) > 0:
            try:
                cameras_dict = self._CollectCameraData()
                scale_params['cameras'] = cameras_dict
                print(f"📷 已添加 {len(cameras_dict)} 个相机的数据到尺度参数文件")
            except Exception as e:
                print(f"⚠️  收集相机数据时出错: {e}")
                import traceback
                traceback.print_exc()
        
        merge_scale_params_file(scale_params_file, scale_params)
        print(f"✅ 已保存SAR尺度参数（初始版本）: {scale_params_file}")
        # ========== 修改结束 ==========
        
        # 检查是否使用SO-RCG方法
        use_sorc = getattr(self.opts, 'sfm_use_sorc', False)
        
        # 🔧 重要修复：radius_scale应该始终使用统一值
        # radius_scale影响相机位置，必须保持一致，否则相机会在不同尺度的球面上
        # 这与焦距不同，焦距可以因图像而异，但相机位置必须一致
        self.unified_radius_scale = unified_radius_scale
        
        if use_sorc and SORCG_AVAILABLE:
            # SO-RCG方法：SfM过程中K矩阵参数使用每张图像的实际参数
            # 但radius_scale必须使用统一值以保持相机位置一致
            print(f"ℹ️ 使用SO-RCG方法，K矩阵将使用每张图像的实际参数")
            print(f"   但radius_scale={unified_radius_scale:.4f}保持统一（确保相机位置一致）")
            # 不设置统一K矩阵参数用于SfM，让SO-RCG方法使用每张图像的实际K矩阵
            self.unified_focal_azimuth = None
            self.unified_focal_range = None
            self.unified_cx = None
            self.unified_cy = None
            # 注意：unified_radius_scale必须保留！
        else:
            # 矩阵分解方法：SfM过程中使用统一参数保持尺度一致性
            print(f"ℹ️ 使用矩阵分解方法，SfM过程和GS训练都将使用统一参数")
            # 保存统一的焦距值、中心点和半径缩放因子，供SfM流程使用
            self.unified_focal_azimuth = unified_focal_azimuth
            self.unified_focal_range = unified_focal_range
            self.unified_cx = unified_cx
            self.unified_cy = unified_cy
            # unified_radius_scale已在上面设置
        
        for img_name in self.image_names:
            if img_name in self.image_angles:
                dep = self.image_angles[img_name]['depression']
                azi = self.image_angles[img_name]['azimuth']
                
                # 获取该图像的实际尺寸（必须从图像读取，不能使用默认值）
                if img_name in self.image_sizes:
                    width, height = self.image_sizes[img_name]
                else:
                    # 如果还没有读取，立即读取该图像
                    image_files = [f for f in os.listdir(self.images_dir) 
                                  if f.split('.')[-1].lower() in opts.ext and 
                                  f.split('.')[0] == img_name]
                    if image_files:
                        img_path = os.path.join(self.images_dir, image_files[0])
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            height, width = img.shape
                            self.image_sizes[img_name] = (width, height)
                        else:
                            raise ValueError(f"❌ 错误: 无法读取图像 {img_name} 的尺寸")
                    else:
                        raise ValueError(f"❌ 错误: 找不到图像文件 {img_name}")
                
                # 使用实际图像尺寸计算该图像的K矩阵
                # 为每张图像使用其对应的俯视角作为中心下视角
                temp_params = self.sar_params.copy()
                temp_params['Na'] = width
                temp_params['Nr'] = height
                # 不再设置sita0，直接使用depression_angle作为俯视角
                
                # 根据方法选择是否使用统一参数
                if use_sorc and SORCG_AVAILABLE:
                    # SO-RCG方法：不使用统一参数，使用每张图像的实际参数
                    # 不设置unified_focal、unified_cx、unified_cy、radius_scale
                    # 让compute_sar_camera_pose使用每张图像的实际尺寸和角度计算参数
                    pass  # 不设置统一参数
                else:
                    # 矩阵分解方法：使用统一参数保持尺度一致性
                    # 设置统一的sita0用于计算相机半径（保持相机位置不变）
                    temp_params['unified_sita0_for_radius'] = self.unified_sita0_for_radius
                    # 使用统一的焦距尺度，确保K矩阵尺度一致
                    if hasattr(self, 'unified_focal_azimuth') and self.unified_focal_azimuth is not None:
                        temp_params['unified_focal_azimuth'] = self.unified_focal_azimuth
                        temp_params['unified_focal_range'] = self.unified_focal_range
                        temp_params['unified_cx'] = self.unified_cx
                        temp_params['unified_cy'] = self.unified_cy
                    # 使用统一的半径缩放因子，确保所有相机在同一球面上
                    if hasattr(self, 'unified_radius_scale') and self.unified_radius_scale is not None:
                        temp_params['radius_scale'] = self.unified_radius_scale
                        temp_params['default_radius_scale'] = self.unified_radius_scale
                
                _, _, K_img = compute_sar_camera_pose(dep, azi, temp_params)
                self.image_K[img_name] = K_img
        
        # 尝试创建匹配器，如果不是OpenCV标准匹配器则使用默认的BFMatcher
        # 注意：对于SAR_SIFT等自定义特征，匹配已经完成并保存，这里只是占位
        try:
            if hasattr(cv2, opts.matcher):
                self.matcher = getattr(cv2, opts.matcher)(crossCheck=opts.cross_check)
            else:
                # 如果不是OpenCV标准匹配器，使用默认的BFMatcher（实际不会使用）
                print(f"⚠️ 警告: {opts.matcher} 不是OpenCV标准匹配器，使用BFMatcher作为占位（实际匹配已从文件加载）")
                self.matcher = cv2.BFMatcher(crossCheck=opts.cross_check)
        except Exception as e:
            print(f"⚠️ 警告: 创建匹配器失败: {e}，使用BFMatcher作为占位（实际匹配已从文件加载）")
            self.matcher = cv2.BFMatcher(crossCheck=opts.cross_check)

        # SAR模式：使用SAR几何模型计算相机内参（使用第一张图像作为默认K）
        if len(self.image_names) > 0:
            first_name = self.image_names[0]
            if first_name in self.image_K:
                self.K = self.image_K[first_name]
                print(f"✅ 使用SAR几何模型计算相机内参（第一张图像）: {self.K}")
            elif first_name in self.image_angles:
                dep = self.image_angles[first_name]['depression']
                azi = self.image_angles[first_name]['azimuth']
                
                # 获取第一张图像的实际尺寸（必须从图像读取）
                if first_name in self.image_sizes:
                    width, height = self.image_sizes[first_name]
                else:
                    # 如果还没有读取，立即读取
                    image_files = [f for f in os.listdir(self.images_dir) 
                                  if f.split('.')[-1].lower() in opts.ext and 
                                  f.split('.')[0] == first_name]
                    if image_files:
                        img_path = os.path.join(self.images_dir, image_files[0])
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            height, width = img.shape
                            self.image_sizes[first_name] = (width, height)
                        else:
                            raise ValueError(f"❌ 错误: 无法读取图像 {first_name} 的尺寸")
                    else:
                        raise ValueError(f"❌ 错误: 找不到图像文件 {first_name}")
                
                # 为第一张图像使用其对应的俯视角作为中心下视角
                temp_params = self.sar_params.copy()
                temp_params['Na'] = width
                temp_params['Nr'] = height
                # 不再设置sita0，直接使用depression_angle作为俯视角
                
                # 根据方法选择是否使用统一参数
                if use_sorc and SORCG_AVAILABLE:
                    # SO-RCG方法：不使用统一参数
                    pass  # 不设置统一参数
                else:
                    # 矩阵分解方法：使用统一参数
                    # 设置统一的sita0用于计算相机半径（保持相机位置不变）
                    if hasattr(self, 'unified_sita0_for_radius'):
                        temp_params['unified_sita0_for_radius'] = self.unified_sita0_for_radius
                    # 使用统一的焦距（如果已设置）
                    if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                        if self.unified_focal_azimuth is not None:
                            temp_params['unified_focal_azimuth'] = self.unified_focal_azimuth
                            temp_params['unified_focal_range'] = self.unified_focal_range
                            temp_params['unified_cx'] = self.unified_cx
                            temp_params['unified_cy'] = self.unified_cy
                
                _, _, self.K = compute_sar_camera_pose(dep, azi, temp_params)
                self.image_K[first_name] = self.K
                print(f"✅ 使用SAR几何模型计算相机内参（第一张图像）: {self.K}")
            else:
                # 回退到从图像尺寸计算
                if os.path.exists(self.images_dir):
                    image_files = [f for f in os.listdir(self.images_dir) 
                                  if f.split('.')[-1].lower() in opts.ext]
                    if image_files:
                        first_image_path = os.path.join(self.images_dir, image_files[0])
                        first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
                        if first_image is not None:
                            height, width = first_image.shape
                            # SAR图像使用斜距作为等效焦距
                            # 根据相机高度和俯视角计算参考斜距
                            # 注意：俯视角（Depression angle）是从水平方向测量的角度
                            camera_height = self.sar_params['camera_height']
                            # 使用第一张图像的俯视角
                            dep = 30.0  # 默认值
                            if hasattr(self, 'image_angles') and len(self.image_angles) > 0:
                                first_img_name = list(self.image_angles.keys())[0]
                                dep = self.image_angles[first_img_name].get('depression', 30.0)
                            dep_rad = np.deg2rad(dep)
                            R0_computed = camera_height / np.cos(dep_rad)  # 计算参考斜距
                            focal_length = R0_computed
                            self.K = np.array([
                                [focal_length, 0, width / 2.0],
                                [0, focal_length, height / 2.0],
                                [0, 0, 1]
                            ])
                            self.image_K[first_name] = self.K
                            print(f"✅ 使用SAR参数计算相机内参: {self.K} (相机高度={camera_height}m, 计算斜距={R0_computed:.2f}m)")
                        else:
                            raise ValueError("无法读取图像以计算标定矩阵")
                    else:
                        raise ValueError("图像目录为空，无法计算标定矩阵")
                else:
                    raise ValueError("图像目录不存在，无法计算标定矩阵")
        else:
            raise ValueError("没有找到图像文件")
        
    def _LoadFeatures(self, name): 
        # Python 3兼容性：使用'rb'模式读取二进制文件
        with open(os.path.join(self.feat_dir,'kp_{}.pkl'.format(name)),'rb') as f: 
            kp = pickle.load(f)
        kp = DeserializeKeypoints(kp)

        with open(os.path.join(self.feat_dir,'desc_{}.pkl'.format(name)),'rb') as f: 
            desc = pickle.load(f)

        return kp, desc 

    def _GetNormalizedMatchKey(self, name1, name2):
        """
        统一匹配对的方向：返回规范化的匹配对键（较小的图像名在前）
        
        Args:
            name1, name2: 图像名称
        
        Returns:
            (normalized_name1, normalized_name2): 规范化的匹配对键
        """
        if name1 < name2:
            return (name1, name2)
        else:
            return (name2, name1)
    
    def _LoadMatches(self, name1, name2): 
        """
        加载匹配文件，支持双向文件名查找（name1_name2 或 name2_name1）
        在加载时去除重复匹配（基于queryIdx和trainIdx）
        如果文件不存在，返回空列表
        """
        # 尝试两种文件名顺序
        match_file1 = os.path.join(self.matches_dir, 'match_{}_{}.pkl'.format(name1, name2))
        match_file2 = os.path.join(self.matches_dir, 'match_{}_{}.pkl'.format(name2, name1))
        
        match_file = None
        if os.path.exists(match_file1):
            match_file = match_file1
        elif os.path.exists(match_file2):
            match_file = match_file2
        else:
            # 文件不存在，返回空列表
            return []
        
        try:
            # Python 3兼容性：使用'rb'模式读取二进制文件
            with open(match_file, 'rb') as f: 
                matches = pickle.load(f)
            matches = DeserializeMatches(matches)
            
            # 如果使用的是反向文件名（name2_name1），需要交换queryIdx和trainIdx
            if match_file == match_file2:
                # 交换匹配的索引
                swapped_matches = []
                for m in matches:
                    new_match = cv2.DMatch()
                    new_match.queryIdx = m.trainIdx  # 交换
                    new_match.trainIdx = m.queryIdx   # 交换
                    new_match.imgIdx = m.imgIdx
                    new_match.distance = m.distance
                    swapped_matches.append(new_match)
                matches = swapped_matches
            
            # ========== 去除重复匹配 ==========
            # 使用字典存储唯一匹配，key为(queryIdx, trainIdx)，value为匹配对象
            # 如果存在重复，保留距离更小的匹配
            unique_matches_dict = {}
            duplicate_count = 0
            
            for m in matches:
                match_key = (m.queryIdx, m.trainIdx)
                if match_key in unique_matches_dict:
                    # 如果已存在，保留距离更小的匹配
                    existing_match = unique_matches_dict[match_key]
                    if m.distance < existing_match.distance:
                        unique_matches_dict[match_key] = m
                    duplicate_count += 1
                else:
                    unique_matches_dict[match_key] = m
            
            # 转换回列表
            unique_matches = list(unique_matches_dict.values())
            
            if duplicate_count > 0:
                print(f"   🔍 {name1} 和 {name2}: 去除了 {duplicate_count} 个重复匹配 ({len(matches)} -> {len(unique_matches)})")
            
            return unique_matches
        except Exception as e:
            print(f"   ⚠️ 警告: 加载匹配文件 {match_file} 时出错: {e}")
            return []

    def _GetAlignedMatches(self,kp1,desc1,kp2,desc2,matches):
        # 检查特征点是否为空
        if kp1 is None or len(kp1) == 0 or kp2 is None or len(kp2) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), np.array([])
        
        # 筛选出索引在有效范围内的匹配
        valid_matches = []
        for m in matches:
            # 检查queryIdx和trainIdx是否在有效范围内
            if (m.queryIdx >= 0 and m.queryIdx < len(kp1) and 
                m.trainIdx >= 0 and m.trainIdx < len(kp2)):
                valid_matches.append(m)
        
        if len(valid_matches) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2)), np.array([]), np.array([])
        
        img1idx = np.array([m.queryIdx for m in valid_matches])
        img2idx = np.array([m.trainIdx for m in valid_matches])

        #filtering out the keypoints that were matched. 
        kp1_ = (np.array(kp1))[img1idx]
        kp2_ = (np.array(kp2))[img2idx]

        #retreiving the image coordinates of matched keypoints
        img1pts = np.array([kp.pt for kp in kp1_])
        img2pts = np.array([kp.pt for kp in kp2_])

        return img1pts, img2pts, img1idx, img2idx

    def _BaselinePoseEstimation(self, name1, name2):

        kp1, desc1 = self._LoadFeatures(name1)
        kp2, desc2 = self._LoadFeatures(name2)  

        matches = self._LoadMatches(name1, name2)
        matches = sorted(matches, key = lambda x:x.distance)

        img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
                                                                    desc2,matches)
        
        # SAR几何模型：从角度信息计算SAR相机姿态
        if name1 in self.image_angles and name2 in self.image_angles:
            dep1 = self.image_angles[name1]['depression']
            azi1 = self.image_angles[name1]['azimuth']
            dep2 = self.image_angles[name2]['depression']
            azi2 = self.image_angles[name2]['azimuth']
            
            # 使用SAR几何计算姿态（使用实际图像尺寸）
            # 获取图像尺寸
            width1, height1 = self.image_sizes.get(name1, (self.sar_params['Na'], self.sar_params['Nr']))
            width2, height2 = self.image_sizes.get(name2, (self.sar_params['Na'], self.sar_params['Nr']))
            
            # 使用实际图像尺寸更新SAR参数
            params1 = self.sar_params.copy()
            params1['Na'] = width1
            params1['Nr'] = height1
            params1['sita0'] = np.deg2rad(dep1)  # 使用图像1的俯视角作为中心下视角
            # 设置统一的sita0用于计算相机半径（保持相机位置不变）
            if hasattr(self, 'unified_sita0_for_radius'):
                params1['unified_sita0_for_radius'] = self.unified_sita0_for_radius
            # 使用统一的焦距和中心点（如果已设置）
            if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                params1['unified_focal_azimuth'] = self.unified_focal_azimuth
                params1['unified_focal_range'] = self.unified_focal_range
            if hasattr(self, 'unified_cx') and hasattr(self, 'unified_cy'):
                params1['unified_cx'] = self.unified_cx
                params1['unified_cy'] = self.unified_cy
            # 使用统一的半径缩放因子，确保所有相机在同一球面上
            if hasattr(self, 'unified_radius_scale') and self.unified_radius_scale is not None:
                params1['radius_scale'] = self.unified_radius_scale
                params1['default_radius_scale'] = self.unified_radius_scale
            
            params2 = self.sar_params.copy()
            params2['Na'] = width2
            params2['Nr'] = height2
            params2['sita0'] = np.deg2rad(dep2)  # 使用图像2的俯视角作为中心下视角
            # 设置统一的sita0用于计算相机半径（保持相机位置不变）
            if hasattr(self, 'unified_sita0_for_radius'):
                params2['unified_sita0_for_radius'] = self.unified_sita0_for_radius
            # 使用统一的焦距和中心点（如果已设置）
            if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                params2['unified_focal_azimuth'] = self.unified_focal_azimuth
                params2['unified_focal_range'] = self.unified_focal_range
            if hasattr(self, 'unified_cx') and hasattr(self, 'unified_cy'):
                params2['unified_cx'] = self.unified_cx
                params2['unified_cy'] = self.unified_cy
            # 使用统一的半径缩放因子，确保所有相机在同一球面上
            if hasattr(self, 'unified_radius_scale') and self.unified_radius_scale is not None:
                params2['radius_scale'] = self.unified_radius_scale
                params2['default_radius_scale'] = self.unified_radius_scale
            
            R1, t1, K1 = compute_sar_camera_pose(dep1, azi1, params1)
            R2, t2, K2 = compute_sar_camera_pose(dep2, azi2, params2)
            
            # 保存每张图像的K矩阵
            self.image_K[name1] = K1
            self.image_K[name2] = K2
            
            # 使用匹配点优化姿态（可选）
            num_matches = len(img1pts)
            skip_fundamental = getattr(self.opts, 'sfm_skip_fundamental_matrix', False)
            # SAR图像匹配困难，当匹配点少于8个时自动跳过基础矩阵验证（不输出警告）
            if not skip_fundamental and num_matches >= 8:
                try:
                    # 尝试使用RANSAC估计基础矩阵来验证和微调
                    F, mask = cv2.findFundamentalMat(img1pts, img2pts,
                                                    method=self.opts.fund_method,
                                                    ransacReprojThreshold=self.opts.outlier_thres,
                                                    confidence=self.opts.fund_prob)
                    
                    if F is not None and mask is not None:
                        mask = mask.astype(bool).flatten()
                        num_inliers = np.sum(mask)
                        inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
                        
                        min_inliers = getattr(self.opts, 'min_inliers_for_triangulation', 1)  # 默认1，SAR图像建议低值
                        if num_inliers >= min_inliers:
                            # 使用匹配点进行微调（但主要使用从角度计算的姿态）
                            # 这里可以选择性地使用F来验证姿态
                            if inlier_ratio < 0.5:
                                print(f"   ⚠️ 注意: {name1} 和 {name2} 之间的基础矩阵内点比例较低 ({inlier_ratio:.2%})")
                        else:
                            print(f"   ⚠️ 注意: {name1} 和 {name2} 之间的基础矩阵内点数量不足 ({num_inliers} < {min_inliers})")
                    else:
                        print(f"   ⚠️ 注意: {name1} 和 {name2} 之间的基础矩阵估计失败，但使用SAR几何计算的姿态")
                except Exception as e:
                    print(f"   ⚠️ 注意: {name1} 和 {name2} 之间的基础矩阵估计异常: {e}，使用SAR几何计算的姿态")
                    mask = np.ones(len(img1pts), dtype=bool)
            elif skip_fundamental:
                # 如果禁用了基础矩阵验证，直接使用所有匹配点
                mask = np.ones(len(img1pts), dtype=bool)
                print(f"   ℹ️ 跳过基础矩阵验证（已禁用），使用所有 {num_matches} 个匹配点")
            else:
                min_matches = getattr(self.opts, 'min_matches_for_triangulation', 1)
                if num_matches < min_matches:
                    print(f"   ⚠️ 注意: {name1} 和 {name2} 之间的匹配点数量不足 ({num_matches} < {min_matches})，跳过基础矩阵验证")
                mask = np.ones(len(img1pts), dtype=bool)
            
            # 设置第一张图像为世界坐标系原点
            # R1和t1是从世界坐标系到相机1坐标系的变换（从compute_sar_camera_pose得到）
            # 为了统一世界坐标系，我们设置：
            # - 第一张图像：R1_world = R1（从世界到相机1），t1_world = t1
            # - 第二张图像：R2_world = R2（从世界到相机2），t2_world = t2
            # 
            # 重要：compute_sar_camera_pose返回的R和t遵循OpenCV/SfM约定：
            # - R: 从世界坐标系到相机坐标系的旋转矩阵
            # - t: 满足 t = -R @ C，其中C是相机中心在世界坐标系中的位置
            # 所以相机位置C = -R.T @ t（在保存时会进行此转换）
            
            R1_world = R1  # 从世界到相机1的旋转矩阵
            t1_world = t1  # 平移向量 t1 = -R1 @ C1，其中C1是相机1在世界坐标系中的位置
            
            R2_world = R2  # 从世界到相机2的旋转矩阵
            t2_world = t2  # 平移向量 t2 = -R2 @ C2，其中C2是相机2在世界坐标系中的位置
            
            self.image_data[name1] = [R1_world, t1_world, np.ones((len(kp1),))*-1]
            self.image_data[name2] = [R2_world, t2_world, np.ones((len(kp2),))*-1]
            
            # 保存匹配数据（使用统一的方向键）
            match_key = self._GetNormalizedMatchKey(name1, name2)
            if 'F' in locals() and F is not None and 'mask' in locals() and mask is not None:
                mask = mask.astype(bool).flatten()
                # 检查是否已存在，如果存在则跳过（避免重复存储）
                if match_key not in self.matches_data:
                    self.matches_data[match_key] = [matches, img1pts[mask], img2pts[mask], 
                                                    img1idx[mask], img2idx[mask]]
            else:
                # 检查是否已存在，如果存在则跳过（避免重复存储）
                if match_key not in self.matches_data:
                    self.matches_data[match_key] = [matches, img1pts, img2pts, 
                                                    img1idx, img2idx]
            
            return R2_world, t2_world
        else:
            raise ValueError(f"❌ 错误: 无法获取 {name1} 或 {name2} 的角度信息，SAR重建需要角度信息")

    def _TriangulateTwoViews(self, name1, name2): 

        def __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2): 
            # SAR几何模型：使用SAR立体视觉三角化
            # 获取过滤配置
            filter_behind = getattr(self.opts, 'filter_behind_camera', False)
            
            if name1 in self.image_angles and name2 in self.image_angles:
                dep1 = self.image_angles[name1]['depression']
                azi1 = self.image_angles[name1]['azimuth']
                dep2 = self.image_angles[name2]['depression']
                azi2 = self.image_angles[name2]['azimuth']
                
                # 重新计算SAR相机内参（使用实际图像尺寸）
                width1, height1 = self.image_sizes.get(name1, (self.sar_params['Na'], self.sar_params['Nr']))
                width2, height2 = self.image_sizes.get(name2, (self.sar_params['Na'], self.sar_params['Nr']))
                
                params1 = self.sar_params.copy()
                params1['Na'] = width1
                params1['Nr'] = height1
                params1['sita0'] = np.deg2rad(dep1)  # 使用图像1的俯视角作为中心下视角
                # 设置统一的sita0用于计算相机半径（保持相机位置不变）
                if hasattr(self, 'unified_sita0_for_radius'):
                    params1['unified_sita0_for_radius'] = self.unified_sita0_for_radius
                # 使用统一的焦距（如果已设置）
                if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                    params1['unified_focal_azimuth'] = self.unified_focal_azimuth
                    params1['unified_focal_range'] = self.unified_focal_range
                if hasattr(self, 'unified_cx') and hasattr(self, 'unified_cy'):
                    params1['unified_cx'] = self.unified_cx
                    params1['unified_cy'] = self.unified_cy
                # 使用统一的半径缩放因子，确保所有相机在同一球面上
                if hasattr(self, 'unified_radius_scale') and self.unified_radius_scale is not None:
                    params1['radius_scale'] = self.unified_radius_scale
                    params1['default_radius_scale'] = self.unified_radius_scale
                
                params2 = self.sar_params.copy()
                params2['Na'] = width2
                params2['Nr'] = height2
                params2['sita0'] = np.deg2rad(dep2)  # 使用图像2的俯视角作为中心下视角
                # 设置统一的sita0用于计算相机半径（保持相机位置不变）
                if hasattr(self, 'unified_sita0_for_radius'):
                    params2['unified_sita0_for_radius'] = self.unified_sita0_for_radius
                # 使用统一的焦距（如果已设置）
                if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                    params2['unified_focal_azimuth'] = self.unified_focal_azimuth
                    params2['unified_focal_range'] = self.unified_focal_range
                if hasattr(self, 'unified_cx') and hasattr(self, 'unified_cy'):
                    params2['unified_cx'] = self.unified_cx
                    params2['unified_cy'] = self.unified_cy
                # 使用统一的半径缩放因子，确保所有相机在同一球面上
                if hasattr(self, 'unified_radius_scale') and self.unified_radius_scale is not None:
                    params2['radius_scale'] = self.unified_radius_scale
                    params2['default_radius_scale'] = self.unified_radius_scale
                
                # 重新计算R1, t1, R2, t2，确保使用修正后的俯视角和统一的参数
                R1_new, t1_new, K1 = compute_sar_camera_pose(dep1, azi1, params1)
                R2_new, t2_new, K2 = compute_sar_camera_pose(dep2, azi2, params2)
                
                # 使用新计算的R和t（基于修正后的俯视角）
                R1 = R1_new
                t1 = t1_new
                R2 = R2_new
                t2 = t2_new
                
                # 确保K矩阵已保存
                if name1 not in self.image_K:
                    self.image_K[name1] = K1
                if name2 not in self.image_K:
                    self.image_K[name2] = K2
                
                # 使用SAR立体视觉三角化
                num_points = len(img1pts)
                if num_points == 0:
                    print(f"   ⚠️ 警告: {name1} 和 {name2} 之间没有匹配点，无法进行三角化")
                    return np.zeros((0, 3)), np.array([], dtype=bool)
                
                try:
                    # 创建统一的sar_params用于三角化，确保使用正确的图像尺寸
                    triangulation_params = self.sar_params.copy()
                    # 使用第一张图像的尺寸作为参考（确保一致性）
                    # 注意：虽然K1和K2可能基于不同的图像尺寸，但三角化时应该使用统一的参数
                    # 实际上，由于我们使用了统一的焦距，K矩阵的尺度应该是一致的
                    # 但为了安全，我们使用第一张图像的尺寸
                    triangulation_params['Na'] = width1
                    triangulation_params['Nr'] = height1
                    # 确保使用统一的焦距参数（如果已设置）
                    if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                        # 注意：sar_stereo_triangulation内部不使用这些参数，但为了保持一致性
                        pass
                    pts3d = sar_stereo_triangulation(
                        img1pts, img2pts, R1, t1, R2, t2, K1, K2,
                        dep1, azi1, dep2, azi2, triangulation_params
                    )
                    
                    # 检查三角化结果
                    if pts3d is None or len(pts3d) == 0:
                        print(f"   ⚠️ 警告: {name1} 和 {name2} 之间的三角化结果为空")
                        return np.zeros((0, 3)), np.array([], dtype=bool)
                    
                    # 跟踪有效点的mask（相对于原始输入img1pts/img2pts）
                    # 初始所有点都有效
                    original_valid_mask = np.ones(len(img1pts), dtype=bool)
                    
                    # 检查是否有无效点（NaN或Inf）
                    valid_finite_mask = np.isfinite(pts3d).all(axis=1)
                    num_valid_finite = np.sum(valid_finite_mask)
                    if num_valid_finite < len(pts3d):
                        print(f"   ⚠️ 警告: {name1} 和 {name2} 之间的三角化结果包含 {len(pts3d) - num_valid_finite} 个无效点")
                        pts3d = pts3d[valid_finite_mask]
                        # 更新原始mask：只有有限值点保留
                        original_valid_mask = original_valid_mask & valid_finite_mask
                    
                    # 检查点的深度（在相机坐标系中的Z坐标）
                    # 根据COLMAP规范：depth = (R * P_world + t).z
                    # 深度必须为正（点在相机前方）
                    if len(pts3d) > 0:
                        # 将3D点转换到相机1坐标系
                        # 公式：P_cam = R @ P_world + t
                        pts3d_cam1 = (R1 @ pts3d.T + t1).T  # 转换到相机1坐标系
                        depth1 = pts3d_cam1[:, 2]  # 相机1坐标系中的深度（Z坐标）
                        
                        # 将3D点转换到相机2坐标系
                        pts3d_cam2 = (R2 @ pts3d.T + t2).T  # 转换到相机2坐标系
                        depth2 = pts3d_cam2[:, 2]  # 相机2坐标系中的深度（Z坐标）
                        
                        # 检查深度是否为正（点在两个相机前方）
                        valid_depth_mask = (depth1 > 0) & (depth2 > 0)
                        num_behind_camera = np.sum(~valid_depth_mask)
                        
                        if num_behind_camera > 0:
                            num_behind_cam1 = np.sum(depth1 <= 0)
                            num_behind_cam2 = np.sum(depth2 <= 0)
                            print(f"   ⚠️ 注意: {name1} 和 {name2} 之间的三角化结果包含 {num_behind_camera} 个在相机后方的点")
                            print(f"      相机1后方: {num_behind_cam1} 个, 相机2后方: {num_behind_cam2} 个")
                            
                            # 根据配置决定是否过滤相机后方的点
                            if filter_behind and num_behind_camera > 0:
                                pts3d = pts3d[valid_depth_mask]
                                # 更新原始mask：只有深度有效的点保留
                                # valid_depth_mask是相对于当前pts3d的，需要映射回原始输入
                                # 找到当前有效点在原始mask中的位置
                                current_valid_indices = np.where(original_valid_mask)[0]
                                if len(current_valid_indices) > 0:
                                    # 只保留深度有效的点
                                    depth_valid_indices = current_valid_indices[valid_depth_mask]
                                    original_valid_mask = np.zeros(len(img1pts), dtype=bool)
                                    original_valid_mask[depth_valid_indices] = True
                                print(f"   ✅ 已过滤 {num_behind_camera} 个在相机后方的点，剩余 {len(pts3d)} 个点")
                            else:
                                # 即使不过滤，也记录深度统计信息
                                print(f"   📊 深度统计: 相机1深度范围=[{depth1.min():.2f}, {depth1.max():.2f}], "
                                      f"相机2深度范围=[{depth2.min():.2f}, {depth2.max():.2f}]")
                    
                    # 🔧 修复双层点云问题：添加深度一致性检查
                    # 过滤掉Z坐标异常的点（可能来自深度歧义）
                    # ⚠️ 注意：此过滤只基于Z坐标，不会影响X或Y方向的点
                    # 如果点云缺少某个方向的部分（如坦克的前半部分），可能是其他原因：
                    # 1. 某些视角的匹配点不足
                    # 2. 三角化时某些区域没有被覆盖
                    # 3. 点云合并时某些点被错误合并
                    if len(pts3d) > 0:
                        # 计算点云的Z坐标统计信息
                        z_coords = pts3d[:, 2]
                        z_median = np.median(z_coords)
                        z_std = np.std(z_coords)
                        
                        # 使用IQR方法检测异常值（更鲁棒）
                        # 🔧 放宽IQR倍数：从1.5改为3.0，减少误过滤
                        z_q1 = np.percentile(z_coords, 25)
                        z_q3 = np.percentile(z_coords, 75)
                        z_iqr = z_q3 - z_q1
                        iqr_multiplier = 3.0  # 放宽到3.0倍IQR（原来1.5倍可能过于严格）
                        z_lower_bound = z_q1 - iqr_multiplier * z_iqr
                        z_upper_bound = z_q3 + iqr_multiplier * z_iqr
                        
                        # 过滤掉Z坐标异常的点（可能来自深度歧义导致的第二层点云）
                        # 只保留在合理范围内的点（基于IQR方法）
                        z_valid_mask = (z_coords >= z_lower_bound) & (z_coords <= z_upper_bound)
                        
                        # 如果过滤后还有足够的点，应用过滤
                        # 🔧 放宽保留比例：从50%改为30%，减少误过滤
                        min_keep_ratio = 0.3  # 至少保留30%的点（原来50%可能过于严格）
                        if np.sum(z_valid_mask) >= max(3, len(pts3d) * min_keep_ratio):
                            num_filtered = np.sum(~z_valid_mask)
                            if num_filtered > 0:
                                # 检查过滤是否会影响空间分布
                                x_coords = pts3d[:, 0]
                                y_coords = pts3d[:, 1]
                                x_range_before = x_coords.max() - x_coords.min()
                                y_range_before = y_coords.max() - y_coords.min()
                                
                                pts3d_filtered = pts3d[z_valid_mask]
                                x_coords_filtered = pts3d_filtered[:, 0]
                                y_coords_filtered = pts3d_filtered[:, 1]
                                x_range_after = x_coords_filtered.max() - x_coords_filtered.min()
                                y_range_after = y_coords_filtered.max() - y_coords_filtered.min()
                                
                                # 如果过滤后X或Y范围显著缩小（>20%），可能是误过滤
                                if x_range_after < x_range_before * 0.8 or y_range_after < y_range_before * 0.8:
                                    if self.verbose:
                                        print(f"   ⚠️ 警告: 深度一致性过滤可能导致空间分布损失")
                                        print(f"      X范围: {x_range_before:.2f} -> {x_range_after:.2f} ({100*x_range_after/x_range_before:.1f}%)")
                                        print(f"      Y范围: {y_range_before:.2f} -> {y_range_after:.2f} ({100*y_range_after/y_range_before:.1f}%)")
                                        print(f"      跳过此次过滤，保留所有点")
                                    # 跳过过滤，保留所有点
                                else:
                                    pts3d = pts3d_filtered
                                    # 更新原始mask
                                    current_valid_indices = np.where(original_valid_mask)[0]
                                    if len(current_valid_indices) > 0:
                                        z_valid_indices = current_valid_indices[z_valid_mask]
                                        original_valid_mask = np.zeros(len(img1pts), dtype=bool)
                                        original_valid_mask[z_valid_indices] = True
                                    if self.verbose:
                                        print(f"   🔧 深度一致性过滤: 移除了 {num_filtered} 个Z坐标异常的点（可能来自深度歧义）")
                                        print(f"      Z坐标范围: [{z_coords.min():.2f}, {z_coords.max():.2f}] -> "
                                              f"[{pts3d[:, 2].min():.2f}, {pts3d[:, 2].max():.2f}]")
                    
                    # 返回过滤后的点和有效点的mask（相对于原始输入img1pts/img2pts）
                    return pts3d, original_valid_mask
                except Exception as e:
                    print(f"   ❌ 错误: {name1} 和 {name2} 之间的三角化失败: {e}")
                    print(f"   诊断信息: 匹配点数={num_points}, 俯视角1={dep1}°, 方位角1={azi1}°")
                    print(f"   俯视角2={dep2}°, 方位角2={azi2}°")
                    # 调试：检查循环俯视角模式
                    disable_cyclic = getattr(self.opts, 'disable_cyclic_depression', False)
                    enable_cyclic = not disable_cyclic  # 默认启用
                    print(f"   🔍 调试: 循环俯视角模式启用状态={enable_cyclic}")
                    print(f"   🔍 调试: 图像1名称={name1}, 图像2名称={name2}")
                    print(f"   🔍 调试: self.image_angles[{name1}]={self.image_angles.get(name1, 'NOT_FOUND')}")
                    print(f"   🔍 调试: self.image_angles[{name2}]={self.image_angles.get(name2, 'NOT_FOUND')}")
                    import traceback
                    traceback.print_exc()
                    return np.zeros((0, 3)), np.array([], dtype=bool)
            else:
                raise ValueError(f"❌ 错误: 无法获取 {name1} 或 {name2} 的角度信息，SAR三角化需要角度信息")

        def _Update3DReference(ref1, ref2, img1idx, img2idx, upp_limit, low_limit=0): 

            ref1[img1idx] = np.arange(upp_limit) + low_limit
            ref2[img2idx] = np.arange(upp_limit) + low_limit

            return ref1, ref2

        R1, t1, ref1 = self.image_data[name1]
        R2, t2, ref2 = self.image_data[name2]

        # 使用统一的方向键获取匹配数据
        match_key = self._GetNormalizedMatchKey(name1, name2)
        if match_key not in self.matches_data:
            raise ValueError(f"❌ 错误: 匹配对 {name1} 和 {name2} 的数据不存在于 matches_data 中")
        
        # 检查匹配数据的方向是否与当前调用一致
        stored_name1, stored_name2 = match_key
        if stored_name1 == name1 and stored_name2 == name2:
            # 方向一致，直接使用
            _, img1pts, img2pts, img1idx, img2idx = self.matches_data[match_key]
        else:
            # 方向相反，需要交换img1和img2的数据
            _, img2pts_temp, img1pts_temp, img2idx_temp, img1idx_temp = self.matches_data[match_key]
            img1pts = img1pts_temp
            img2pts = img2pts_temp
            img1idx = img1idx_temp
            img2idx = img2idx_temp
        
        result = __TriangulateTwoViews(img1pts, img2pts, R1, t1, R2, t2)
        
        # 处理返回值：可能是(pts3d, valid_mask)或只有pts3d
        if isinstance(result, tuple):
            new_point_cloud, valid_mask = result
        else:
            new_point_cloud = result
            valid_mask = np.ones(len(new_point_cloud), dtype=bool)
        
        # 检查是否有有效的三角化点
        if new_point_cloud.shape[0] == 0:
            print(f"   ⚠️ 警告: {name1} 和 {name2} 之间的三角化没有产生有效的3D点（所有点都被过滤），跳过更新ref数组")
            return
        
        self.point_cloud = np.concatenate((self.point_cloud, new_point_cloud), axis=0)

        # 只更新有效点的索引
        # valid_mask指示哪些原始匹配点成功三角化了
        valid_img1idx = img1idx[valid_mask]
        valid_img2idx = img2idx[valid_mask]
        num_valid_points = len(new_point_cloud)
        
        ref1, ref2 = _Update3DReference(ref1, ref2, valid_img1idx, valid_img2idx, num_valid_points,
                                        self.point_cloud.shape[0]-num_valid_points)
        self.image_data[name1][-1] = ref1 
        self.image_data[name2][-1] = ref2 

    def _TriangulateNewView(self, name): 
        
        for prev_name in self.image_data.keys(): 
            if prev_name != name: 
                kp1, desc1 = self._LoadFeatures(prev_name)
                kp2, desc2 = self._LoadFeatures(name)
                
                # 检查特征点是否加载成功
                if kp1 is None or len(kp1) == 0 or kp2 is None or len(kp2) == 0:
                    continue

                prev_name_ref = self.image_data[prev_name][-1]
                name_ref = self.image_data[name][-1]
                matches = self._LoadMatches(prev_name,name)
                
                # 筛选匹配时也要检查索引是否在有效范围内
                # 改进：检查两个图像的点是否都已经被三角化，如果是，合并到已有的3D点
                valid_matches = []
                merged_count = 0
                new_point_count = 0  # 统计用于创建新3D点的匹配数
                for match in matches:
                    # 检查索引是否在有效范围内
                    if not (match.queryIdx >= 0 and match.queryIdx < len(prev_name_ref) and
                            match.queryIdx < len(kp1) and match.trainIdx >= 0 and 
                            match.trainIdx < len(kp2)):
                        continue
                    
                    prev_ref_idx = int(prev_name_ref[match.queryIdx])
                    name_ref_idx = int(name_ref[match.trainIdx]) if match.trainIdx < len(name_ref) else -1
                    
                    # 如果两个点都已经被三角化
                    if prev_ref_idx >= 0 and name_ref_idx >= 0:
                        # 如果指向同一个3D点，跳过（已经存在）
                        if prev_ref_idx == name_ref_idx:
                            continue
                        else:
                            # 指向不同的3D点，检查是否需要合并
                            # 计算两个3D点的距离
                            if prev_ref_idx < len(self.point_cloud) and name_ref_idx < len(self.point_cloud):
                                pt1 = self.point_cloud[prev_ref_idx]
                                pt2 = self.point_cloud[name_ref_idx]
                                distance = np.linalg.norm(pt1 - pt2)
                                
                                # 🔧 改进的点云合并：考虑3D距离和Z坐标差异
                                # 对于双层点云问题，需要更智能的合并策略
                                merge_threshold = getattr(self.opts, 'sfm_merge_point_threshold', 1.0)
                                
                                # 计算Z坐标差异（双层点云的主要特征）
                                z_diff = abs(pt1[2] - pt2[2])
                                
                                # 如果Z坐标差异很大（可能是双层点云），使用更宽松的合并阈值
                                # 但只合并XY平面距离很近的点
                                xy_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                                
                                # 合并条件：
                                # 1. 如果Z坐标差异小（< 5米），使用标准合并阈值
                                # 2. 如果Z坐标差异大但XY距离很近（< 0.5米），可能是双层点云，也合并
                                if z_diff < 5.0:
                                    # 标准合并：3D距离小于阈值
                                    should_merge = distance < merge_threshold
                                else:
                                    # 双层点云合并：Z坐标差异大但XY距离很近
                                    should_merge = xy_distance < (merge_threshold * 0.5)
                                
                                if should_merge:
                                    # 合并：保留prev_ref_idx，将name_ref_idx的所有引用改为prev_ref_idx
                                    # 计算合并后的位置（加权平均，可以根据track长度加权）
                                    merged_pt = (pt1 + pt2) / 2.0
                                    self.point_cloud[prev_ref_idx] = merged_pt
                                    
                                    # 更新所有指向name_ref_idx的ref
                                    for img_name in self.image_data.keys():
                                        R, t, ref = self.image_data[img_name]
                                        ref[ref == name_ref_idx] = prev_ref_idx
                                        self.image_data[img_name][-1] = ref
                                    
                                    merged_count += 1
                                    continue
                    
                    # 如果prev_name中的点已经被三角化，但name中的点没有
                    # 将name中的点添加到已有的3D点track中
                    if prev_ref_idx >= 0 and name_ref_idx < 0:
                        if match.trainIdx < len(name_ref):
                            name_ref[match.trainIdx] = prev_ref_idx
                            self.image_data[name][-1] = name_ref
                            merged_count += 1
                        continue
                    
                    # 如果name中的点已经被三角化，但prev_name中的点没有
                    # 将prev_name中的点添加到已有的3D点track中
                    if name_ref_idx >= 0 and prev_ref_idx < 0:
                        if match.queryIdx < len(prev_name_ref):
                            prev_name_ref[match.queryIdx] = name_ref_idx
                            self.image_data[prev_name][-1] = prev_name_ref
                            merged_count += 1
                        continue
                    
                    # 如果两个点都没有被三角化，进行三角化
                    if prev_ref_idx < 0 and name_ref_idx < 0:
                        valid_matches.append(match)
                        new_point_count += 1
                
                if merged_count > 0:
                    print(f"   ✅ {prev_name} 和 {name}: 合并了 {merged_count} 个已有3D点的观测")
                    self.stats_merged_points += merged_count
                if new_point_count > 0:
                    print(f"   🆕 {prev_name} 和 {name}: 将创建 {new_point_count} 个新3D点")
                    self.stats_new_points += new_point_count
                
                matches = valid_matches

                if len(matches) > 0: 
                    matches = sorted(matches, key = lambda x:x.distance)

                    img1pts, img2pts, img1idx, img2idx = self._GetAlignedMatches(kp1,desc1,kp2,
                                                                                desc2,matches)
                    
                    # OpenCV 4.x API: 使用 ransacReprojThreshold 和 confidence 替代 param1 和 param2
                    # 详细诊断：基础矩阵估计
                    num_matches = len(img1pts)
                    min_matches = getattr(self.opts, 'min_matches_for_triangulation', 1)
                    skip_fundamental = getattr(self.opts, 'sfm_skip_fundamental_matrix', False)
                    
                    if num_matches < min_matches:
                        print(f"⚠️ 警告: {prev_name} 和 {name} 之间的匹配点数量不足 ({num_matches} < {min_matches})，跳过此匹配对")
                        continue
                    
                    # 如果禁用了基础矩阵验证，直接使用所有匹配点
                    if skip_fundamental:
                        mask = np.ones(len(img1pts), dtype=bool)
                        print(f"   ℹ️ 跳过基础矩阵验证（已禁用），使用所有 {num_matches} 个匹配点")
                    else:
                        try:
                            F, mask = cv2.findFundamentalMat(img1pts, img2pts,
                                                            method=self.opts.fund_method,
                                                            ransacReprojThreshold=self.opts.outlier_thres,
                                                            confidence=self.opts.fund_prob)
                            
                            # 检查F和mask是否为None（当findFundamentalMat失败时）
                            if F is None or mask is None:
                                # SAR图像匹配困难，基础矩阵失败是正常的，只在匹配点较多时输出警告
                                if num_matches >= 8:
                                    print(f"⚠️ 警告: {prev_name} 和 {name} 之间的基础矩阵估计失败")
                                    print(f"   诊断信息: 匹配点数={num_matches}, RANSAC阈值={self.opts.outlier_thres}, 置信度={self.opts.fund_prob}")
                                    print(f"   可能原因: 匹配点分布不佳、存在大量外点、或RANSAC参数设置不当")
                                # 如果基础矩阵失败但匹配点足够，仍然使用所有匹配点
                                if num_matches >= min_matches:
                                    mask = np.ones(len(img1pts), dtype=bool)
                                    # 只在匹配点较多时输出信息
                                    if num_matches >= 8:
                                        print(f"   ℹ️ 基础矩阵失败但匹配点足够，使用所有 {num_matches} 个匹配点")
                                else:
                                    continue
                            else:
                                mask = mask.astype(bool).flatten()
                                num_inliers = np.sum(mask)
                                inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0
                                
                                min_inliers = getattr(self.opts, 'min_inliers_for_triangulation', 1)
                                if num_inliers < min_inliers:
                                    # 如果内点不足但匹配点足够，仍然使用所有匹配点
                                    if num_matches >= min_matches:
                                        mask = np.ones(len(img1pts), dtype=bool)
                                        # 只在匹配点较多时输出信息
                                        if num_matches >= 8:
                                            print(f"   ℹ️ 内点不足但匹配点足够，使用所有 {num_matches} 个匹配点")
                                    else:
                                        # 只在匹配点较多时输出警告
                                        if num_matches >= 8:
                                            print(f"⚠️ 警告: {prev_name} 和 {name} 之间的基础矩阵内点数量不足 ({num_inliers} < {min_inliers})，跳过此匹配对")
                                            print(f"   诊断信息: 总匹配点数={num_matches}, 内点数={num_inliers}, 内点比例={inlier_ratio:.2%}")
                                        continue
                        except Exception as e:
                            # SAR图像匹配困难，基础矩阵异常是正常的，只在匹配点较多时输出警告
                            if num_matches >= 8:
                                print(f"⚠️ 警告: {prev_name} 和 {name} 之间的基础矩阵估计异常: {e}")
                            # 如果异常但匹配点足够，仍然使用所有匹配点
                            if num_matches >= min_matches:
                                mask = np.ones(len(img1pts), dtype=bool)
                                # 只在匹配点较多时输出信息
                                if num_matches >= 8:
                                    print(f"   ℹ️ 基础矩阵异常但匹配点足够，使用所有 {num_matches} 个匹配点")
                            else:
                                continue
                        
                        # 保存匹配数据（使用统一的方向键和mask过滤后的匹配点）
                        match_key = self._GetNormalizedMatchKey(prev_name, name)
                        # 检查是否已存在，如果存在则跳过（避免重复存储）
                        if match_key not in self.matches_data:
                            self.matches_data[match_key] = [matches, img1pts[mask], img2pts[mask],
                                                            img1idx[mask], img2idx[mask]]
                        self._TriangulateTwoViews(prev_name, name)

                else: 
                    print('skipping {} and {}'.format(prev_name, name))
        
    def _NewViewPoseEstimation(self, name): 
        
        def _Find2D3DMatches(): 
            # 尝试使用OpenCV匹配器进行2D-3D匹配
            # 如果matcher不是OpenCV标准匹配器，则回退到直接加载匹配数据的方法
            try:
                if hasattr(cv2, self.opts.matcher):
                    matcher_temp = getattr(cv2, self.opts.matcher)()
                else:
                    # 如果不是OpenCV标准匹配器，返回None，让外部函数使用__Find2D3DMatches
                    print(f"⚠️ 警告: {self.opts.matcher} 不是OpenCV标准匹配器，将使用直接加载匹配数据的方法")
                    return None, None, 0
                
                kps, descs = [], []
                for n in self.image_names: 
                    if n in self.image_data.keys():
                        kp, desc = self._LoadFeatures(n)

                        kps.append(kp)
                        descs.append(desc)
                
                matcher_temp.add(descs)
                matcher_temp.train()

                kp, desc = self._LoadFeatures(name)

                matches_2d3d = matcher_temp.match(queryDescriptors=desc)

                #retrieving 2d and 3d points
                pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
                for m in matches_2d3d: 
                    train_img_idx, desc_idx, new_img_idx = m.imgIdx, m.trainIdx, m.queryIdx
                    point_cloud_idx = self.image_data[self.image_names[train_img_idx]][-1][desc_idx]
                    
                    #if the match corresponds to a point in 3d point cloud
                    if point_cloud_idx >= 0: 
                        new_pt = self.point_cloud[int(point_cloud_idx)]
                        pts3d = np.concatenate((pts3d, new_pt[np.newaxis]),axis=0)

                        new_pt = np.array(kp[int(new_img_idx)].pt)
                        pts2d = np.concatenate((pts2d, new_pt[np.newaxis]),axis=0)

                return pts3d, pts2d, len(kp)
            except Exception as e:
                # 如果匹配器方法失败，返回None表示失败，让外部函数使用__Find2D3DMatches
                print(f"⚠️ 警告: 使用匹配器进行2D-3D匹配失败: {e}，将使用直接加载匹配数据的方法")
                return None, None, 0
        
        def __Find2D3DMatches():
            # 直接从保存的匹配文件中加载2D-3D对应关系
            # 这个方法适用于已经完成匹配并保存的情况（如SAR_SIFT）
            pts3d, pts2d = np.zeros((0,3)), np.zeros((0,2))
            kp, desc = self._LoadFeatures(name)
            
            # 检查特征点是否加载成功
            if kp is None or len(kp) == 0:
                print(f"   ⚠️ 警告: {name} 的特征点为空，无法进行2D-3D匹配")
                return pts3d, pts2d, 0

            i = 0 
            
            while i < len(self.image_names): 
                curr_name = self.image_names[i]

                if curr_name in self.image_data.keys(): 
                    matches = self._LoadMatches(curr_name, name)
                    
                    # 检查匹配是否加载成功
                    if len(matches) == 0:
                        i += 1
                        continue

                    ref = self.image_data[curr_name][-1]
                    # 筛选出已三角化的匹配点，并检查索引是否在有效范围内
                    # 确保queryIdx在ref范围内，trainIdx在kp范围内
                    valid_matches = []
                    for m in matches:
                        # 检查queryIdx是否在ref范围内
                        if m.queryIdx < 0 or m.queryIdx >= len(ref):
                            continue
                        # 检查trainIdx是否在kp范围内
                        if m.trainIdx < 0 or m.trainIdx >= len(kp):
                            continue
                        # 检查是否有3D点
                        if ref[m.queryIdx] >= 0:
                            valid_matches.append(m)
                    
                    if len(valid_matches) == 0:
                        i += 1
                        continue
                    
                    pts3d_idx = np.array([ref[m.queryIdx] for m in valid_matches])
                    pts2d_list = [kp[m.trainIdx].pt for m in valid_matches]
                    
                    # 确保pts2d_是2维数组
                    if len(pts2d_list) > 0:
                        pts2d_ = np.array(pts2d_list)
                        # 如果pts2d_是1维的（只有一个点），需要reshape为2维
                        if pts2d_.ndim == 1:
                            pts2d_ = pts2d_.reshape(1, -1)
                        # 确保是2维数组，形状为 (N, 2)
                        if pts2d_.shape[1] != 2:
                            print(f"   ⚠️ 警告: pts2d_ 形状不正确: {pts2d_.shape}，跳过")
                            i += 1
                            continue
                    else:
                        # 如果没有有效点，创建一个空的2维数组
                        pts2d_ = np.zeros((0, 2))
                    
                    # 确保pts3d_idx对应的点存在
                    if len(pts3d_idx) > 0:
                        pts3d_idx_int = pts3d_idx.astype(int)
                        # 检查索引是否在有效范围内
                        valid_3d_mask = (pts3d_idx_int >= 0) & (pts3d_idx_int < len(self.point_cloud))
                        if np.sum(valid_3d_mask) > 0:
                            pts3d_new = self.point_cloud[pts3d_idx_int[valid_3d_mask]]
                            pts2d_new = pts2d_[valid_3d_mask]
                            
                            pts3d = np.concatenate((pts3d, pts3d_new), axis=0)
                            pts2d = np.concatenate((pts2d, pts2d_new), axis=0)

                i += 1 

            return pts3d, pts2d, len(kp)

        # 对于SAR_SIFT等自定义特征，匹配已经完成并保存，优先使用__Find2D3DMatches
        # 它从保存的匹配文件中加载数据，不需要创建匹配器
        # 如果_Find2D3DMatches失败或返回None，则使用__Find2D3DMatches
        pts3d, pts2d, ref_len = _Find2D3DMatches()
        if pts3d is None or pts2d is None or len(pts3d) == 0:
            print("使用直接加载匹配数据的方法（__Find2D3DMatches）")
            pts3d, pts2d, ref_len = __Find2D3DMatches()
        
        # 使用该图像对应的K矩阵（如果存在），否则使用全局K
        K_used = self.image_K.get(name, self.K)
        
        # 对于SAR图像，也可以尝试从角度信息直接计算姿态
        if name in self.image_angles and len(pts3d) > 0:
            # 如果有足够的2D-3D对应，使用PnP优化
            num_2d3d_correspondences = len(pts3d)
            if num_2d3d_correspondences < 4:
                print(f"⚠️ 警告: {name} 的2D-3D对应点数量不足 ({num_2d3d_correspondences} < 4)，无法使用PnP")
                print(f"   诊断信息: 需要至少4个2D-3D对应点才能进行PnP求解")
            else:
                try:
                    success, R, t, inliers = cv2.solvePnPRansac(
                        pts3d[:,np.newaxis], pts2d[:,np.newaxis], K_used, None,
                        confidence=self.opts.pnp_prob,
                        flags=getattr(cv2, self.opts.pnp_method),
                        reprojectionError=self.opts.reprojection_thres
                    )
                    if success and R is not None and t is not None:
                        num_inliers = len(inliers) if inliers is not None else 0
                        inlier_ratio = num_inliers / num_2d3d_correspondences if num_2d3d_correspondences > 0 else 0
                        
                        if num_inliers < 4:
                            print(f"⚠️ 警告: {name} 的PnP求解内点数量不足 ({num_inliers} < 4)")
                            print(f"   诊断信息: 总对应点数={num_2d3d_correspondences}, 内点数={num_inliers}, 内点比例={inlier_ratio:.2%}")
                            print(f"   可能原因: 3D点质量不佳、相机姿态估计不准确、或重投影误差阈值({self.opts.reprojection_thres})过小")
                            raise ValueError("PnP内点数量不足")
                        
                        R, _ = cv2.Rodrigues(R)
                        # R和t是从世界坐标系到该相机坐标系的变换（从PnP得到）
                        # 直接使用，因为所有相机都在同一个世界坐标系中
                        self.image_data[name] = [R, t, np.ones((ref_len,))*-1]
                        
                        if inlier_ratio < 0.5:
                            print(f"   ⚠️ 注意: {name} 的PnP内点比例较低 ({inlier_ratio:.2%})，姿态估计可能不准确")
                    else:
                        raise ValueError(f"PnP求解失败: success={success}, R={R is not None}, t={t is not None}")
                except Exception as e:
                    print(f"⚠️ 警告: {name} 的PnP求解失败: {e}")
                    print(f"   诊断信息: 2D-3D对应点数={num_2d3d_correspondences}, 方法={self.opts.pnp_method}")
                    print(f"   重投影误差阈值={self.opts.reprojection_thres}, 置信度={self.opts.pnp_prob}")
                    print(f"   尝试从角度信息计算姿态")
                # 回退到从角度信息计算姿态
                dep = self.image_angles[name]['depression']
                azi = self.image_angles[name]['azimuth']
                width, height = self.image_sizes.get(name, (self.sar_params['Na'], self.sar_params['Nr']))
                params = self.sar_params.copy()
                params['Na'] = width
                params['Nr'] = height
                # 不再设置sita0，直接使用depression_angle作为俯视角
                # 设置统一的sita0用于计算相机半径（保持相机位置不变）
                if hasattr(self, 'unified_sita0_for_radius'):
                    params['unified_sita0_for_radius'] = self.unified_sita0_for_radius
                # 使用统一的焦距和中心点（如果已设置）
                if hasattr(self, 'unified_focal_azimuth') and hasattr(self, 'unified_focal_range'):
                    params['unified_focal_azimuth'] = self.unified_focal_azimuth
                    params['unified_focal_range'] = self.unified_focal_range
                if hasattr(self, 'unified_cx') and hasattr(self, 'unified_cy'):
                    params['unified_cx'] = self.unified_cx
                    params['unified_cy'] = self.unified_cy
                # 使用统一的半径缩放因子，确保所有相机在同一球面上
                if hasattr(self, 'unified_radius_scale') and self.unified_radius_scale is not None:
                    params['radius_scale'] = self.unified_radius_scale
                    params['default_radius_scale'] = self.unified_radius_scale
                R_cam, t_cam, K_cam = compute_sar_camera_pose(dep, azi, params)
                self.image_K[name] = K_cam
                
                # R_cam和t_cam是从世界坐标系到该相机坐标系的变换
                # 直接使用，因为所有相机都在同一个世界坐标系中
                self.image_data[name] = [R_cam, t_cam, np.ones((ref_len,))*-1]
        else:
            # 标准PnP方法
            _, R, t, _ = cv2.solvePnPRansac(
                pts3d[:,np.newaxis], pts2d[:,np.newaxis], K_used, None,
                confidence=self.opts.pnp_prob,
                flags=getattr(cv2, self.opts.pnp_method),
                reprojectionError=self.opts.reprojection_thres
            )
            R, _ = cv2.Rodrigues(R)
            self.image_data[name] = [R, t, np.ones((ref_len,))*-1]

    def ToPly(self, filename):
        
        def _GetColors(): 
            colors = np.zeros_like(self.point_cloud)
            
            # 获取默认颜色
            default_r = getattr(self.opts, 'sfm_default_point_color_r', 50)
            default_g = getattr(self.opts, 'sfm_default_point_color_g', 50)
            default_b = getattr(self.opts, 'sfm_default_point_color_b', 200)
            default_color = np.array([default_r, default_g, default_b])
            
            # 初始化所有点为默认颜色
            colors[:] = default_color
            
            for k in self.image_data.keys(): 
                R_cam, t_cam, ref = self.image_data[k]
                
                # 获取已三角化的点的索引
                valid_ref_mask = ref >= 0
                if np.sum(valid_ref_mask) == 0:
                    continue  # 如果没有已三角化的点，跳过
                
                ref_indices = ref[valid_ref_mask].astype(int)
                
                # 获取对应的3D点
                points3d = self.point_cloud[ref_indices]
                
                # 获取K矩阵
                K_used = self.image_K.get(k, self.K)
                
                # 重投影3D点到图像坐标
                def _ComputeReprojections(X, R, t, K):
                    """将3D点投影到图像坐标"""
                    # 将3D点转换到相机坐标系
                    X_cam = R @ X.T + t
                    # 检查深度是否为正
                    valid_depth_mask = X_cam[2, :] > 1e-6
                    # 投影到图像平面
                    outh = K @ X_cam
                    # 转换为齐次坐标
                    outh[0, :] /= (outh[2, :] + 1e-8)
                    outh[1, :] /= (outh[2, :] + 1e-8)
                    # 提取图像坐标
                    reproj_pts = outh[:2, :].T
                    return reproj_pts, valid_depth_mask
                
                # 计算重投影坐标
                reproj_pts, valid_depth_mask = _ComputeReprojections(points3d, R_cam, t_cam, K_used)
                
                # 尝试读取图像（支持多种格式）
                image = None
                image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
                for ext in image_extensions:
                    image_path = os.path.join(self.images_dir, f'{k}.{ext}')
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        if image is not None:
                            break
                
                if image is None:
                    # 如果无法读取图像，使用默认颜色（已经在上面设置了）
                    continue
                
                # 处理图像格式：如果是BGR，转换为RGB；如果是灰度图，转换为RGB
                if len(image.shape) == 2:
                    # 灰度图，转换为RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif len(image.shape) == 3:
                    # BGR图像，转换为RGB
                    image = image[:,:,::-1]
                else:
                    continue
                
                # 确保图像尺寸正确
                height, width = image.shape[:2]
                
                # 检查重投影点是否在图像范围内
                valid_pts_mask = valid_depth_mask & \
                                (reproj_pts[:, 0] >= 0) & (reproj_pts[:, 0] < width) & \
                                (reproj_pts[:, 1] >= 0) & (reproj_pts[:, 1] < height)
                
                if np.sum(valid_pts_mask) == 0:
                    # 如果所有点都在图像范围外，使用默认颜色（已经在上面设置了）
                    continue
                
                # 只处理有效的点
                valid_reproj_pts = reproj_pts[valid_pts_mask].astype(int)
                valid_ref_indices_subset = ref_indices[valid_pts_mask]
                
                # 提取颜色（注意：OpenCV图像是 [height, width, channels]，所以索引顺序是 [y, x]）
                try:
                    # 使用双线性插值提取颜色（更准确）
                    for i, (ref_idx, pt) in enumerate(zip(valid_ref_indices_subset, valid_reproj_pts)):
                        x, y = pt[0], pt[1]
                        # 确保坐标在有效范围内
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        colors[ref_idx] = image[y, x]
                except Exception as e:
                    # 如果提取颜色失败，使用默认颜色（已经在上面设置了）
                    continue
            
            return colors

        colors = _GetColors()
        pts2ply(self.point_cloud, colors, filename)
    
    @staticmethod
    def _finite_json_float(x, fallback=0.0, lo=-1e15, hi=1e15):
        """JSON 可序列化的有限浮点，避免 inf/nan/天文数字"""
        try:
            v = float(x)
            if not math.isfinite(v):
                return float(fallback)
            return float(min(hi, max(lo, v)))
        except (TypeError, ValueError):
            return float(fallback)
    
    def _scene_geometry_stats_from_poses(self):
        """从 image_data 估计中位斜距/|Y|（仅全自由度 SO-RCG 写 JSON 时使用）。"""
        from sar.sfm_scale_params import pose_geometry_stats

        return pose_geometry_stats(self.image_data)
    
    def _CollectCameraData(self):
        """
        收集所有相机数据，返回可序列化的字典
        用于保存到sar_scale_params.json
        """
        cameras_dict = {}
        
        for img_name in self.image_data.keys():
            R_cam, t_cam, ref = self.image_data[img_name]
            
            # 确保旋转矩阵是有效的（正交且行列式为1）
            U, S, Vt = np.linalg.svd(R_cam)
            R_cam_corrected = U @ Vt
            
            # 确保行列式为1（如果是-1，则取反最后一列）
            det = np.linalg.det(R_cam_corrected)
            if det < 0:
                R_cam_corrected[:, 2] = -R_cam_corrected[:, 2]
            
            # 将旋转矩阵转换为四元数（w, x, y, z格式，COLMAP格式）
            rotation = Rot.from_matrix(R_cam_corrected)
            quat = rotation.as_quat()  # [x, y, z, w]
            
            # 使用修正后的旋转矩阵
            R_cam = R_cam_corrected
            
            # 计算相机在世界坐标系中的位置
            camera_position_world = -R_cam.T @ t_cam
            
            # 构建相机数据字典
            camera_info = {
                'qw': float(quat[3]),
                'qx': float(quat[0]),
                'qy': float(quat[1]),
                'qz': float(quat[2]),
                'tx': float(camera_position_world[0, 0]),
                'ty': float(camera_position_world[1, 0]),
                'tz': float(camera_position_world[2, 0]),
                't_camera_frame': t_cam.tolist(),
                'R': R_cam.tolist(),
                't': t_cam.tolist()
            }
            
            # 从相机位置计算俯视角和方位角
            P_radar = camera_position_world.flatten()
            sphere_radius = np.linalg.norm(P_radar)
            
            if sphere_radius > 1e-6:
                xz_dist = np.sqrt(P_radar[0]**2 + P_radar[2]**2)
                sin_dep = np.clip(xz_dist / sphere_radius, -1.0, 1.0)
                depression_angle_rad = np.arcsin(sin_dep)
                depression_angle_deg = np.rad2deg(depression_angle_rad)
                azimuth_angle_rad = np.arctan2(P_radar[2], P_radar[0])
                azimuth_angle_deg = np.rad2deg(azimuth_angle_rad)
            else:
                # 如果半径太小，使用image_angles中的值作为备选
                if img_name in self.image_angles:
                    depression_angle_deg = self.image_angles[img_name]['depression']
                    azimuth_angle_deg = self.image_angles[img_name]['azimuth']
                else:
                    depression_angle_deg = 0.0
                    azimuth_angle_deg = 0.0
            
            camera_info['depression_angle'] = float(depression_angle_deg)
            camera_info['azimuth_angle'] = float(azimuth_angle_deg)
            
            # 添加K矩阵
            if img_name in self.image_K:
                camera_info['K'] = self.image_K[img_name].tolist()
            
            # 添加图像尺寸
            if img_name in self.image_sizes:
                width, height = self.image_sizes[img_name]
                camera_info['image_size'] = [int(width), int(height)]
            
            # 添加原始角度信息（如果存在）
            if img_name in self.image_angles:
                camera_info['original_depression'] = float(self.image_angles[img_name]['depression'])
                camera_info['original_azimuth'] = float(self.image_angles[img_name]['azimuth'])
            
            cameras_dict[img_name] = camera_info
        
        return cameras_dict
    
    def _SaveCameraPoses(self):
        """保存相机位姿到文件"""
        import json
        
        poses_file = os.path.join(self.out_cloud_dir, 'camera_poses.json')
        poses_dict = {}
        
        for img_name in self.image_data.keys():
            R_cam, t_cam, ref = self.image_data[img_name]
            
            # 确保旋转矩阵是有效的（正交且行列式为1）
            # 使用SVD分解来修正旋转矩阵
            U, S, Vt = np.linalg.svd(R_cam)
            R_cam_corrected = U @ Vt
            
            # 确保行列式为1（如果是-1，则取反最后一列）
            det = np.linalg.det(R_cam_corrected)
            if det < 0:
                R_cam_corrected[:, 2] = -R_cam_corrected[:, 2]
            
            # 将旋转矩阵转换为四元数（w, x, y, z格式，COLMAP格式）
            rotation = Rot.from_matrix(R_cam_corrected)
            quat = rotation.as_quat()  # [x, y, z, w]
            
            # 使用修正后的旋转矩阵
            R_cam = R_cam_corrected
            
            # 计算相机在世界坐标系中的位置
            # 在OpenCV中，t是从世界到相机的平移，所以相机位置C = -R^T @ t
            camera_position_world = -R_cam.T @ t_cam
            
            # COLMAP格式：四元数为 [w, x, y, z]
            poses_dict[img_name] = {
                'qw': float(quat[3]),
                'qx': float(quat[0]),
                'qy': float(quat[1]),
                'qz': float(quat[2]),
                'tx': float(camera_position_world[0, 0]),  # 相机在世界坐标系中的位置
                'ty': float(camera_position_world[1, 0]),
                'tz': float(camera_position_world[2, 0]),
                't_camera_frame': t_cam.tolist(),  # 在相机坐标系中的平移向量
                'R': R_cam.tolist(),
                't': t_cam.tolist()  # 保留原始的t（在相机坐标系中）
            }
            
            # 🔧 修复：从优化后的相机位置重新计算俯视角和方位角，而不是使用image_angles中的值
            # 因为优化过程中image_angles可能被更新（使用错误的公式），所以应该从实际相机位置计算
            P_radar = camera_position_world.flatten()  # 相机在世界坐标系中的位置
            sphere_radius = np.linalg.norm(P_radar)
            
            if sphere_radius > 1e-6:
                # 计算俯视角（COLMAP坐标系：XZ平面是水平面，Y是高度）
                # 在球面分布模式下，相机位置为：
                #   x = R * sin(depression) * cos(azimuth)
                #   y = -R * cos(depression)  # 高度在Y轴负方向
                #   z = R * sin(depression) * sin(azimuth)
                # 因此，俯视角 depression = arcsin(xz_dist / R)
                xz_dist = np.sqrt(P_radar[0]**2 + P_radar[2]**2)  # XZ平面距离（水平距离）
                sin_dep = np.clip(xz_dist / sphere_radius, -1.0, 1.0)
                depression_angle_rad = np.arcsin(sin_dep)
                depression_angle_deg = np.rad2deg(depression_angle_rad)
                
                # 计算方位角（COLMAP坐标系：XZ平面是水平面）
                azimuth_angle_rad = np.arctan2(P_radar[2], P_radar[0])  # atan2(Z, X)
                azimuth_angle_deg = np.rad2deg(azimuth_angle_rad)
            else:
                # 如果半径太小，使用image_angles中的值作为备选
                if img_name in self.image_angles:
                    depression_angle_deg = self.image_angles[img_name]['depression']
                    azimuth_angle_deg = self.image_angles[img_name]['azimuth']
                else:
                    depression_angle_deg = 0.0
                    azimuth_angle_deg = 0.0
            
            poses_dict[img_name]['depression_angle'] = float(depression_angle_deg)
            poses_dict[img_name]['azimuth_angle'] = float(azimuth_angle_deg)
            
            # 如果有K矩阵，也保存
            if img_name in self.image_K:
                poses_dict[img_name]['K'] = self.image_K[img_name].tolist()
        
        with open(poses_file, 'w') as f:
            json.dump(poses_dict, f, indent=2)
        
        print(f"✅ 相机位姿已保存到: {poses_file}")
        
        # 同时保存COLMAP格式的images.txt和cameras.txt
        self._SaveColmapFormat()
    
    def _BuildCompleteTracks(self):
        """
        从所有匹配数据构建完整的track信息
        这个方法会读取所有匹配文件，找出每个3D点被哪些图像观测到
        改进：合并位置相似的3D点，构建完整的track
        """
        print("🔧 构建完整的track信息...")
        
        # 建立图像名称到图像ID的映射
        img_name_to_id = {}
        image_id = 1
        for img_name in sorted(self.image_data.keys()):
            img_name_to_id[img_name] = image_id
            image_id += 1
        
        # 建立初始的point_tracks（基于ref数组）
        point_tracks = {}
        for img_name in self.image_data.keys():
            R_cam, t_cam, ref = self.image_data[img_name]
            img_id = img_name_to_id.get(img_name, 0)
            
            for point2d_idx, point3d_idx in enumerate(ref):
                if point3d_idx >= 0:
                    if point3d_idx not in point_tracks:
                        point_tracks[point3d_idx] = []
                    point_tracks[point3d_idx].append((img_id, point2d_idx))
        
        # 遍历所有匹配对，扩展track信息
        # 改进：多次迭代，确保所有连接都被发现
        extended_count = 0
        max_iterations = 5  # 最多迭代5次，确保所有连接都被发现
        
        for iteration in range(max_iterations):
            iteration_extended = 0
            for match_key in self.matches_data.keys():
                img1, img2 = match_key  # match_key已经是统一方向的元组
                if img1 not in self.image_data or img2 not in self.image_data:
                    continue
                
                matches, img1pts, img2pts, img1idx, img2idx = self.matches_data[match_key]
                
                # 获取两个图像的ref数组
                ref1 = self.image_data[img1][-1]
                ref2 = self.image_data[img2][-1]
                img1_id = img_name_to_id.get(img1, 0)
                img2_id = img_name_to_id.get(img2, 0)
                
                # 对于每个匹配点
                for i in range(len(img1idx)):
                    idx1 = img1idx[i]
                    idx2 = img2idx[i]
                    
                    if idx1 >= len(ref1) or idx2 >= len(ref2):
                        continue
                    
                    point3d_idx1 = int(ref1[idx1])
                    point3d_idx2 = int(ref2[idx2])
                    
                    # 如果两个点都指向3D点
                    if point3d_idx1 >= 0 and point3d_idx2 >= 0:
                        # 如果指向不同的3D点，检查是否需要合并
                        if point3d_idx1 != point3d_idx2:
                            # 检查两个3D点是否相似（距离很近）
                            if (point3d_idx1 < len(self.point_cloud) and 
                                point3d_idx2 < len(self.point_cloud)):
                                pt1 = self.point_cloud[point3d_idx1]
                                pt2 = self.point_cloud[point3d_idx2]
                                distance = np.linalg.norm(pt1 - pt2)
                                
                                # 🔧 改进的点云合并：考虑3D距离和Z坐标差异
                                # 对于双层点云问题，需要更智能的合并策略
                                merge_threshold = getattr(self.opts, 'sfm_merge_point_threshold', 1.0)
                                
                                # 计算Z坐标差异（双层点云的主要特征）
                                z_diff = abs(pt1[2] - pt2[2])
                                
                                # 如果Z坐标差异很大（可能是双层点云），使用更宽松的合并阈值
                                # 但只合并XY平面距离很近的点
                                xy_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                                
                                # 合并条件：
                                # 1. 如果Z坐标差异小（< 5米），使用标准合并阈值
                                # 2. 如果Z坐标差异大但XY距离很近（< 0.5米），可能是双层点云，也合并
                                if z_diff < 5.0:
                                    # 标准合并：3D距离小于阈值
                                    should_merge = distance < merge_threshold
                                else:
                                    # 双层点云合并：Z坐标差异大但XY距离很近
                                    should_merge = xy_distance < (merge_threshold * 0.5)
                                
                                if should_merge:
                                    # 合并点：保留point3d_idx1，删除point3d_idx2
                                    # 计算合并后的位置（加权平均，根据track长度）
                                    track1_len = len(point_tracks.get(point3d_idx1, []))
                                    track2_len = len(point_tracks.get(point3d_idx2, []))
                                    total_len = track1_len + track2_len
                                    
                                    if total_len > 0:
                                        weight1 = track1_len / total_len
                                        weight2 = track2_len / total_len
                                        merged_pt = weight1 * pt1 + weight2 * pt2
                                        self.point_cloud[point3d_idx1] = merged_pt
                                    
                                    # 更新所有指向point3d_idx2的ref
                                    for img_name in self.image_data.keys():
                                        R_cam, t_cam, ref = self.image_data[img_name]
                                        ref[ref == point3d_idx2] = point3d_idx1
                                        self.image_data[img_name][-1] = ref
                                    
                                    # 合并track信息
                                    if point3d_idx2 in point_tracks:
                                        point_tracks[point3d_idx1].extend(point_tracks[point3d_idx2])
                                        del point_tracks[point3d_idx2]
                                    
                                    extended_count += 1
                                    iteration_extended += 1
                    
                    # 如果只有一个点指向3D点，将另一个点也指向同一个3D点
                    elif point3d_idx1 >= 0 and point3d_idx2 < 0:
                        # 将img2中的点也指向同一个3D点
                        ref2[idx2] = point3d_idx1
                        self.image_data[img2][-1] = ref2
                        
                        # 添加到track
                        if point3d_idx1 not in point_tracks:
                            point_tracks[point3d_idx1] = []
                        point_tracks[point3d_idx1].append((img2_id, idx2))
                        extended_count += 1
                        iteration_extended += 1
                        
                    elif point3d_idx2 >= 0 and point3d_idx1 < 0:
                        # 将img1中的点也指向同一个3D点
                        ref1[idx1] = point3d_idx2
                        self.image_data[img1][-1] = ref1
                        
                        # 添加到track
                        if point3d_idx2 not in point_tracks:
                            point_tracks[point3d_idx2] = []
                        point_tracks[point3d_idx2].append((img1_id, idx1))
                        extended_count += 1
                        iteration_extended += 1
            
            # 如果这一轮没有扩展任何track，说明已经收敛
            if iteration_extended == 0:
                break
            
            if iteration > 0:
                print(f"   第{iteration+1}轮迭代: 扩展了 {iteration_extended} 个track连接")
        
        if extended_count > 0:
            print(f"   ✅ 扩展了 {extended_count} 个track观测")
        
        # 计算平均track长度
        if len(point_tracks) > 0:
            mean_track_length = sum(len(track) for track in point_tracks.values()) / len(point_tracks)
            print(f"   📊 平均track长度: {mean_track_length:.2f}")
        
        return point_tracks, img_name_to_id
    
    def _RunSarBundleAdjustment(self):
        """
        运行SAR专用的Bundle Adjustment
        使用SAR的重投影误差（斜距误差）而不是像素误差
        """
        try:
            from sar_bundle_adjustment import sar_bundle_adjustment, prepare_sar_ba_data
            
            print("\n" + "="*60)
            print("🔧 运行SAR专用Bundle Adjustment...")
            print("="*60)
            
            # 准备数据
            reconstruction_data = prepare_sar_ba_data(self)
            
            # 获取SAR BA参数
            max_iterations = getattr(self.opts, 'sfm_sar_ba_max_iterations', 50)
            ftol = getattr(self.opts, 'sfm_sar_ba_ftol', 1e-6)
            xtol = getattr(self.opts, 'sfm_sar_ba_xtol', 1e-8)
            
            # 运行SAR BA
            optimized_data, error_history = sar_bundle_adjustment(
                reconstruction_data,
                self.sar_params,
                max_iterations=max_iterations,
                ftol=ftol,
                xtol=xtol,
                verbose=True
            )
            
            # 更新重建结果
            self.point_cloud = optimized_data['point_cloud']
            self.image_data = optimized_data['image_data']
            if error_history.get('optimized_camera_height') is not None:
                self.sar_params['camera_height'] = float(error_history['optimized_camera_height'])
            
            print("="*60)
            print("✅ SAR Bundle Adjustment完成")
            print(f"   - 初始误差: {error_history['initial_error']:.4f} 米")
            print(f"   - 最终误差: {error_history['final_error']:.4f} 米")
            if error_history['initial_error'] > 0:
                improvement = (error_history['initial_error'] - error_history['final_error']) / error_history['initial_error'] * 100
                print(f"   - 误差改善: {improvement:.2f}%")
            print(f"   - 迭代次数: {error_history['iterations']}")
            print("="*60 + "\n")
            
        except ImportError as e:
            print(f"⚠️ 警告: 无法导入sar_bundle_adjustment模块: {e}")
            print("   SAR Bundle Adjustment将被跳过")
        except Exception as e:
            print(f"⚠️ 警告: SAR Bundle Adjustment失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _FilterPointsByReprojectionError(self):
        """
        基于重投影误差过滤点云，改善点云形状
        过滤掉重投影误差过大的点，保留形状准确的点
        """
        reprojection_thres = getattr(self.opts, 'sfm_reprojection_thres', 12.0)
        # 对于SAR图像，由于观测角度限制，track长度要求可以放宽
        # 如果所有点都没有被观测到，则使用更宽松的条件
        min_track_length = 1  # 至少被1个视角观测到（放宽要求）
        
        # 检查是否使用了SO-RCG方法
        use_sorc = SORCG_AVAILABLE and hasattr(self, 'image_angles')
        reprojection_thres_meters = reprojection_thres  # 默认值（像素单位）
        
        if use_sorc:
            # 对于SAR的RD模型，重投影误差单位是米，转换为像素后阈值需要调整
            # 默认阈值12像素对应约12*range_pixel_size米
            # 但考虑到SAR的重投影误差可能较大，使用更宽松的阈值
            sar_params = self.sar_params if hasattr(self, 'sar_params') else {}
            fs = sar_params.get('fs', 1.1 * 240e6)
            C = sar_params.get('C', 3e8)
            range_pixel_size = C / (2 * fs)  # 每个像素对应的斜距（米）
            # 将像素阈值转换为米，然后乘以一个安全系数（SAR误差通常较大）
            # 注意：SAR的重投影误差通常较大（几百米），所以阈值需要相应放宽
            # 根据实际误差情况，需要更大的阈值（放宽100-200倍）
            reprojection_thres_meters = reprojection_thres * range_pixel_size * 200  # 放宽200倍（SAR误差通常较大，需要更宽松的阈值）
            print(f"   ℹ️ 使用SAR RD模型过滤，阈值: {reprojection_thres_meters:.2f}米 (约{reprojection_thres*200:.1f}像素等效)")
            print(f"   ℹ️ 注意: SAR的重投影误差通常较大（几百到上千米），阈值已相应放宽")
        
        print("\n" + "="*60)
        print("🔧 基于重投影误差过滤点云...")
        print("="*60)
        
        if len(self.point_cloud) == 0:
            print("⚠️ 点云为空，跳过过滤")
            return
        
        # 计算每个点的重投影误差
        point_errors = {}
        point_track_lengths = {}
        
        # 统计有多少图像有有效的ref
        images_with_valid_ref = 0
        total_valid_refs = 0
        
        for img_name in self.image_data.keys():
            R_cam, t_cam, ref = self.image_data[img_name]
            K_used = self.image_K.get(img_name, self.K)
            
            # 获取该图像观测到的所有3D点
            valid_ref_mask = ref >= 0
            num_valid_refs = np.sum(valid_ref_mask)
            if num_valid_refs > 0:
                images_with_valid_ref += 1
                total_valid_refs += num_valid_refs
            
            if np.sum(valid_ref_mask) == 0:
                continue
            
            ref_indices = ref[valid_ref_mask].astype(int)
            point_indices = ref_indices[ref_indices < len(self.point_cloud)]
            
            if len(point_indices) == 0:
                continue
            
            # 获取对应的3D点
            points3d = self.point_cloud[point_indices]
            
            # 获取对应的2D关键点
            kp, _ = self._LoadFeatures(img_name)
            if kp is None:
                continue
            
            valid_kp_indices = np.where(valid_ref_mask)[0]
            
            # 检查是否使用SO-RCG方法（通过检查是否有SAR角度信息）
            use_sar_rd_model = False
            if img_name in self.image_angles and SORCG_AVAILABLE:
                # 如果使用了SO-RCG，使用SAR的RD模型计算重投影误差
                use_sar_rd_model = True
            
            if use_sar_rd_model:
                # 使用SAR的RD模型计算重投影误差
                from sar_sorc_sfm import SORCG_SAR_SfM
                # 创建临时求解器实例用于计算重投影误差
                if not hasattr(self, '_sorc_solver_for_filtering'):
                    self._sorc_solver_for_filtering = SORCG_SAR_SfM(
                        sar_params=self.sar_params if hasattr(self, 'sar_params') else None,
                        verbose=False
                    )
                
                dep = self.image_angles[img_name].get('depression', 31.57)
                
                # 计算每个点的重投影误差（使用RD模型）
                for i, pt_idx in enumerate(point_indices):
                    if i >= len(valid_kp_indices):
                        continue
                    kp_idx = valid_kp_indices[i]
                    if kp_idx >= len(kp):
                        continue
                    
                    point3d = points3d[i] if points3d.ndim == 2 else points3d
                    gt_pt = np.array(kp[kp_idx].pt)
                    
                    # 使用SAR的RD模型计算重投影误差
                    try:
                        error_range, error_azimuth, error_total = self._sorc_solver_for_filtering.compute_rd_reprojection_error(
                            point3d, R_cam, t_cam, gt_pt, dep
                        )
                        # RD模型的误差是米，直接使用米作为单位
                        # 后续会与阈值（也是米）比较
                        # 注意：优化迭代中使用的是 |error_range| 和 |error_azimuth| 的平均值
                        # 过滤阶段也应该使用相同的计算方式，保持一致性
                        # 使用斜距误差和方位误差的绝对值之和（与优化迭代中的统计方式一致）
                        error_meters = abs(error_range) + abs(error_azimuth)  # 使用绝对值之和，与优化迭代统计一致
                    except Exception as e:
                        # 如果计算失败，使用一个较大的默认误差
                        if not hasattr(self, '_warned_rd_error_calc'):
                            print(f"   ⚠️ 警告: 部分点的RD重投影误差计算失败: {e}")
                            self._warned_rd_error_calc = True
                        error_meters = 10000.0  # 10公里，很大的误差
                    
                    if pt_idx not in point_errors:
                        point_errors[pt_idx] = []
                        point_track_lengths[pt_idx] = 0
                    
                    point_errors[pt_idx].append(error_meters)
                    point_track_lengths[pt_idx] += 1
            else:
                # 使用透视投影模型计算重投影误差（原始方法）
                X_cam = R_cam @ points3d.T + t_cam
                valid_depth_mask = X_cam[2, :] > 1e-6
                
                if np.sum(valid_depth_mask) == 0:
                    continue
                
                # 投影到图像平面
                outh = K_used @ X_cam[:, valid_depth_mask]
                outh[0, :] /= (outh[2, :] + 1e-8)
                outh[1, :] /= (outh[2, :] + 1e-8)
                reproj_pts = outh[:2, :].T
                
                valid_kp_indices_filtered = valid_kp_indices[valid_depth_mask]
                if len(valid_kp_indices_filtered) != len(reproj_pts):
                    continue
                
                # 计算重投影误差
                for i, (pt_idx, kp_idx) in enumerate(zip(point_indices[valid_depth_mask], valid_kp_indices_filtered)):
                    if kp_idx >= len(kp):
                        continue
                
                gt_pt = np.array(kp[kp_idx].pt)
                reproj_pt = reproj_pts[i]
                error = np.linalg.norm(gt_pt - reproj_pt)
                
                if pt_idx not in point_errors:
                    point_errors[pt_idx] = []
                    point_track_lengths[pt_idx] = 0
                
                point_errors[pt_idx].append(error)
                point_track_lengths[pt_idx] += 1
        
        # 检查是否有任何点被观测到
        if len(point_errors) == 0:
            print(f"⚠️ 警告: 没有找到任何被观测到的3D点，跳过过滤")
            print(f"   诊断信息:")
            print(f"   - 总图像数: {len(self.image_data)}")
            print(f"   - 有有效ref的图像数: {images_with_valid_ref}")
            print(f"   - 总有效ref数: {total_valid_refs}")
            print(f"   - 点云大小: {len(self.point_cloud)}")
            print(f"   可能原因: ref数组未正确构建，或点云与图像观测未正确关联")
            print("="*60 + "\n")
            return
        
        # 过滤点云
        points_to_keep = []
        filtered_count = 0
        
        # 确定使用的阈值（根据是否使用SO-RCG）
        if use_sorc:
            # 对于SAR RD模型，使用基于误差分布的动态阈值
            all_errors = []
            for errors in point_errors.values():
                all_errors.extend(errors)
            
            if len(all_errors) > 0:
                all_errors = np.array(all_errors)
                median_error = np.median(all_errors)
                mean_error = np.mean(all_errors)
                
                # 诊断：显示误差分布
                print(f"\n  📊 重投影误差分布统计:")
                print(f"    - 误差数量: {len(all_errors)}")
                print(f"    - 平均误差: {mean_error:.2f}米")
                print(f"    - 中位数误差: {median_error:.2f}米")
                print(f"    - 最小误差: {np.min(all_errors):.2f}米")
                print(f"    - 最大误差: {np.max(all_errors):.2f}米")
                print(f"    - 误差标准差: {np.std(all_errors):.2f}米")
                print(f"    ℹ️ 注意: 过滤阶段的误差 = |斜距误差| + |方位误差|")
                print(f"    ℹ️ 优化迭代中的误差 = |斜距误差| 和 |方位误差| 分别统计")
                print(f"    ℹ️ 如果过滤阶段误差远大于优化迭代，可能是:")
                print(f"       1. 使用了不同的位姿（过滤阶段应使用优化后的位姿）")
                print(f"       2. 误差计算方式不一致（已修复：使用绝对值之和）")
                
                # 使用动态阈值：基于中位数误差
                # 如果中位数误差很大（>10000米），说明优化未收敛，使用更宽松的阈值
                # 否则使用固定阈值
                if median_error > 10000:
                    # 优化未收敛，使用中位数的5倍作为阈值（更宽松）
                    error_threshold = median_error * 5.0
                    print(f"    ⚠️ 检测到中位数误差较大 ({median_error:.2f}米)，使用动态阈值: {error_threshold:.2f}米 (中位数的5倍)")
                elif median_error > 2000:
                    # 中位数误差较大，使用中位数的3倍作为阈值
                    error_threshold = median_error * 3.0
                    print(f"    ⚠️ 检测到中位数误差较大 ({median_error:.2f}米)，使用动态阈值: {error_threshold:.2f}米 (中位数的3倍)")
                elif median_error > 1000:
                    # 中位数误差中等，使用中位数的2倍作为阈值
                    error_threshold = median_error * 2.0
                    print(f"    ⚠️ 检测到中位数误差中等 ({median_error:.2f}米)，使用动态阈值: {error_threshold:.2f}米 (中位数的2倍)")
                else:
                    # 中位数误差合理，使用固定阈值
                    error_threshold = reprojection_thres_meters
                    print(f"    - 使用固定阈值: {error_threshold:.2f}米")
                
                print(f"    - 预计保留点数: {np.sum(all_errors <= error_threshold)}/{len(all_errors)}")
                
                # 如果预计保留点数太少（<10%），进一步放宽阈值
                predicted_keep = np.sum(all_errors <= error_threshold)
                if predicted_keep < len(all_errors) * 0.1:
                    error_threshold = median_error * 10.0  # 使用中位数的10倍
                    print(f"    ⚠️ 预计保留点数过少，放宽阈值至: {error_threshold:.2f}米 (中位数的10倍)")
                    print(f"    - 更新后预计保留点数: {np.sum(all_errors <= error_threshold)}/{len(all_errors)}")
            else:
                error_threshold = reprojection_thres_meters
        else:
            error_threshold = reprojection_thres
        
        for pt_idx in range(len(self.point_cloud)):
            if pt_idx not in point_errors:
                # 没有被任何图像观测到的点，删除
                filtered_count += 1
                continue
            
            if point_track_lengths[pt_idx] < min_track_length:
                # track长度不足的点，删除
                filtered_count += 1
                continue
            
            # 计算平均重投影误差和中位数误差
            avg_error = np.mean(point_errors[pt_idx])
            median_error = np.median(point_errors[pt_idx])
            max_error = np.max(point_errors[pt_idx])
            
            # 对于SAR数据，使用更宽松的过滤条件
            # 如果中位数误差在合理范围内，保留该点（即使有少数异常大的误差）
            # 只有当中位数误差也超过阈值时才删除
            if use_sorc:
                # SAR RD模型：主要看中位数误差，允许少数异常值
                # 如果中位数误差超过阈值，删除该点
                if median_error > error_threshold:
                    filtered_count += 1
                    continue
            else:
                # 透视投影模型：使用原来的逻辑
                if avg_error > error_threshold or max_error > error_threshold * 2:
                    filtered_count += 1
                    continue
            
            points_to_keep.append(pt_idx)
        
        # 更新点云
        if len(points_to_keep) < len(self.point_cloud):
            old_point_cloud = self.point_cloud.copy()
            self.point_cloud = self.point_cloud[points_to_keep]
            
            # 更新ref数组：重新映射点索引
            point_idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(points_to_keep)}
            
            for img_name in self.image_data.keys():
                R_cam, t_cam, ref = self.image_data[img_name]
                new_ref = ref.copy()
                
                for i in range(len(ref)):
                    if ref[i] >= 0:
                        if ref[i] in point_idx_mapping:
                            new_ref[i] = point_idx_mapping[ref[i]]
                        else:
                            new_ref[i] = -1  # 点被过滤了
                
                self.image_data[img_name] = [R_cam, t_cam, new_ref]
            
            print(f"✅ 点云过滤完成:")
            print(f"   - 原始点数: {len(old_point_cloud)}")
            print(f"   - 过滤后点数: {len(self.point_cloud)}")
            print(f"   - 过滤点数: {filtered_count}")
            print(f"   - 保留比例: {len(self.point_cloud)/len(old_point_cloud)*100:.1f}%")
        else:
            print(f"✅ 所有点都满足重投影误差要求，无需过滤")
        
        print("="*60 + "\n")
    
    def _SaveColmapFormat(self):
        """保存COLMAP格式的相机、图像和点云文件（完全符合COLMAP标准格式）
        保存到 sparse/0/ 目录，包括文本格式、二进制格式和PLY格式
        """
        import struct
        
        # 定义二进制写入辅助函数
        def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
            """pack and write to a binary file."""
            if isinstance(data, (list, tuple)):
                bytes = struct.pack(endian_character + format_char_sequence, *data)
            else:
                bytes = struct.pack(endian_character + format_char_sequence, data)
            fid.write(bytes)
        
        # 保存cameras.txt和cameras.bin
        cameras_file_txt = os.path.join(self.out_sparse_dir, 'cameras.txt')
        cameras_file_bin = os.path.join(self.out_sparse_dir, 'cameras.bin')
        with open(cameras_file_txt, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(f"# Number of cameras: {len(self.image_K)}\n")
            
            # 🔧 修复：检查所有图像的K矩阵是否一致
            # 如果所有图像的K矩阵相同（使用统一焦距），只保存一个相机
            # 如果不同，需要为每张图像创建不同的相机ID
            unique_K = {}
            img_to_camera_id = {}
            camera_id = 1
            # 保存为实例变量，供后续二进制格式保存使用
            self._unique_K = unique_K
            self._img_to_camera_id = img_to_camera_id
            
            for img_name in sorted(self.image_data.keys()):
                if img_name in self.image_K:
                    K = self.image_K[img_name]
                    width, height = self.image_sizes.get(img_name, (self.sar_params['Na'], self.sar_params['Nr']))
                    
                    # 将K矩阵转换为可哈希的元组（用于比较）
                    K_key = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)
                    
                    if K_key not in unique_K:
                        unique_K[K_key] = camera_id
                        # 保存这个唯一的相机参数
                        fx = K[0, 0]
                        fy = K[1, 1]
                        cx = K[0, 2]
                        cy = K[1, 2]
                        f.write(f"{camera_id} PINHOLE {int(width)} {int(height)} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
                        camera_id += 1
                    
                    img_to_camera_id[img_name] = unique_K[K_key]
                else:
                    # 如果没有K矩阵，使用第一张图像的K矩阵
                    if len(self.image_K) > 0:
                        first_img = list(self.image_K.keys())[0]
                        K = self.image_K[first_img]
                        width, height = self.image_sizes.get(first_img, (self.sar_params['Na'], self.sar_params['Nr']))
                        K_key = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)
                        if K_key not in unique_K:
                            unique_K[K_key] = camera_id
                            fx = K[0, 0]
                            fy = K[1, 1]
                            cx = K[0, 2]
                            cy = K[1, 2]
                            f.write(f"{camera_id} PINHOLE {int(width)} {int(height)} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
                            camera_id += 1
                        img_to_camera_id[img_name] = unique_K[K_key]
                    else:
                        # 回退：使用默认值
                        img_to_camera_id[img_name] = 1
                        if 1 not in unique_K:
                            width, height = self.sar_params['Na'], self.sar_params['Nr']
                            fx = width * 1.5
                            fy = height * 1.5
                            cx = width / 2.0
                            cy = height / 2.0
                            f.write(f"1 PINHOLE {int(width)} {int(height)} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
                            unique_K[(fx, fy, cx, cy, width, height)] = 1
        
        # 建立图像名称到图像ID的映射
        img_name_to_id = {}
        image_id = 1
        for img_name in sorted(self.image_data.keys()):
            img_name_to_id[img_name] = image_id
            image_id += 1
        
        # 建立3D点的track信息（每个3D点被哪些图像观测到，以及对应的2D点索引）
        # point_tracks[point3d_idx] = [(image_id, point2d_idx), ...]
        # 改进：从所有匹配数据构建完整的track
        build_complete_tracks = getattr(self.opts, 'sfm_build_complete_tracks', True)
        if build_complete_tracks:
            point_tracks, img_name_to_id = self._BuildCompleteTracks()
        else:
            # 使用简单方法（基于ref数组）
            point_tracks = {}
            for img_name in self.image_data.keys():
                R_cam, t_cam, ref = self.image_data[img_name]
                img_id = img_name_to_id.get(img_name, 0)
                
                for point2d_idx, point3d_idx in enumerate(ref):
                    if point3d_idx >= 0:
                        if point3d_idx not in point_tracks:
                            point_tracks[point3d_idx] = []
                        point_tracks[point3d_idx].append((img_id, point2d_idx))
        
        point_colors = {}
        
        # 🔧 诊断：检查相机位姿的多样性
        print(f"\n{'='*70}")
        print(f"📐 相机位姿诊断（检查所有相机是否都有不同的位姿）")
        print(f"{'='*70}")
        camera_positions = []
        camera_rotations = []
        for img_name in sorted(self.image_data.keys()):
            R_cam, t_cam, ref = self.image_data[img_name]
            # 计算相机位置：C = -R^T * t
            camera_pos = -R_cam.T @ t_cam.flatten()
            camera_positions.append(camera_pos)
            camera_rotations.append(R_cam.copy())
            if len(camera_positions) <= 5:  # 只打印前5个
                print(f"  - {img_name}: 位置=({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})")
        
        # 检查位置差异
        if len(camera_positions) > 1:
            camera_positions = np.array(camera_positions)
            pos_std = np.std(camera_positions, axis=0)
            pos_range = np.ptp(camera_positions, axis=0)  # peak-to-peak (max - min)
            print(f"\n  位置统计:")
            print(f"    - 标准差: X={pos_std[0]:.2f}, Y={pos_std[1]:.2f}, Z={pos_std[2]:.2f}")
            print(f"    - 范围: X={pos_range[0]:.2f}, Y={pos_range[1]:.2f}, Z={pos_range[2]:.2f}")
            
            # 检查是否所有相机都在同一位置
            if np.all(pos_std < 0.01):
                print(f"    ⚠️ 警告: 所有相机位置几乎相同（标准差<0.01），这会导致高斯泼溅无法正确整合所有图像！")
            elif np.all(pos_range < 1.0):
                print(f"    ⚠️ 警告: 相机位置范围很小（<1.0米），可能导致视角旋转问题！")
            else:
                print(f"    ✅ 相机位置有足够的多样性")
        
        print(f"{'='*70}\n")
        
        # 保存images.txt并建立track信息到sparse/0/
        images_file_txt = os.path.join(self.out_sparse_dir, 'images.txt')
        with open(images_file_txt, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(self.image_data)}\n")
            
            image_id = 1
            for img_name in sorted(self.image_data.keys()):
                R_cam, t_cam, ref = self.image_data[img_name]
                
                # 确保旋转矩阵是有效的（正交且行列式为1）
                U, S, Vt = np.linalg.svd(R_cam)
                R_cam_corrected = U @ Vt
                det = np.linalg.det(R_cam_corrected)
                if det < 0:
                    R_cam_corrected[:, 2] = -R_cam_corrected[:, 2]
                R_cam = R_cam_corrected
                
                # 转换为四元数
                rotation = Rot.from_matrix(R_cam)
                quat = rotation.as_quat()  # [x, y, z, w]
                
                # COLMAP格式说明：
                # - qvec (QW, QX, QY, QZ): 从世界到相机的旋转（四元数，Hamilton约定）
                # - tvec (TX, TY, TZ): 从世界到相机的平移向量（不是相机位置！）
                # - 相机位置在世界坐标系中是 -R^T * tvec
                # 我们的R_cam和t_cam已经是世界到相机的变换，所以直接使用t_cam作为tvec
                tvec = t_cam.flatten()  # 从世界到相机的平移向量
                
                # COLMAP格式：四元数为 [w, x, y, z]，平移为从世界到相机的平移向量
                # 查找图像文件的实际扩展名（img_name可能没有扩展名）
                image_filename = img_name
                if not os.path.splitext(img_name)[1]:  # 如果没有扩展名
                    # 尝试查找实际文件（支持多种扩展名）
                    image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
                    found = False
                    for ext in image_extensions:
                        potential_path = os.path.join(self.images_dir, f'{img_name}.{ext}')
                        if os.path.exists(potential_path):
                            image_filename = f'{img_name}.{ext}'
                            found = True
                            break
                    if not found:
                        # 如果找不到，使用opts.ext中的第一个扩展名（向后兼容）
                        if hasattr(self.opts, 'ext') and len(self.opts.ext) > 0:
                            image_filename = f'{img_name}.{self.opts.ext[0]}'
                        else:
                            image_filename = f'{img_name}.jpg'  # 默认使用jpg
                
                # 🔧 使用正确的相机ID（如果K矩阵不同，每张图像可能有不同的相机ID）
                camera_id_for_image = img_to_camera_id.get(img_name, 1)
                f.write(f"{image_id} {quat[3]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} "
                       f"{tvec[0]:.6f} {tvec[1]:.6f} {tvec[2]:.6f} {camera_id_for_image} {image_filename}\n")
                
                # 保存2D-3D对应关系并建立track信息（COLMAP格式：包含所有特征点）
                try:
                    kp, desc = self._LoadFeatures(img_name)
                    num_points2d = len(kp)
                    
                    # 写入所有2D点信息（COLMAP格式要求包含所有特征点）
                    for point2d_idx in range(num_points2d):
                        kp_obj = kp[point2d_idx]
                        ref_idx = ref[point2d_idx]
                        
                        # point3D_id: 如果有3D点则从1开始，否则为-1
                        if ref_idx >= 0:
                            point3d_id = int(ref_idx) + 1
                            
                            # 建立track信息（point3d_idx从0开始，但在points3D.txt中从1开始）
                            if ref_idx not in point_tracks:
                                point_tracks[ref_idx] = []
                            point_tracks[ref_idx].append((image_id, point2d_idx))
                            
                            # 提取颜色（如果还没有）
                            if ref_idx not in point_colors:
                                try:
                                    # 尝试从图像中提取颜色
                                    image = None
                                    image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
                                    for ext in image_extensions:
                                        image_path = os.path.join(self.images_dir, f'{img_name}.{ext}')
                                        if os.path.exists(image_path):
                                            image = cv2.imread(image_path)
                                            if image is not None:
                                                break
                                    
                                    if image is not None:
                                        # 处理图像格式
                                        if len(image.shape) == 2:
                                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                                        elif len(image.shape) == 3:
                                            image = image[:,:,::-1]  # BGR to RGB
                                        
                                        # 提取颜色
                                        pt = kp_obj.pt
                                        if 0 <= pt[1] < image.shape[0] and 0 <= pt[0] < image.shape[1]:
                                            color = image[int(pt[1]), int(pt[0])]
                                            point_colors[ref_idx] = (int(color[0]), int(color[1]), int(color[2]))
                                        else:
                                            # 使用配置的默认颜色（深色，在白色背景上更易观察）
                                            default_r = getattr(self.opts, 'sfm_default_point_color_r', 50)
                                            default_g = getattr(self.opts, 'sfm_default_point_color_g', 50)
                                            default_b = getattr(self.opts, 'sfm_default_point_color_b', 200)
                                            point_colors[ref_idx] = (default_r, default_g, default_b)
                                    else:
                                        # 使用配置的默认颜色
                                        default_r = getattr(self.opts, 'sfm_default_point_color_r', 50)
                                        default_g = getattr(self.opts, 'sfm_default_point_color_g', 50)
                                        default_b = getattr(self.opts, 'sfm_default_point_color_b', 200)
                                        point_colors[ref_idx] = (default_r, default_g, default_b)
                                except:
                                    # 使用配置的默认颜色
                                    default_r = getattr(self.opts, 'sfm_default_point_color_r', 50)
                                    default_g = getattr(self.opts, 'sfm_default_point_color_g', 50)
                                    default_b = getattr(self.opts, 'sfm_default_point_color_b', 200)
                                    point_colors[ref_idx] = (default_r, default_g, default_b)
                        else:
                            point3d_id = -1
                        
                        # 写入images.txt中的2D点信息
                        f.write(f"{kp_obj.pt[0]:.6f} {kp_obj.pt[1]:.6f} {point3d_id} ")
                    
                    f.write("\n")
                except Exception as e:
                    # 如果无法加载特征，写入空行
                    print(f"   ⚠️ 警告: 处理图像 {img_name} 时出错: {e}")
                    f.write("\n")
                
                image_id += 1
        
        # 保存points3D.txt到sparse/0/
        points3d_file_txt = os.path.join(self.out_sparse_dir, 'points3D.txt')
        with open(points3d_file_txt, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            num_points = len(self.point_cloud)
            if num_points > 0:
                # 计算平均track长度
                mean_track_length = sum(len(track) for track in point_tracks.values()) / num_points if num_points > 0 else 0
                f.write(f"# Number of points: {num_points}, mean track length: {mean_track_length:.2f}\n")
                
                # 为每个3D点写入信息
                for point3d_idx in range(num_points):
                    # POINT3D_ID从1开始
                    point3d_id = point3d_idx + 1
                    x, y, z = self.point_cloud[point3d_idx]
                    
                    # 获取颜色（使用配置的默认颜色）
                    default_r = getattr(self.opts, 'sfm_default_point_color_r', 50)
                    default_g = getattr(self.opts, 'sfm_default_point_color_g', 50)
                    default_b = getattr(self.opts, 'sfm_default_point_color_b', 200)
                    r, g, b = point_colors.get(point3d_idx, (default_r, default_g, default_b))
                    
                    # 重投影误差（暂时设为0，可以后续计算）
                    error = 0.0
                    
                    # 写入基本信息
                    f.write(f"{point3d_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f}")
                    
                    # 写入track信息
                    if point3d_idx in point_tracks:
                        for image_id, point2d_idx in point_tracks[point3d_idx]:
                            f.write(f" {image_id} {point2d_idx}")
                    
                    f.write("\n")
            else:
                f.write("# Number of points: 0, mean track length: 0.00\n")
        
        # ========== 保存二进制格式 ==========
        import struct
        
        def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
            """pack and write to a binary file."""
            if isinstance(data, (list, tuple)):
                bytes = struct.pack(endian_character + format_char_sequence, *data)
            else:
                bytes = struct.pack(endian_character + format_char_sequence, data)
            fid.write(bytes)
        
        # 保存cameras.bin（COLMAP二进制格式）
        cameras_file_bin = os.path.join(self.out_sparse_dir, 'cameras.bin')
        with open(cameras_file_bin, 'wb') as f:
            # 🔧 修复：使用文本格式保存时创建的unique_K和img_to_camera_id
            if hasattr(self, '_unique_K') and hasattr(self, '_img_to_camera_id'):
                unique_K = self._unique_K
                img_to_camera_id = self._img_to_camera_id
            else:
                # 如果没有，重新计算（向后兼容）
                unique_K = {}
                img_to_camera_id = {}
                camera_id = 1
                for img_name in sorted(self.image_data.keys()):
                    if img_name in self.image_K:
                        K = self.image_K[img_name]
                        width, height = self.image_sizes.get(img_name, (self.sar_params['Na'], self.sar_params['Nr']))
                        K_key = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)
                        if K_key not in unique_K:
                            unique_K[K_key] = camera_id
                            camera_id += 1
                        img_to_camera_id[img_name] = unique_K[K_key]
            
            num_cameras = len(unique_K) if len(unique_K) > 0 else (1 if len(self.image_K) > 0 else 0)
            write_next_bytes(f, num_cameras, "Q")  # 相机数量
            
            if num_cameras > 0:
                # 按相机ID顺序保存
                for camera_id in sorted(unique_K.values()) if len(unique_K) > 0 else [1]:
                    # 找到使用这个相机ID的第一张图像
                    img_for_camera = None
                    for img_name, cam_id in img_to_camera_id.items():
                        if cam_id == camera_id and img_name in self.image_K:
                            img_for_camera = img_name
                            break
                    
                    if img_for_camera is None and len(self.image_K) > 0:
                        img_for_camera = list(self.image_K.keys())[0]
                    
                    if img_for_camera and img_for_camera in self.image_K:
                        K = self.image_K[img_for_camera]
                        width, height = self.image_sizes.get(img_for_camera, (self.sar_params['Na'], self.sar_params['Nr']))
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        
                        # COLMAP格式：camera_id, model_id, width, height
                        # model_id=1表示PINHOLE模型
                        write_next_bytes(f, [camera_id, 1, int(width), int(height)], "iiQQ")
                        # PINHOLE模型参数: fx, fy, cx, cy（4个double）
                        write_next_bytes(f, [float(fx), float(fy), float(cx), float(cy)], "dddd")
        
        # 保存images.bin
        images_file_bin = os.path.join(self.out_sparse_dir, 'images.bin')
        # 🔧 获取相机ID映射（从文本格式保存时创建）
        if hasattr(self, '_img_to_camera_id'):
            img_to_camera_id = self._img_to_camera_id
        else:
            # 如果没有，创建默认映射（所有图像使用相机ID=1）
            img_to_camera_id = {img_name: 1 for img_name in self.image_data.keys()}
        
        with open(images_file_bin, 'wb') as f:
            write_next_bytes(f, len(self.image_data), "Q")  # 图像数量
            image_id = 1
            for img_name in sorted(self.image_data.keys()):
                R_cam, t_cam, ref = self.image_data[img_name]
                
                # 确保旋转矩阵是有效的（正交且行列式为1）
                U, S, Vt = np.linalg.svd(R_cam)
                R_cam_corrected = U @ Vt
                det = np.linalg.det(R_cam_corrected)
                if det < 0:
                    R_cam_corrected[:, 2] = -R_cam_corrected[:, 2]
                R_cam = R_cam_corrected
                
                rotation = Rot.from_matrix(R_cam)
                quat = rotation.as_quat()  # [x, y, z, w]
                
                # COLMAP格式：tvec是从世界到相机的平移向量（不是相机位置）
                tvec = t_cam.flatten()  # 从世界到相机的平移向量
                
                # 写入图像ID
                write_next_bytes(f, image_id, "i")
                # 写入四元数 [w, x, y, z]（COLMAP格式）
                write_next_bytes(f, [quat[3], quat[0], quat[1], quat[2]], "dddd")
                # 写入平移向量（从世界到相机的平移）
                write_next_bytes(f, [tvec[0], tvec[1], tvec[2]], "ddd")
                # 🔧 写入正确的相机ID（如果K矩阵不同，每张图像可能有不同的相机ID）
                camera_id_for_image = img_to_camera_id.get(img_name, 1)
                write_next_bytes(f, camera_id_for_image, "i")
                # 查找图像文件的实际扩展名（img_name可能没有扩展名）
                image_filename = img_name
                if not os.path.splitext(img_name)[1]:  # 如果没有扩展名
                    # 尝试查找实际文件（支持多种扩展名）
                    image_extensions = ['jpg', 'jpeg', 'png', 'tif', 'tiff']
                    found = False
                    for ext in image_extensions:
                        potential_path = os.path.join(self.images_dir, f'{img_name}.{ext}')
                        if os.path.exists(potential_path):
                            image_filename = f'{img_name}.{ext}'
                            found = True
                            break
                    if not found:
                        # 如果找不到，使用opts.ext中的第一个扩展名（向后兼容）
                        if hasattr(self.opts, 'ext') and len(self.opts.ext) > 0:
                            image_filename = f'{img_name}.{self.opts.ext[0]}'
                        else:
                            image_filename = f'{img_name}.jpg'  # 默认使用jpg
                # 写入图像名称（以null结尾）
                # COLMAP格式：逐个字符写入UTF-8编码的字符串，以null结尾
                # 注意：与read_write_model.py保持一致，逐个字符编码
                for char in image_filename:
                    write_next_bytes(f, char.encode("utf-8"), "c")
                write_next_bytes(f, b"\x00", "c")
                
                # 写入2D点信息（COLMAP格式：包含所有特征点，没有3D点的用-1）
                try:
                    kp, desc = self._LoadFeatures(img_name)
                    num_points2d = len(kp)
                    write_next_bytes(f, num_points2d, "Q")
                    
                    # 创建ref索引映射（从特征点索引到3D点索引）
                    for point2d_idx in range(num_points2d):
                        kp_obj = kp[point2d_idx]
                        ref_idx = ref[point2d_idx]
                        
                        # point3D_id: 如果有3D点则从1开始，否则为-1
                        if ref_idx >= 0:
                            point3d_id = int(ref_idx) + 1
                        else:
                            point3d_id = -1
                        
                        # x, y, point3D_id
                        write_next_bytes(f, [kp_obj.pt[0], kp_obj.pt[1], point3d_id], "ddq")
                except Exception as e:
                    # 如果无法加载特征，写入0个点
                    write_next_bytes(f, 0, "Q")
                
                image_id += 1
        
        # 保存points3D.bin
        points3d_file_bin = os.path.join(self.out_sparse_dir, 'points3D.bin')
        with open(points3d_file_bin, 'wb') as f:
            num_points = len(self.point_cloud)
            write_next_bytes(f, num_points, "Q")
            for point3d_idx in range(num_points):
                point3d_id = point3d_idx + 1
                x, y, z = self.point_cloud[point3d_idx]
                r, g, b = point_colors.get(point3d_idx, (255, 255, 255))
                error = 0.0
                
                # 写入点ID
                write_next_bytes(f, point3d_id, "Q")
                # 写入坐标
                write_next_bytes(f, [x, y, z], "ddd")
                # 写入颜色
                write_next_bytes(f, [r, g, b], "BBB")
                # 写入误差
                write_next_bytes(f, error, "d")
                # 写入track信息
                if point3d_idx in point_tracks:
                    track_length = len(point_tracks[point3d_idx])
                    write_next_bytes(f, track_length, "Q")
                    for image_id, point2d_idx in point_tracks[point3d_idx]:
                        write_next_bytes(f, [image_id, point2d_idx], "ii")
                else:
                    write_next_bytes(f, 0, "Q")
        
        # ========== 保存PLY格式（与3DGS训练代码兼容）==========
        points3d_file_ply = os.path.join(self.out_sparse_dir, 'points3D.ply')
        num_points = len(self.point_cloud)
        
        # 检查点云是否为空
        if num_points == 0:
            print(f"   ⚠️ 警告: 点云为空，无法保存PLY文件")
            return
        
        # 准备颜色数组
        colors_array = np.zeros((num_points, 3), dtype=np.uint8)
        for point3d_idx in range(num_points):
            r, g, b = point_colors.get(point3d_idx, (255, 255, 255))
            colors_array[point3d_idx] = [r, g, b]
        
        # 使用与storePly相同的格式保存PLY文件（包含法线属性）
        # 这样可以直接被3DGS训练代码的fetchPly函数读取
        try:
            from plyfile import PlyData, PlyElement
            
            # 定义数据类型（与storePly一致）
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            
            # 确保点云是Nx3格式（每行一个点，3列分别是x, y, z）
            # self.point_cloud 应该是 Nx3 格式
            if len(self.point_cloud.shape) == 2 and self.point_cloud.shape[1] == 3:
                xyz = self.point_cloud.astype(np.float32)
            elif len(self.point_cloud.shape) == 2 and self.point_cloud.shape[0] == 3:
                # 如果是3xN格式，转置为Nx3
                xyz = self.point_cloud.T.astype(np.float32)
            else:
                raise ValueError(f"点云形状不正确: {self.point_cloud.shape}，期望 (N, 3) 或 (3, N)")
            
            # 创建法线数组（全为0，因为SfM重建的点云没有法线信息）
            normals = np.zeros_like(xyz, dtype=np.float32)
            
            # 创建结构化数组
            elements = np.empty(num_points, dtype=dtype)
            attributes = np.concatenate((xyz, normals, colors_array), axis=1)
            elements[:] = list(map(tuple, attributes))
            
            # 创建PlyElement并写入文件
            vertex_element = PlyElement.describe(elements, 'vertex')
            ply_data = PlyData([vertex_element])
            # 确保路径使用正确的格式（规范化路径，避免混合正斜杠和反斜杠）
            points3d_file_ply_normalized = os.path.normpath(os.path.abspath(points3d_file_ply))
            # 确保目录存在
            dir_path = os.path.dirname(points3d_file_ply_normalized)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            # 使用字符串路径（plyfile库需要字符串路径）
            ply_data.write(str(points3d_file_ply_normalized))
            
            print(f"   ✅ PLY文件已保存（包含法线属性，兼容3DGS训练）: {points3d_file_ply_normalized}")
        except ImportError:
            # 如果plyfile不可用，回退到简单的pts2ply格式
            print(f"   ⚠️ 警告: plyfile模块不可用，使用简单PLY格式（可能不兼容3DGS训练）")
            from sfm_utils import pts2ply
            pts2ply(self.point_cloud, colors_array, points3d_file_ply)
        
        print(f"✅ COLMAP格式文件已保存到 {self.out_sparse_dir}:")
        print(f"   文本格式:")
        print(f"   - {cameras_file_txt}")
        print(f"   - {images_file_txt}")
        print(f"   - {points3d_file_txt}")
        print(f"   二进制格式:")
        print(f"   - {cameras_file_bin}")
        print(f"   - {images_file_bin}")
        print(f"   - {points3d_file_bin}")
        print(f"   PLY格式:")
        print(f"   - {points3d_file_ply}")
    
    def _PrintPointCloudStatistics(self):
        """打印点云统计信息（包括空间分布诊断）"""
        if len(self.point_cloud) == 0:
            print("⚠️ 警告: 点云为空")
            return
        
        print(f"\n📊 点云统计信息:")
        print(f"   - 总点数: {len(self.point_cloud)}")
        
        # 打印匹配点用途统计
        total_matches_used = self.stats_new_points + self.stats_merged_points
        if total_matches_used > 0:
            print(f"\n📈 匹配点用途统计:")
            print(f"   - 用于创建新3D点: {self.stats_new_points} 个匹配 ({self.stats_new_points/total_matches_used*100:.1f}%)")
            print(f"   - 用于合并/扩展已有track: {self.stats_merged_points} 个匹配 ({self.stats_merged_points/total_matches_used*100:.1f}%)")
            print(f"   - 匹配点利用率: {total_matches_used} 个匹配被使用")
            if self.stats_merged_points > self.stats_new_points:
                print(f"   ℹ️ 说明: 更多匹配点用于扩展已有track，说明track连接良好（这是好的！）")
            elif self.stats_new_points > self.stats_merged_points:
                print(f"   ℹ️ 说明: 更多匹配点用于创建新点，说明图像重叠较少")
        
        # 计算点云的分布
        x_coords = self.point_cloud[:, 0]
        y_coords = self.point_cloud[:, 1]
        z_coords = self.point_cloud[:, 2]
        
        print(f"   - X坐标范围: [{x_coords.min():.2f}, {x_coords.max():.2f}], 标准差: {x_coords.std():.2f}")
        print(f"   - Y坐标范围: [{y_coords.min():.2f}, {y_coords.max():.2f}], 标准差: {y_coords.std():.2f}")
        print(f"   - Z坐标范围: [{z_coords.min():.2f}, {z_coords.max():.2f}], 标准差: {z_coords.std():.2f}")
        
        # 检查点云是否在一条线上
        # 计算主成分分析
        centered_points = self.point_cloud - self.point_cloud.mean(axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]  # 降序排列
        
        # 计算方差比例
        total_var = eigenvals.sum()
        var_ratios = eigenvals / total_var if total_var > 1e-8 else eigenvals
        
        print(f"   - 主成分方差比例: [{var_ratios[0]:.3f}, {var_ratios[1]:.3f}, {var_ratios[2]:.3f}]")
        
        if var_ratios[0] > 0.95:
            print(f"   ⚠️ 警告: 点云可能退化为一维（一条线），第一主成分方差占比 {var_ratios[0]:.3%}")
            print(f"   可能原因: 相机姿态计算错误、三角化失败、或所有点都在同一条线上")
        elif var_ratios[0] + var_ratios[1] > 0.95:
            print(f"   ⚠️ 警告: 点云可能退化为一维（一个平面），前两个主成分方差占比 {var_ratios[0] + var_ratios[1]:.3%}")
            print(f"   可能原因: 相机姿态计算错误、三角化失败、或所有点都在同一个平面上")
        
        # 检查点云中心
        center = self.point_cloud.mean(axis=0)
        print(f"   - 点云中心: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        # 🔧 新增：检查点云的空间分布，诊断缺失区域
        print(f"\n🔍 点云空间分布诊断（检查是否有缺失区域）:")
        # 将点云分成几个区域，检查每个区域的点数
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        z_min, z_max = z_coords.min(), z_coords.max()
        
        # 在X、Y、Z三个方向上各分成3段，检查每段的点数
        num_segments = 3
        x_segments = np.linspace(x_min, x_max, num_segments + 1)
        y_segments = np.linspace(y_min, y_max, num_segments + 1)
        z_segments = np.linspace(z_min, z_max, num_segments + 1)
        
        # X方向分布
        x_distribution = []
        for i in range(num_segments):
            mask = (x_coords >= x_segments[i]) & (x_coords < x_segments[i+1])
            if i == num_segments - 1:  # 最后一个段包含最大值
                mask = (x_coords >= x_segments[i]) & (x_coords <= x_segments[i+1])
            count = np.sum(mask)
            x_distribution.append(count)
            print(f"   - X方向 [{x_segments[i]:.2f}, {x_segments[i+1]:.2f}]: {count} 个点 ({100*count/len(self.point_cloud):.1f}%)")
        
        # 检查X方向是否有明显的缺失区域
        x_distribution = np.array(x_distribution)
        x_mean = x_distribution.mean()
        x_std = x_distribution.std()
        if x_std > x_mean * 0.5:  # 如果标准差大于均值的50%，说明分布不均匀
            min_segment_idx = np.argmin(x_distribution)
            max_segment_idx = np.argmax(x_distribution)
            ratio = x_distribution[max_segment_idx] / (x_distribution[min_segment_idx] + 1e-6)
            if ratio > 3.0:  # 如果最大段点数是最小段的3倍以上
                print(f"   ⚠️ 警告: X方向分布不均匀！")
                print(f"      最多点数的段: [{x_segments[max_segment_idx]:.2f}, {x_segments[max_segment_idx+1]:.2f}] ({x_distribution[max_segment_idx]} 个点)")
                print(f"      最少点数的段: [{x_segments[min_segment_idx]:.2f}, {x_segments[min_segment_idx+1]:.2f}] ({x_distribution[min_segment_idx]} 个点)")
                print(f"      比例: {ratio:.1f}:1")
                print(f"      可能原因: 某些视角的匹配点不足，导致该区域没有被三角化")
        
        # Y方向分布
        y_distribution = []
        for i in range(num_segments):
            mask = (y_coords >= y_segments[i]) & (y_coords < y_segments[i+1])
            if i == num_segments - 1:
                mask = (y_coords >= y_segments[i]) & (y_coords <= y_segments[i+1])
            count = np.sum(mask)
            y_distribution.append(count)
            print(f"   - Y方向 [{y_segments[i]:.2f}, {y_segments[i+1]:.2f}]: {count} 个点 ({100*count/len(self.point_cloud):.1f}%)")
        
        # Z方向分布
        z_distribution = []
        for i in range(num_segments):
            mask = (z_coords >= z_segments[i]) & (z_coords < z_segments[i+1])
            if i == num_segments - 1:
                mask = (z_coords >= z_segments[i]) & (z_coords <= z_segments[i+1])
            count = np.sum(mask)
            z_distribution.append(count)
            print(f"   - Z方向 [{z_segments[i]:.2f}, {z_segments[i+1]:.2f}]: {count} 个点 ({100*count/len(self.point_cloud):.1f}%)")
        
        # 检查相机位置
        print(f"\n📷 相机位姿信息（前5个）:")
        camera_positions = []
        for i, img_name in enumerate(sorted(self.image_data.keys())):
            if i >= 5:
                break
            R_cam, t_cam, ref = self.image_data[img_name]
            # 计算相机在世界坐标系中的位置
            camera_position_world = -R_cam.T @ t_cam.flatten()
            camera_positions.append(camera_position_world)
            print(f"   - {img_name}: 世界位置=({camera_position_world[0]:.2f}, {camera_position_world[1]:.2f}, {camera_position_world[2]:.2f})")
        
        if len(camera_positions) > 0:
            camera_positions = np.array(camera_positions)
            print(f"   - 相机位置范围: X=[{camera_positions[:, 0].min():.2f}, {camera_positions[:, 0].max():.2f}], "
                  f"Y=[{camera_positions[:, 1].min():.2f}, {camera_positions[:, 1].max():.2f}], "
                  f"Z=[{camera_positions[:, 2].min():.2f}, {camera_positions[:, 2].max():.2f}]")
        
        print(f"\n💾 点云文件保存在: {self.out_cloud_dir}")
        print(f"💾 相机位姿文件保存在: {os.path.join(self.out_cloud_dir, 'camera_poses.json')}")
    
    def _ComputeReprojectionError(self, name):
        
        def _ComputeReprojections(X,R,t,K): 
            outh = K.dot(R.dot(X.T) + t )
            out = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]
            return out 

        R, t, ref = self.image_data[name]
        # 使用该图像对应的K矩阵（如果存在），否则使用全局K
        K_used = self.image_K.get(name, self.K)
        
        # 获取已三角化的3D点索引
        valid_ref_indices = ref[ref >= 0]
        if len(valid_ref_indices) == 0:
            print(f"⚠️ 警告: {name} 没有已三角化的3D点，无法计算重投影误差")
            return 0.0
        
        point_cloud_indices = valid_ref_indices.astype(int)
        reproj_pts = _ComputeReprojections(self.point_cloud[point_cloud_indices], R, t, K_used)

        kp, desc = self._LoadFeatures(name)
        img_pts = np.array([kp_.pt for i, kp_ in enumerate(kp) if ref[i] >= 0])
        
        # 计算重投影误差
        errors = np.sqrt(np.sum((img_pts - reproj_pts)**2, axis=-1))
        err = np.mean(errors)
        
        # 详细诊断信息
        num_points = len(errors)
        if num_points > 0:
            max_err = np.max(errors)
            min_err = np.min(errors)
            median_err = np.median(errors)
            std_err = np.std(errors)
            p95_err = np.percentile(errors, 95)
            p99_err = np.percentile(errors, 99)
            
            # 检查是否有异常大的误差
            large_error_threshold = err * 3  # 超过平均误差3倍的点
            num_large_errors = np.sum(errors > large_error_threshold)
            
            if err > 50 or num_large_errors > num_points * 0.1:  # 如果平均误差>50或超过10%的点有异常大误差
                print(f"   📊 重投影误差详细统计 ({name}):")
                print(f"      - 3D点数量: {num_points}")
                print(f"      - 平均误差: {err:.2f} 像素")
                print(f"      - 中位数误差: {median_err:.2f} 像素")
                print(f"      - 标准差: {std_err:.2f} 像素")
                print(f"      - 最小误差: {min_err:.2f} 像素")
                print(f"      - 最大误差: {max_err:.2f} 像素")
                print(f"      - 95%分位数: {p95_err:.2f} 像素")
                print(f"      - 99%分位数: {p99_err:.2f} 像素")
                print(f"      - 异常大误差点数 (> {large_error_threshold:.1f}): {num_large_errors} ({num_large_errors/num_points*100:.1f}%)")
                
                # 检查可能的原因
                if max_err > 1000:
                    print(f"      ⚠️ 可能原因: 存在极端异常点，建议检查匹配质量或三角化结果")
                elif std_err > err * 2:
                    print(f"      ⚠️ 可能原因: 误差分布不均匀，可能存在外点或错误的相机姿态")
                elif num_large_errors > num_points * 0.2:
                    print(f"      ⚠️ 可能原因: 超过20%的点有异常大误差，建议检查相机内参K矩阵是否正确")

        if self.opts.plot_error: 
            fig,ax = plt.subplots()
            image = cv2.imread(os.path.join(self.images_dir, name+'.jpg'))[:,:,::-1]
            ax = DrawCorrespondences(image, img_pts, reproj_pts, ax)
            
            ax.set_title('reprojection error = {}'.format(err))

            fig.savefig(os.path.join(self.out_err_dir, '{}.png'.format(name)))
            plt.close(fig)
            
        return err
        
    def Run(self):
        """
        运行SAR SfM重建
        
        根据配置选择使用SO-RCG方法或矩阵分解方法
        """
        # 检查是否使用SO-RCG方法
        use_sorc = getattr(self.opts, 'sfm_use_sorc', False)
        
        # 调试信息
        print(f"\n🔍 SfM方法选择:")
        print(f"   - use_sorc参数: {use_sorc}")
        print(f"   - SORCG_AVAILABLE: {SORCG_AVAILABLE}")
        print(f"   - MATRIX_FACTORIZATION_AVAILABLE: {MATRIX_FACTORIZATION_AVAILABLE}")
        
        if use_sorc and SORCG_AVAILABLE:
            print(f"   ✅ 选择SO-RCG方法")
            return self.RunSORCG()
        else:
            if use_sorc and not SORCG_AVAILABLE:
                print(f"   ⚠️ SO-RCG方法不可用，回退到矩阵分解方法")
            elif not use_sorc:
                print(f"   ℹ️ 使用矩阵分解方法（use_sorc=False）")
            else:
                print(f"   ℹ️ 使用矩阵分解方法")
            # 默认使用矩阵分解方法
        return self.RunMatrixFactorization()
    
    def RunSORCG(self):
        """
        使用SO-RCG方法进行SAR SfM重建
        
        这是基于论文"Keypoint-Based SAR Structure From Motion via Riemannian Optimization"的方法
        使用SO-RCG算法在SO(3)群上优化旋转矩阵
        """
        if not SORCG_AVAILABLE:
            raise ImportError("❌ 错误: sar_sorc_sfm模块未找到，无法使用SO-RCG方法")
        
        print("\n" + "="*60)
        print("🔧 使用SO-RCG方法进行SAR SfM重建")
        print("="*60)
        
        # 准备数据
        keypoints_dict = {}
        matches_dict = {}
        
        # 加载所有关键点
        for img_name in self.image_names:
            kp, desc = self._LoadFeatures(img_name)
            if kp is not None and len(kp) > 0:
                keypoints_dict[img_name] = kp
        
        # 加载所有匹配
        for i, name1 in enumerate(self.image_names):
            for name2 in self.image_names[i+1:]:
                matches = self._LoadMatches(name1, name2)
                if len(matches) > 0:
                    matches_dict[(name1, name2)] = matches
        
        if len(keypoints_dict) == 0:
            raise ValueError("❌ 错误: 没有找到任何关键点")
        
        if len(matches_dict) == 0:
            raise ValueError("❌ 错误: 没有找到任何匹配")

        if len(self.point_cloud) > 0:
            try:
                from sar.scene_scale_closure import scale_point_cloud_to_mstar_target_extent

                self.point_cloud = scale_point_cloud_to_mstar_target_extent(
                    self.point_cloud, self.sar_params, args=self.opts
                )
            except Exception as e:
                print(f"⚠️ MSTAR 点云初值缩放跳过: {e}")

        print(f"✅ 加载了 {len(keypoints_dict)} 个图像的关键点")
        print(f"✅ 加载了 {len(matches_dict)} 个图像对的匹配")
        
        # 准备初始位姿（如果可用）
        initial_poses = None
        if len(self.image_data) > 0:
            initial_poses = {}
            for img_name in self.image_names:
                if img_name in self.image_data:
                    R_cam, t_cam, ref = self.image_data[img_name]
                    initial_poses[img_name] = {
                        'R': R_cam,
                        't': t_cam
                    }
        
        # 准备图像角度信息
        # 优先使用self.image_angles（如果启用循环俯视角模式，已包含循环俯视角）
        image_angles = {}
        if hasattr(self, 'image_angles'):
            image_angles = self.image_angles.copy()  # 使用副本，避免修改原始数据
            
            # 检查是否启用了循环俯视角模式（默认启用）
            disable_cyclic_depression = getattr(self.opts, 'disable_cyclic_depression', False) or getattr(
                self.opts, 'sfm_ba_fix_angles_height_only', False
            )
            if self.sar_params.get('ba_fix_angles_optimize_height_only'):
                disable_cyclic_depression = True
            if getattr(self, "_azimuth_ring_dataset", False):
                disable_cyclic_depression = True
            enable_cyclic_depression = not disable_cyclic_depression  # 默认启用
            
            # 明确显示循环俯视角模式状态
            if enable_cyclic_depression:
                print("✅ 循环俯视角模式: 已启用")
                print("✅ SO-RCG将使用循环俯视角模式生成的俯视角")
                # 保存循环俯视角参数用于诊断
                start_angle = getattr(self.opts, 'cyclic_depression_start', 15.0)
                end_angle = getattr(self.opts, 'cyclic_depression_end', 45.0)  # 默认值应与config.py一致
                step_angle = getattr(self.opts, 'cyclic_depression_step', 15.0)  # 默认值应与config.py一致
                print(f"   循环俯视角参数: 起始角度={start_angle}°, 结束角度={end_angle}°, 步长={step_angle}°")
                # 显示前几个图像的俯视角，确认使用的是循环俯视角
                sample_count = min(5, len(self.image_names))
                print(f"   前{sample_count}个图像的俯视角:")
                for i, img_name in enumerate(self.image_names[:sample_count]):
                    if img_name in image_angles:
                        dep = image_angles[img_name].get('depression', 0)
                        azi = image_angles[img_name].get('azimuth', 0)
                        print(f"     - {img_name}: 俯视角={dep}°, 方位角={azi}°")
            else:
                print("ℹ️ 循环俯视角模式: 未启用（使用实际俯视角）")
                # 显示前几个图像的实际俯视角
                sample_count = min(5, len(self.image_names))
                print(f"   前{sample_count}个图像的实际俯视角:")
                for i, img_name in enumerate(self.image_names[:sample_count]):
                    if img_name in image_angles:
                        dep = image_angles[img_name].get('depression', 0)
                        azi = image_angles[img_name].get('azimuth', 0)
                        print(f"     - {img_name}: 俯视角={dep}°, 方位角={azi}°")
        else:
            # 如果没有image_angles，尝试从文件名解析角度信息
            # 但这种情况不应该发生，因为image_angles在__init__中已经初始化
            print("⚠️ 警告: self.image_angles不存在，从文件名解析角度信息")
            for img_name in self.image_names:
                if SAR_GEOMETRY_AVAILABLE:
                    try:
                        dep, azi = parse_mstar_filename(img_name)
                        image_angles[img_name] = {
                            'depression': dep,
                            'azimuth': azi
                        }
                    except:
                        image_angles[img_name] = {
                            'depression': 31.57,
                            'azimuth': 0
                        }
        
        # 创建SO-RCG求解器
        max_iterations = getattr(self.opts, 'sfm_sorc_max_iterations', 100)
        if sar_params_for_solver.get('ba_fix_angles_optimize_height_only'):
            ho_cap = int(getattr(self.opts, 'sfm_sorc_height_only_max_iterations', 20))
            sar_params_for_solver['sorc_height_only_max_iterations'] = ho_cap
            if max_iterations > ho_cap:
                print(f"ℹ️ 固定俯仰/方位：SO-RCG 外层迭代 {max_iterations} → {ho_cap}")
                max_iterations = ho_cap
        tolerance = getattr(self.opts, 'sfm_sorc_tolerance', 1e-6)
        use_5dof = getattr(self.opts, 'sfm_sorc_use_5dof', True)
        
        # GPU加速设置（默认禁用，使用CPU以确保参数计算的规范性）
        use_gpu = getattr(self.opts, 'sfm_sorc_use_gpu', False)
        _ho = bool(
            getattr(self.opts, 'sfm_ba_fix_angles_height_only', False)
            or (hasattr(self, 'sar_params') and self.sar_params.get('ba_fix_angles_optimize_height_only'))
        )
        if not use_gpu and _ho:
            try:
                import torch
                if torch.cuda.is_available():
                    use_gpu = True
                    print("ℹ️ 固定俯仰/方位模式：检测到 CUDA，自动启用 SO-RCG GPU 批处理")
            except ImportError:
                pass
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.device('cuda:0')
                    print(f"✅ SO-RCG将使用GPU加速: {torch.cuda.get_device_name(0)}")
                else:
                    use_gpu = False
                    print("⚠️ CUDA不可用，SO-RCG将使用CPU")
            except ImportError:
                use_gpu = False
                print("⚠️ PyTorch未安装，SO-RCG将使用CPU")
        else:
            device = None
            print("ℹ️ SO-RCG将使用CPU计算（确保参数计算的规范性）")
        
        # 确保SAR参数正确传递，并使用实际读取的图像尺寸
        sar_params_for_solver = self.sar_params.copy() if hasattr(self, 'sar_params') else None
        if sar_params_for_solver is None:
            print("⚠️ 警告: 未找到SAR参数，SO-RCG将使用默认参数")
        else:
            ho_ratio = getattr(self.opts, "sfm_sorc_height_only_search_ratio", 0.35)
            sar_params_for_solver["sorc_height_only_search_ratio"] = float(ho_ratio)
            # 确保使用实际读取的图像尺寸（如果已读取）
            # 优先使用实际读取的图像尺寸，而不是默认值
            # Na和Nr应该等于实际图像尺寸，不是固定值
            if len(self.image_names) > 0:
                # 尝试从第一张图像获取实际尺寸
                first_img_name = self.image_names[0]
                if first_img_name in self.image_sizes:
                    width, height = self.image_sizes[first_img_name]
                    sar_params_for_solver['Na'] = width
                    sar_params_for_solver['Nr'] = height
                    print(f"   ✅ 使用实际图像尺寸更新SAR参数: Na={width}, Nr={height}")
                    
                    # 验证所有图像的尺寸是否一致（SO-RCG要求所有图像尺寸相同）
                    image_sizes_set = set()
                    for img_name in self.image_names:
                        if img_name in self.image_sizes:
                            image_sizes_set.add(self.image_sizes[img_name])
                    
                    if len(image_sizes_set) > 1:
                        print(f"   ⚠️ 警告: 检测到不同尺寸的图像: {image_sizes_set}")
                        print(f"      SO-RCG要求所有图像尺寸相同，可能导致计算错误")
                        print(f"      建议: 预处理时统一所有图像的尺寸")
                    else:
                        print(f"   ✅ 所有图像尺寸一致: {width} x {height}")
                elif os.path.exists(self.images_dir):
                    # 如果image_sizes中没有，尝试读取第一张图像
                    image_files = [f for f in os.listdir(self.images_dir) 
                                  if f.split('.')[-1].lower() in self.opts.ext]
                    if image_files:
                        first_image_path = os.path.join(self.images_dir, image_files[0])
                        first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
                        if first_image is not None:
                            height, width = first_image.shape
                            sar_params_for_solver['Na'] = width
                            sar_params_for_solver['Nr'] = height
                            print(f"   ✅ 从图像文件读取尺寸并更新SAR参数: Na={width}, Nr={height}")
                            
                            # 验证所有图像的尺寸是否一致
                            image_sizes_set = set()
                            for img_file in image_files[:10]:  # 检查前10张图像
                                img_path = os.path.join(self.images_dir, img_file)
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    h, w = img.shape
                                    image_sizes_set.add((w, h))
                            
                            if len(image_sizes_set) > 1:
                                print(f"   ⚠️ 警告: 检测到不同尺寸的图像: {image_sizes_set}")
                                print(f"      SO-RCG要求所有图像尺寸相同，可能导致计算错误")
                            else:
                                print(f"   ✅ 所有图像尺寸一致: {width} x {height}")
            
            # 显示关键SAR参数，确认参数正确
            camera_height = sar_params_for_solver.get('camera_height', '未知')
            fs = sar_params_for_solver.get('fs', '未知')
            prf = sar_params_for_solver.get('prf', '未知')
            Va = sar_params_for_solver.get('Va', '未知')
            Na = sar_params_for_solver.get('Na', '未知')
            Nr = sar_params_for_solver.get('Nr', '未知')
            print(f"   📊 SO-RCG使用的SAR参数:")
            print(f"      - camera_height: {camera_height}米")
            print(f"      - fs: {fs} Hz")
            print(f"      - prf: {prf} Hz")
            print(f"      - Va: {Va} m/s")
            print(f"      - 图像尺寸: {Na} x {Nr}")
            
            # 添加位置约束相关参数（从config.py中读取，如果不存在则使用config.py中的默认值）
            # 注意：对于 action='store_true' 的参数，如果没有在命令行中指定，值为 False
            # 但 config.py 中设置了 default=True，所以我们需要检查属性是否存在
            if hasattr(self.opts, 'sfm_enable_position_constraint'):
                sar_params_for_solver['enable_position_constraint'] = getattr(self.opts, 'sfm_enable_position_constraint')
            else:
                # 如果属性不存在，使用 config.py 中的默认值
                sar_params_for_solver['enable_position_constraint'] = True
            
            if hasattr(self.opts, 'sfm_position_tolerance_ratio'):
                sar_params_for_solver['position_tolerance_ratio'] = getattr(self.opts, 'sfm_position_tolerance_ratio')
            else:
                sar_params_for_solver['position_tolerance_ratio'] = 0.3
            
            if hasattr(self.opts, 'sfm_enable_depression_constraint'):
                sar_params_for_solver['enable_depression_constraint'] = getattr(self.opts, 'sfm_enable_depression_constraint')
            else:
                sar_params_for_solver['enable_depression_constraint'] = True
            
            if hasattr(self.opts, 'sfm_depression_min'):
                sar_params_for_solver['depression_min'] = getattr(self.opts, 'sfm_depression_min')
            else:
                sar_params_for_solver['depression_min'] = 10.0
            
            if hasattr(self.opts, 'sfm_depression_max'):
                sar_params_for_solver['depression_max'] = getattr(self.opts, 'sfm_depression_max')
            else:
                sar_params_for_solver['depression_max'] = 80.0
            # 添加点云高度约束相关参数（从config.py中读取，如果不存在则使用config.py中的默认值）
            if hasattr(self.opts, 'sfm_enable_point_height_constraint'):
                sar_params_for_solver['enable_point_height_constraint'] = getattr(self.opts, 'sfm_enable_point_height_constraint')
            else:
                sar_params_for_solver['enable_point_height_constraint'] = True
            
            _phl = getattr(self.opts, 'sfm_point_height_length_ratio', None)
            if _phl is not None:
                sar_params_for_solver['point_height_length_ratio'] = float(_phl)
            else:
                sar_params_for_solver.pop('point_height_length_ratio', None)
            
            if hasattr(self.opts, 'sfm_point_height_constraint_mode'):
                sar_params_for_solver['point_height_constraint_mode'] = getattr(self.opts, 'sfm_point_height_constraint_mode')
            else:
                sar_params_for_solver['point_height_constraint_mode'] = 'scale'
            
            # 🔧 传递宽度约束参数
            if hasattr(self.opts, 'sfm_enable_point_width_constraint'):
                sar_params_for_solver['enable_point_width_constraint'] = getattr(self.opts, 'sfm_enable_point_width_constraint')
            else:
                sar_params_for_solver['enable_point_width_constraint'] = True  # 默认启用，与高度约束一致
            if getattr(self.opts, 'sfm_ba_fix_angles_height_only', False) or self.sar_params.get(
                'ba_fix_angles_optimize_height_only'
            ):
                sar_params_for_solver['ba_fix_angles_optimize_height_only'] = True
            # 🔧 传递特征提取输出目录，用于查找阴影计算的汇总文件
            feature_output_dir = getattr(self.opts, 'feature_output_dir', None)
            if feature_output_dir is None:
                _dd = getattr(self.opts, 'data_dir', None)
                _vd = getattr(self.opts, 'visualization_dir', None)
                if _dd and _vd:
                    feature_output_dir = os.path.join(_dd, _vd)
            if feature_output_dir is None:
                feature_output_dir = os.path.join(getattr(self.opts, 'out_dir', '.'), getattr(self.opts, 'dataset', ''), 'features')
            sar_params_for_solver['feature_output_dir'] = feature_output_dir
            _td_sum = getattr(self.opts, 'target_dimension_summary_file', None)
            if _td_sum:
                sar_params_for_solver['target_dimension_summary_file'] = _td_sum
        
        # 获取多线程数量设置
        num_threads = getattr(self.opts, 'sfm_sorc_num_threads', None)
        
        # 保存循环俯视角参数到solver（用于诊断）
        solver = SORCG_SAR_SfM(
            sar_params=sar_params_for_solver,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=True,
            use_5dof=use_5dof,
            use_gpu=use_gpu,
            device=device,
            num_threads=num_threads
        )
        
        # 传递循环俯视角参数用于诊断
        # 注意：这些参数在sfm.py的__init__方法中已经从config读取并用于生成循环俯视角序列
        # 这里再次传递给solver，用于诊断信息显示
        disable_cyclic_depression = getattr(self.opts, 'disable_cyclic_depression', False) or getattr(
            self.opts, 'sfm_ba_fix_angles_height_only', False
        )
        if self.sar_params.get('ba_fix_angles_optimize_height_only'):
            disable_cyclic_depression = True
        if getattr(self, "_azimuth_ring_dataset", False):
            disable_cyclic_depression = True
        enable_cyclic_depression = not disable_cyclic_depression  # 默认启用
        if enable_cyclic_depression:
            # 从config中读取循环俯视角参数（与__init__中使用的参数一致）
            solver._cyclic_start = getattr(self.opts, 'cyclic_depression_start', 15.0)
            solver._cyclic_end = getattr(self.opts, 'cyclic_depression_end', 45.0)  # 默认值应与config.py一致
            solver._cyclic_step = getattr(self.opts, 'cyclic_depression_step', 15.0)  # 默认值应与config.py一致
        
        # 求解
        result = solver.solve(
            keypoints_dict=keypoints_dict,
            matches_dict=matches_dict,
            initial_poses=initial_poses,
            image_angles=image_angles
        )

        # 同步求解器内可能更新过的 SAR 参数（例如仅优化共享高度后的 camera_height）
        if hasattr(solver, 'sar_params') and isinstance(solver.sar_params, dict):
            if 'camera_height' in solver.sar_params:
                self.sar_params['camera_height'] = solver.sar_params['camera_height']
        
        # 处理歧义
        result['poses'], result['points_3d'] = solver.handle_ambiguity(
            result['poses'], result['points_3d'], result['point_tracks'],
            keypoints_dict, image_angles
        )
        
        # 更新self.image_data和self.point_cloud
        print("\n更新重建结果...")
        
        # 更新点云
        self.point_cloud = result['points_3d']
        
        # 🔧 关键修复：SO-RCG优化后，从新的相机位置重新计算SAR参数
        print("\n🔧 从优化后的相机位置重新计算SAR参数（R0、俯视角、K矩阵）...")
        
        # 更新图像数据
        for img_name in self.image_names:
            if img_name in result['poses']:
                pose = result['poses'][img_name]
                R_cam = np.array(pose['R'])
                t_cam = np.array(pose['t']).reshape(3, 1)
                
                # 🔧 从新的相机位置计算实际的SAR参数
                # 1. 计算相机位置（世界坐标系）
                P_radar = -R_cam.T @ t_cam.flatten()  # 相机位置
                R0_actual = np.linalg.norm(P_radar)  # 实际斜距
                
                # 2. 俯视角与方位角：若启用「数据给定角度 + 仅优化高度」，不从位姿反算角度，始终用 image_angles
                fix_angles_ba = self.sar_params.get('ba_fix_angles_optimize_height_only', False)
                
                if fix_angles_ba:
                    dep_actual_deg = float(self.image_angles.get(img_name, {}).get('depression', 30.0))
                    azi_actual_deg = float(self.image_angles.get(img_name, {}).get('azimuth', 0.0))
                # 🔧 修复SVD错误：检查R0的有效性
                elif R0_actual < 1e-6 or np.any(np.isnan(P_radar)) or np.any(np.isinf(P_radar)):
                    print(f"   ⚠️ 警告: {img_name} 的相机位置无效 (R0={R0_actual:.2f})，使用原始角度")
                    dep_actual_deg = self.image_angles.get(img_name, {}).get('depression', 30.0)
                    azi_actual_deg = self.image_angles.get(img_name, {}).get('azimuth', 0.0)
                else:
                    # 🔧 修复俯视角计算：从相机位置向量直接计算，而不是使用未缩放的camera_height
                    # 俯视角是相机位置向量与水平面的夹角
                    # COLMAP坐标系约定：Y轴负方向为高度正方向，所以使用Y坐标计算高度
                    # 使用 arcsin(|Y| / R0)，其中Y是相机高度（负值），R0是相机到原点的距离
                    # 注意：P_radar是相机在世界坐标系中的位置，Y坐标负方向是高度
                    camera_y = P_radar[1]  # Y坐标（COLMAP约定：负值表示高度）
                    camera_height_scaled = abs(camera_y)  # 缩放后的相机高度（取绝对值）
                    
                    # 俯视角：depression = arcsin(camera_height_scaled / R0_actual)
                    # 🔧 修复：确保arcsin参数在有效范围内 [-1, 1]
                    if R0_actual > 1e-6:
                        sin_dep = camera_height_scaled / R0_actual
                        sin_dep = np.clip(sin_dep, -1.0, 1.0)  # 限制在有效范围
                        dep_actual_rad = np.arcsin(sin_dep)
                        dep_actual_deg = np.rad2deg(dep_actual_rad)
                    else:
                        # R0太小，使用原始俯视角
                        dep_actual_deg = self.image_angles.get(img_name, {}).get('depression', 30.0)
                    
                    # 方位角：从相机位置的X、Z坐标计算（COLMAP约定：XZ平面是水平面）
                    # COLMAP坐标系：X轴=右，Y轴=下（高度），Z轴=前
                    # 方位角在XZ平面上：azimuth = atan2(Z, X)
                    if np.linalg.norm([P_radar[0], P_radar[2]]) > 1e-6:
                        azi_actual_rad = np.arctan2(P_radar[2], P_radar[0])  # atan2(Z, X)
                        azi_actual_deg = np.rad2deg(azi_actual_rad)
                    else:
                        # 如果X、Y都为0，使用原始方位角
                        azi_actual_deg = self.image_angles.get(img_name, {}).get('azimuth', 0.0)
                
                # 3. 基于新的相机位置和角度重新计算K矩阵
                # 🔧 关键修复：焦距必须基于实际R0计算，而不是使用统一值
                # 使用实际的图像尺寸
                width, height = self.image_sizes.get(img_name, (self.sar_params['Na'], self.sar_params['Nr']))
                
                # 准备SAR参数（使用实际图像尺寸和新的角度）
                temp_params = self.sar_params.copy()
                temp_params['Na'] = width
                temp_params['Nr'] = height
                temp_params['depression_angle'] = dep_actual_deg  # 使用实际俯视角
                
                # 🔧 关键修复：不使用统一焦距，而是基于实际R0计算焦距
                # 移除统一焦距参数，让焦距随R0变化
                # 如果使用统一焦距，当R0改变时焦距不变，会导致投影不匹配
                # 现在焦距将基于实际R0计算，确保投影正确
                
                # 设置实际R0到sar_params，用于焦距计算
                temp_params['actual_R0'] = R0_actual  # 传递实际R0
                temp_params['use_actual_R0_for_focal'] = True  # 标志：使用实际R0计算焦距
                
                # 保留统一的主点（cx, cy），因为它们不随R0变化
                if hasattr(self, 'unified_cx') and hasattr(self, 'unified_cy'):
                    temp_params['unified_cx'] = self.unified_cx
                    temp_params['unified_cy'] = self.unified_cy
                
                # 重新计算K矩阵（基于新的相机位置和实际R0）
                try:
                    _, _, K_new = compute_sar_camera_pose(dep_actual_deg, azi_actual_deg, temp_params)
                    # 检查K矩阵的有效性
                    if np.any(np.isnan(K_new)) or np.any(np.isinf(K_new)):
                        print(f"   ⚠️ 警告: {img_name} 的K矩阵无效，使用原始K矩阵")
                        # 使用原始K矩阵（如果存在）
                        if img_name in self.image_K:
                            K_new = self.image_K[img_name]
                        else:
                            # 使用第一张图像的K矩阵
                            if len(self.image_K) > 0:
                                K_new = list(self.image_K.values())[0]
                            else:
                                # 使用默认K矩阵
                                K_new = np.array([[width * 1.5, 0, width / 2.0],
                                                 [0, height * 1.5, height / 2.0],
                                                 [0, 0, 1]])
                    self.image_K[img_name] = K_new
                except Exception as e:
                    print(f"   ⚠️ 警告: {img_name} 重新计算K矩阵失败 ({e})，使用原始K矩阵")
                    # 使用原始K矩阵（如果存在）
                    if img_name not in self.image_K:
                        # 使用第一张图像的K矩阵或默认值
                        if len(self.image_K) > 0:
                            self.image_K[img_name] = list(self.image_K.values())[0]
                        else:
                            self.image_K[img_name] = np.array([[width * 1.5, 0, width / 2.0],
                                                              [0, height * 1.5, height / 2.0],
                                                              [0, 0, 1]])
                
                # 🔧 关键修复：更新image_angles，使保存时使用重新计算的角度
                # 先保存原始值用于诊断输出
                dep_original = self.image_angles.get(img_name, {}).get('depression', 'N/A')
                azi_original = self.image_angles.get(img_name, {}).get('azimuth', 'N/A')
                
                # 更新image_angles
                if img_name not in self.image_angles:
                    self.image_angles[img_name] = {}
                self.image_angles[img_name]['depression'] = dep_actual_deg
                self.image_angles[img_name]['azimuth'] = azi_actual_deg
                
                # 诊断输出（仅前几个）
                if len([k for k in self.image_K.keys() if k < img_name]) < 3:
                    # 获取原始K矩阵（如果存在）用于对比
                    K_original = None
                    if hasattr(self, 'image_K') and img_name in self.image_K:
                        K_original = self.image_K[img_name]
                    
                    print(f"   📐 {img_name}:")
                    print(f"      - 优化后位置: ({P_radar[0]:.2f}, {P_radar[1]:.2f}, {P_radar[2]:.2f})米")
                    print(f"      - 实际斜距R0: {R0_actual:.2f}米")
                    print(f"      - 实际俯视角: {dep_actual_deg:.2f}° (原始: {dep_original}°)")
                    print(f"      - 实际方位角: {azi_actual_deg:.2f}° (原始: {azi_original}°)")
                    print(f"      - 新K矩阵: fx={K_new[0,0]:.2f}, fy={K_new[1,1]:.2f}", end="")
                    if K_original is not None:
                        fx_old, fy_old = K_original[0,0], K_original[1,1]
                        fx_change = ((K_new[0,0] - fx_old) / fx_old * 100) if fx_old > 0 else 0
                        fy_change = ((K_new[1,1] - fy_old) / fy_old * 100) if fy_old > 0 else 0
                        print(f" (原始: fx={fx_old:.2f}, fy={fy_old:.2f}, 变化: fx={fx_change:+.1f}%, fy={fy_change:+.1f}%)")
                    else:
                        print()
                    print(f"      ✅ 已更新image_angles和K矩阵（焦距基于实际R0={R0_actual:.2f}m计算）")
                
                # 创建ref数组
                kp, desc = self._LoadFeatures(img_name)
                if kp is None:
                    continue
                
                ref = np.full(len(kp), -1, dtype=int)
                
                # 从point_tracks构建ref
                point_tracks = result.get('point_tracks', {})
                if len(point_tracks) == 0:
                    print(f"   ⚠️ 警告: {img_name} 的point_tracks为空，无法构建ref数组")
                    self.image_data[img_name] = [R_cam, t_cam, ref]
                    continue
                
                pid_to_idx = {pid: idx for idx, pid in enumerate(sorted(point_tracks.keys()))}
                ref_count = 0
                
                for pid, track in point_tracks.items():
                    if pid not in pid_to_idx:
                        continue
                    point3d_idx = pid_to_idx[pid]
                    
                    # track可能是列表或元组，每个元素是(img_name, kp_idx)
                    for track_item in track:
                        if isinstance(track_item, (list, tuple)) and len(track_item) >= 2:
                            track_img_name, kp_idx = track_item[0], track_item[1]
                        else:
                            continue
                            
                        if track_img_name == img_name:
                            if kp_idx < len(kp) and point3d_idx < len(self.point_cloud):
                                ref[kp_idx] = point3d_idx
                                ref_count += 1
                            break
                
                if ref_count == 0:
                    print(f"   ⚠️ 警告: {img_name} 没有找到匹配的track，ref数组全为-1")
                else:
                    print(f"   ✅ {img_name}: 构建了 {ref_count} 个ref关联")
                
                self.image_data[img_name] = [R_cam, t_cam, ref]
        
        print("✅ SAR参数重新计算完成")
        
        print("✅ SO-RCG重建完成")
        print(f"   - 重建了 {len(self.point_cloud)} 个3D点")
        print(f"   - 估计了 {len(result['poses'])} 个相机位姿")
        
        # ========== 新增：计算并保存实际使用的尺度参数 ==========
        # SO-RCG虽然不使用统一参数，但最终点云被归一化到了合理范围
        # 我们需要估算这个尺度因子，以便GS训练时使用
        if len(self.point_cloud) > 0:
            # 计算点云的实际尺度
            points = np.array(self.point_cloud)
            point_center = points.mean(axis=0)
            point_max_extent = float(np.max(np.linalg.norm(points - point_center, axis=1)))
            
            if hasattr(self, 'unified_sita0_for_radius'):
                avg_dep_deg = float(np.rad2deg(self.unified_sita0_for_radius))
            else:
                avg_dep_deg = 30.0
            
            # 优先：MSTAR/height-only 用公式闭合 H；全自由度 SO-RCG 才用 pose 中位 |Y|
            from sar.sfm_scale_params import (
                build_sorc_final_scale_update,
                finite_json_float,
                merge_scale_params_file,
                save_scale_params_json,
            )

            pose_stats = self._scene_geometry_stats_from_poses()
            ur = getattr(self, 'unified_radius_scale', None)
            est_fallback = float(ur) if ur is not None and math.isfinite(float(ur)) else 0.1
            _r0_hint = float(
                pose_stats['median_slant_range_m'] if pose_stats else point_max_extent * 5.0
            )
            estimated_radius_scale = finite_json_float(
                (point_max_extent * 100.0) / max(_r0_hint, 1e-12),
                fallback=est_fallback,
                lo=1e-20,
                hi=1e6,
            )

            scale_update, resolved = build_sorc_final_scale_update(
                sar_params=self.sar_params,
                pose_stats=pose_stats,
                physical_platform_height_m=self._physical_platform_height_m,
                point_max_extent=point_max_extent,
                avg_depression_deg=avg_dep_deg,
                estimated_radius_scale=estimated_radius_scale,
                camera_distribution_mode=str(
                    self.sar_params.get('camera_distribution_mode', 'ring')
                ),
                args=getattr(self.opts, "_full_cli_args", None) or self.opts,
            )
            camera_height_scene = resolved.camera_height_m
            R0_ref = resolved.slant_range_m
            r0_source = resolved.source

            print(f"\n📐 尺度参数估算（写入 sar_scale_params.json）:")
            print(f"   - 点云最大扩展: {point_max_extent:.2f} m")
            print(f"   - 统一俯视角(参考): {avg_dep_deg:.2f}°")
            print(f"   - 参考斜距 R0: {R0_ref:.4f} m  （来源: {r0_source}）")
            print(f"   - 场景单位 camera_height: {camera_height_scene:.4f} m")
            if resolved.use_pose_median:
                print("   ⚠️ 使用重建位姿中位 |Y|（仅全自由度 SO-RCG）")
            print(f"   - 配置物理平台高度(对照): {self._physical_platform_height_m:.2f} m")
            print(f"   - 估算 estimated_radius_scale: {estimated_radius_scale:.6g}")

            scale_params_file = os.path.join(self.opts.data_dir, 'sar_scale_params.json')
            scale_params = merge_scale_params_file(scale_params_file, scale_update)
            
            # 添加所有相机数据
            if hasattr(self, 'image_data') and len(self.image_data) > 0:
                try:
                    cameras_dict = self._CollectCameraData()
                    scale_params['cameras'] = cameras_dict
                    print(f"📷 已添加 {len(cameras_dict)} 个相机的数据到尺度参数文件")
                except Exception as e:
                    print(f"⚠️  收集相机数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            with open(scale_params_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(scale_params, f, indent=2)
            print(f"✅ 已保存SAR尺度参数: {scale_params_file}")
            print(f"   GS训练将使用 radius_scale={estimated_radius_scale:.4f}")
            
            # 显示保存的关键参数
            if 'unified_focal_azimuth' in scale_params:
                print(f"   ✅ 已保留焦距参数: focal_azimuth={scale_params['unified_focal_azimuth']:.2f}, focal_range={scale_params['unified_focal_range']:.2f}")
            else:
                print(f"   ⚠️  警告：缺少焦距参数，GS训练时将重新计算（可能导致尺度不一致）")
        # ========== 新增结束 ==========
        
        # 保存点云
        if len(self.point_cloud) > 0:
            self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_sorc.ply'))
        
        # SAR Bundle Adjustment（如果启用）
        if getattr(self.opts, 'sfm_use_sar_ba', False):
            self._RunSarBundleAdjustment()
        
        # 点云过滤：基于重投影误差过滤低质量点（改善点云形状）
        self._FilterPointsByReprojectionError()
        
        # 保存相机位姿和点云统计信息
        self._SaveCameraPoses()
        self._PrintPointCloudStatistics()
        
        # 计算并打印重投影误差
        if len(self.image_data) > 0:
            print("\n📊 重投影误差统计:")
            errors = []
            for img_name in self.image_names:
                if img_name in self.image_data:
                    try:
                        err = self._ComputeReprojectionError(img_name)
                        if err is not None:
                            errors.append(err)
                            print(f"   - {img_name}: {err:.4f} 像素")
                    except:
                        pass
            
            if errors:
                mean_error = np.mean(errors)
                print(f"   - 平均重投影误差: {mean_error:.4f} 像素")
        
        return result
    
    def RunMatrixFactorization(self):
        """
        使用矩阵分解方法进行SAR SfM重建
        
        这是基于论文"SAR Structure-from-Motion via Matrix Factorization"的方法
        使用Riemannian共轭梯度算法在Stiefel流形上优化
        """
        if not MATRIX_FACTORIZATION_AVAILABLE:
            raise ImportError("❌ 错误: sar_matrix_factorization_sfm模块未找到，无法使用矩阵分解方法")
        
        print("\n" + "="*60)
        print("🔧 使用矩阵分解方法进行SAR SfM重建")
        print("="*60)
        
        # 准备数据
        keypoints_dict = {}
        matches_dict = {}
        
        # 加载所有关键点
        for img_name in self.image_names:
            kp, desc = self._LoadFeatures(img_name)
            if kp is not None and len(kp) > 0:
                keypoints_dict[img_name] = kp
        
        # 加载所有匹配
        for i, name1 in enumerate(self.image_names):
            for name2 in self.image_names[i+1:]:
                matches = self._LoadMatches(name1, name2)
                if len(matches) > 0:
                    matches_dict[(name1, name2)] = matches
        
        if len(keypoints_dict) == 0:
            raise ValueError("❌ 错误: 没有找到任何关键点")
        
        if len(matches_dict) == 0:
            raise ValueError("❌ 错误: 没有找到任何匹配")
        
        print(f"✅ 加载了 {len(keypoints_dict)} 个图像的关键点")
        print(f"✅ 加载了 {len(matches_dict)} 个图像对的匹配")
        
        # 准备初始位姿（如果可用）
        initial_poses = None
        if len(self.image_data) > 0:
            initial_poses = {}
            for img_name in self.image_names:
                if img_name in self.image_data:
                    R_cam, t_cam, ref = self.image_data[img_name]
                    initial_poses[img_name] = {
                        'R': R_cam,
                        't': t_cam
                    }
        
        # 创建矩阵分解求解器
        max_iterations = getattr(self.opts, 'sfm_matrix_factorization_max_iterations', 100)
        tolerance = getattr(self.opts, 'sfm_matrix_factorization_tolerance', 1e-6)
        
        solver = SARMatrixFactorizationSfM(
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=True
        )
        
        # 求解
        result = solver.solve(
            keypoints_dict=keypoints_dict,
            matches_dict=matches_dict,
            initial_poses=initial_poses
        )
        
        # 转换为位姿和3D点
        # 传递初始位姿、图像角度和SAR参数，以便恢复平移向量
        poses_dict, points_3d = solver.convert_to_poses_and_points(
            result, 
            initial_poses=initial_poses,
            image_angles=getattr(self, 'image_angles', None),
            sar_params=getattr(self, 'sar_params', None)
        )
        
        # 更新self.image_data和self.point_cloud
        print("\n更新重建结果...")
        
        # 更新点云
        self.point_cloud = points_3d
        
        # 更新图像数据
        for img_name in self.image_names:
            if img_name in poses_dict:
                pose = poses_dict[img_name]
                R_cam = np.array(pose['R'])
                t_cam = np.array(pose['t']).reshape(3, 1)
                
                # 创建ref数组（将3D点映射到2D点）
                # 这里需要根据point_tracks来构建ref
                kp, desc = self._LoadFeatures(img_name)
                if kp is None:
                    continue
                
                ref = np.full(len(kp), -1, dtype=int)
                
                # 从result中获取该图像的观测信息
                point_tracks = result['point_tracks']
                image_point_indices = result.get('image_point_indices', {})
                
                if img_name in image_point_indices:
                    img_point_indices = image_point_indices[img_name]
                    point_id_to_col = result['point_id_to_col']
                    
                    for pid, kp_idx in img_point_indices.items():
                        if pid in point_id_to_col:
                            col_idx = point_id_to_col[pid]
                            if col_idx < len(self.point_cloud):
                                ref[kp_idx] = col_idx
                
                self.image_data[img_name] = [R_cam, t_cam, ref]
        
        print("✅ 矩阵分解重建完成")
        print(f"   - 重建了 {len(self.point_cloud)} 个3D点")
        print(f"   - 估计了 {len(poses_dict)} 个相机位姿")
        
        # 保存点云
        if len(self.point_cloud) > 0:
            self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_matrix_factorization.ply'))
        
        # SAR Bundle Adjustment（如果启用）
        if getattr(self.opts, 'sfm_use_sar_ba', False):
            self._RunSarBundleAdjustment()
        
        # 保存相机位姿和点云统计信息
        self._SaveCameraPoses()
        self._PrintPointCloudStatistics()
        
        # 计算并打印重投影误差
        if len(self.image_data) > 0:
            print("\n📊 重投影误差统计:")
            errors = []
            for img_name in self.image_names:
                if img_name in self.image_data:
                    err = self._ComputeReprojectionError(img_name)
                    errors.append(err)
                    print(f"   - {img_name}: {err:.4f} 像素")
            
            if len(errors) > 0:
                mean_error = sum(errors) / len(errors)
                print(f"   - 平均重投影误差: {mean_error:.4f} 像素")
        
        return result
        

def SetArguments(parser): 

    #directory stuff
    parser.add_argument('--data_dir',action='store',type=str,default='../data/',dest='data_dir',
                        help='root directory containing input data (default: ../data/)') 
    parser.add_argument('--dataset',action='store',type=str,default='fountain-P11',dest='dataset',
                        help='name of dataset (default: fountain-P11)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext', 
                        help='comma seperated string of allowed image extensions \
                        (default: jpg,png)') 
    parser.add_argument('--out_dir',action='store',type=str,default='../results/',dest='out_dir',
                        help='root directory to store results in (default: ../results/)') 

    #matching parameters
    parser.add_argument('--features',action='store',type=str,default='SURF',dest='features',
                        help='[SIFT|SURF] Feature algorithm to use (default: SURF)')
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross_check',action='store',type=bool,default=True,dest='cross_check',
                        help='[True|False] Whether to cross check feature matching or not \
                        (default: True)') 

    #epipolar geometry parameters（SAR图像使用SAR几何模型）
    parser.add_argument('--calibration_mat',action='store',type=str,default='sar',
                        dest='calibration_mat',help='SAR图像固定使用sar几何模型')
    parser.add_argument('--fund_method',action='store',type=str,default='FM_RANSAC',
                        dest='fund_method',help='method to estimate fundamental matrix \
                        (default: FM_RANSAC)')
    parser.add_argument('--outlier_thres',action='store',type=float,default=.9,
                        dest='outlier_thres',help='threhold value of outlier to be used in\
                         fundamental matrix estimation (default: 0.9)')
    parser.add_argument('--fund_prob',action='store',type=float,default=.9,dest='fund_prob',
                        help='confidence in fundamental matrix estimation required (default: 0.9)')
    
    #PnP parameters
    parser.add_argument('--pnp_method',action='store',type=str,default='SOLVEPNP_DLS',
                        dest='pnp_method',help='[SOLVEPNP_DLS|SOLVEPNP_EPNP|..] method used for\
                        PnP estimation, see OpenCV doc for more options (default: SOLVEPNP_DLS')
    parser.add_argument('--pnp_prob',action='store',type=float,default=.99,dest='pnp_prob',
                        help='confidence in PnP estimation required (default: 0.99)')
    parser.add_argument('--reprojection_thres',action='store',type=float,default=8.,
                        dest='reprojection_thres',help='reprojection threshold in PnP estimation \
                        (default: 8.)')

    #misc
    parser.add_argument('--plot_error',action='store',type=bool,default=False,dest='plot_error')
    
    # 矩阵分解方法参数（现在这是唯一支持的方法）
    parser.add_argument('--sfm_matrix_factorization_max_iterations',action='store',type=int,default=100,
                        dest='sfm_matrix_factorization_max_iterations',
                        help='矩阵分解方法的最大迭代次数（默认: 100）')
    parser.add_argument('--sfm_matrix_factorization_tolerance',action='store',type=float,default=1e-6,
                        dest='sfm_matrix_factorization_tolerance',
                        help='矩阵分解方法的收敛容差（默认: 1e-6）')

def PostprocessArgs(opts): 
    opts.fund_method = getattr(cv2,opts.fund_method)
    opts.ext = opts.ext.split(',')

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)
    
    sfm = SFM(opts)
    
    # 现在只支持矩阵分解方法
    print("="*60)
    print("使用矩阵分解方法进行SAR SfM重建")
    print("（增量式方法已被移除，现在只支持矩阵分解方法）")
    print("="*60)
    sfm.Run()