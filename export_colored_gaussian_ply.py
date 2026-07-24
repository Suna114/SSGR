#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练好的高斯模型导出带顶点色（red/green/blue）的 PLY，便于在 MeshLab / CloudCompare
等软件中区分结构。不修改训练脚本与 GaussianModel 默认保存逻辑。

颜色模式（--mode）
- dc: 与 scene/GaussianModel.save_ply 一致：0 阶球谐 DC 经 SH2RGB 得到近似颜色。
- dc_stretch: 在 dc 基础上按通道做分位数拉伸（默认 2%~98%），若 SAR/灰度场景下
  各点 DC 接近、肉眼几乎一色，可用此模式拉开差异。
- spatial: 由点在场景包围盒中的归一化坐标着色（伪彩），不表示真实纹理，但最易分辨几何。
- opacity: 由不透明度映射 HSV 伪彩，便于观察透明度分布。

用法（需训练输出目录内含 cfg_args，与 scripts/rendering/render_and_save.py 类似）：

    python export_colored_gaussian_ply.py -m output/your_run --iteration 30000
    python export_colored_gaussian_ply.py -m output/your_run --iteration -1 --mode dc_stretch
    python export_colored_gaussian_ply.py -m output/your_run --mode spatial --simple

默认输出：
    <model_path>/point_cloud/iteration_<N>/point_cloud_colored_<mode>.ply

--simple 会额外写出仅 x,y,z,r,g,b 的 *_xyzrgb.ply，兼容部分只认简单 PLY 的查看器。

查看器提示：需在软件中开启「顶点颜色 / per-vertex color」，部分程序默认按法线或无着色显示。
"""

import os
import colorsys
import numpy as np
import torch
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.sh_utils import SH2RGB
from utils.system_utils import mkdir_p


def _rgb_u8_from_linear_rgb(rgb_float):
    rgb_float = np.clip(rgb_float.astype(np.float64), 0.0, 1.0)
    return (rgb_float * 255.0).round().astype(np.uint8)


def _colors_dc(gaussians):
    f = gaussians._features_dc.detach().squeeze(1)
    rgb = torch.clamp(SH2RGB(f), 0.0, 1.0)
    return rgb.detach().cpu().numpy()


def _colors_dc_stretch(gaussians, low_pct, high_pct):
    rgb = _colors_dc(gaussians)
    out = rgb.copy()
    for c in range(3):
        lo = np.percentile(out[:, c], low_pct)
        hi = np.percentile(out[:, c], high_pct)
        if hi <= lo + 1e-8:
            continue
        out[:, c] = (out[:, c] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _colors_spatial(gaussians):
    xyz = gaussians._xyz.detach().cpu().numpy()
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    span = np.maximum(mx - mn, 1e-8)
    return np.clip((xyz - mn) / span, 0.0, 1.0)


def _colors_opacity(gaussians):
    op = gaussians.get_opacity.detach().cpu().numpy().reshape(-1)
    t = np.clip(op, 0.0, 1.0)
    rgb = np.zeros((len(t), 3), dtype=np.float64)
    for i, v in enumerate(t):
        rgb[i] = colorsys.hsv_to_rgb(v * 0.85, 0.9, 0.95)
    return np.clip(rgb, 0.0, 1.0)


def compute_colors(gaussians, mode, stretch_low=2.0, stretch_high=98.0):
    mode = mode.lower()
    if mode == "dc":
        return _colors_dc(gaussians)
    if mode == "dc_stretch":
        return _colors_dc_stretch(gaussians, stretch_low, stretch_high)
    if mode == "spatial":
        return _colors_spatial(gaussians)
    if mode == "opacity":
        return _colors_opacity(gaussians)
    raise ValueError(f"未知模式: {mode}. 可选: dc, dc_stretch, spatial, opacity")


def write_full_ply_with_colors(gaussians, path, rgb_float):
    """与 GaussianModel.save_ply 相同属性列，仅自定义 red/green/blue。"""
    rgb_u8 = _rgb_u8_from_linear_rgb(rgb_float)
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        mkdir_p(out_dir)

    xyz = gaussians._xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()

    attr_names = gaussians.construct_list_of_attributes(include_vertex_rgb=True)
    dtype_full = [(a, "u1") if a in ("red", "green", "blue") else (a, "f4") for a in attr_names]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements["x"] = xyz[:, 0].astype(np.float32)
    elements["y"] = xyz[:, 1].astype(np.float32)
    elements["z"] = xyz[:, 2].astype(np.float32)
    elements["nx"] = normals[:, 0].astype(np.float32)
    elements["ny"] = normals[:, 1].astype(np.float32)
    elements["nz"] = normals[:, 2].astype(np.float32)
    elements["red"] = rgb_u8[:, 0]
    elements["green"] = rgb_u8[:, 1]
    elements["blue"] = rgb_u8[:, 2]
    for i in range(f_dc.shape[1]):
        elements[f"f_dc_{i}"] = f_dc[:, i].astype(np.float32)
    for i in range(f_rest.shape[1]):
        elements[f"f_rest_{i}"] = f_rest[:, i].astype(np.float32)
    elements["opacity"] = opacities[:, 0].astype(np.float32)
    for i in range(scale.shape[1]):
        elements[f"scale_{i}"] = scale[:, i].astype(np.float32)
    for i in range(rotation.shape[1]):
        elements[f"rot_{i}"] = rotation[:, i].astype(np.float32)
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def write_simple_xyzrgb_ply(path, xyz, rgb_u8):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        mkdir_p(out_dir)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements["x"] = xyz[:, 0].astype(np.float32)
    elements["y"] = xyz[:, 1].astype(np.float32)
    elements["z"] = xyz[:, 2].astype(np.float32)
    elements["red"] = rgb_u8[:, 0]
    elements["green"] = rgb_u8[:, 1]
    elements["blue"] = rgb_u8[:, 2]
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def main():
    parser = ArgumentParser(description="导出带可辨顶点色的高斯 PLY（独立工具脚本）")
    model = ModelParams(parser, sentinel=True)
    _ = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="加载迭代（-1 表示最新）")
    parser.add_argument(
        "--mode",
        type=str,
        default="dc_stretch",
        choices=["dc", "dc_stretch", "spatial", "opacity"],
        help="dc=与训练一致；dc_stretch=对比度增强；spatial/opacity=伪彩",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 PLY 路径")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="额外写出仅 xyz+rgb 的简化 PLY（*_xyzrgb.ply）",
    )
    parser.add_argument("--stretch_low", type=float, default=2.0, help="dc_stretch 分位数下限")
    parser.add_argument("--stretch_high", type=float, default=98.0, help="dc_stretch 分位数上限")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    safe_state(args.quiet)

    model_params = model.extract(args)
    gaussians = GaussianModel(model_params.sh_degree)
    scene = Scene(model_params, gaussians, load_iteration=args.iteration, shuffle=False)
    it = scene.loaded_iter

    rgb_float = compute_colors(
        gaussians,
        args.mode,
        stretch_low=args.stretch_low,
        stretch_high=args.stretch_high,
    )

    if args.output:
        out_full = args.output
    else:
        out_full = os.path.join(
            model_params.model_path,
            "point_cloud",
            f"iteration_{it}",
            f"point_cloud_colored_{args.mode}.ply",
        )

    write_full_ply_with_colors(gaussians, out_full, rgb_float)
    print(f"已写出: {out_full}")

    if args.simple:
        base, ext = os.path.splitext(out_full)
        ext = ext if ext else ".ply"
        out_simple = f"{base}_xyzrgb{ext}"
        xyz = gaussians._xyz.detach().cpu().numpy()
        rgb_u8 = _rgb_u8_from_linear_rgb(rgb_float)
        write_simple_xyzrgb_ply(out_simple, xyz, rgb_u8)
        print(f"已写出简化 PLY: {out_simple}")


if __name__ == "__main__":
    main()
