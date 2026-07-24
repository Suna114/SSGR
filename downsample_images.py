#!/usr/bin/env python3
"""High-quality image downsampling with anti-aliasing.

Reduces image resolution while preserving sharpness by combining:
  1) optional Gaussian pre-blur to suppress aliasing
  2) progressive 2x stepping for large scale reductions
  3) Lanczos / bicubic final resize

Examples:
  python downsample_images.py input.jpg output.jpg --scale 0.5
  python downsample_images.py ./picture ./picture_128 --size 128 128
  python downsample_images.py ./data ./data_half --scale 0.5 --method antialias
  python downsample_images.py ./data ./data_small --max-size 512 --recursive
"""

import argparse
import math
from pathlib import Path

from PIL import Image, ImageFilter

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(path: Path, recursive: bool = False):
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTS else []
    if not path.is_dir():
        return []
    iterator = path.rglob("*") if recursive else path.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files, key=lambda p: str(p).lower())


def compute_target_size(width: int, height: int, args) -> tuple[int, int]:
    if args.size is not None:
        tw, th = args.size
        if args.keep_aspect:
            scale = min(tw / width, th / height)
            return max(1, round(width * scale)), max(1, round(height * scale))
        return tw, th

    if args.max_size is not None:
        longest = max(width, height)
        if longest <= args.max_size:
            return width, height
        scale = args.max_size / longest
        return max(1, round(width * scale)), max(1, round(height * scale))

    if args.scale is not None:
        return max(1, round(width * args.scale)), max(1, round(height * args.scale))

    raise ValueError("Must specify one of --scale, --size, or --max-size")


def gaussian_blur_for_scale(img: Image.Image, scale: float) -> Image.Image:
    """Apply a light blur before downsampling to reduce aliasing."""
    if scale >= 1.0:
        return img
    # sigma grows as we shrink more aggressively
    sigma = max(0.0, (1.0 / scale - 1.0) * 0.5)
    if sigma < 0.3:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_lanczos(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.LANCZOS)


def resize_bicubic(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.BICUBIC)


def resize_box(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.BOX)


def progressive_downsample(
    img: Image.Image,
    target_size: tuple[int, int],
    final_filter,
    pre_blur: bool = True,
) -> Image.Image:
    """Downsample in ~2x steps to preserve detail and limit aliasing."""
    tw, th = target_size
    w, h = img.size
    if (w, h) == (tw, th):
        return img.copy()

    if w <= tw and h <= th:
        return final_filter(img, target_size)

    current = img
    cw, ch = w, h
    while cw > 2 * tw or ch > 2 * th:
        cw = max(tw, cw // 2)
        ch = max(th, ch // 2)
        if pre_blur:
            step_scale = min(cw / w, ch / h)
            current = gaussian_blur_for_scale(current, step_scale)
        current = resize_lanczos(current, (cw, ch))

    if pre_blur:
        final_scale = min(tw / w, th / h)
        current = gaussian_blur_for_scale(current, final_scale)
    return final_filter(current, target_size)


def unsharp_mask(img: Image.Image, amount: float, radius: float = 1.0) -> Image.Image:
    if amount <= 0:
        return img
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return Image.blend(blurred, img, 1.0 + amount)


def downsample_image(img: Image.Image, target_size: tuple[int, int], args) -> Image.Image:
    method = args.method
    if method == "lanczos":
        out = resize_lanczos(img, target_size)
    elif method == "bicubic":
        out = resize_bicubic(img, target_size)
    elif method == "box":
        out = resize_box(img, target_size)
    elif method == "antialias":
        final_filter = resize_lanczos if args.final_filter == "lanczos" else resize_bicubic
        out = progressive_downsample(img, target_size, final_filter, pre_blur=not args.no_pre_blur)
    else:
        raise ValueError(f"Unknown method: {method}")

    if args.sharpen > 0:
        out = unsharp_mask(out, amount=args.sharpen, radius=args.sharpen_radius)
    return out


def resolve_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        if output_root.suffix.lower() in IMAGE_EXTS:
            return output_root
        return output_root / input_path.name
    rel = input_path.relative_to(input_root)
    return output_root / rel


def parse_args():
    parser = argparse.ArgumentParser(description="High-quality image downsampling")
    parser.add_argument("input", type=Path, help="Input image file or directory")
    parser.add_argument("output", type=Path, help="Output image file or directory")
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--scale", type=float, help="Scale factor, e.g. 0.5 for half size")
    size_group.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        help="Target width and height",
    )
    size_group.add_argument(
        "--max-size",
        type=int,
        help="Limit the longest side to this many pixels",
    )
    parser.add_argument(
        "--method",
        default="antialias",
        choices=["antialias", "lanczos", "bicubic", "box"],
        help="Downsampling method (default: antialias)",
    )
    parser.add_argument(
        "--final-filter",
        default="lanczos",
        choices=["lanczos", "bicubic"],
        help="Final interpolation when using antialias method",
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Keep aspect ratio when using --size",
    )
    parser.add_argument(
        "--no-pre-blur",
        action="store_true",
        help="Disable Gaussian pre-blur in antialias mode",
    )
    parser.add_argument(
        "--sharpen",
        type=float,
        default=0.0,
        help="Unsharp mask amount after resize, e.g. 0.2~0.5",
    )
    parser.add_argument(
        "--sharpen-radius",
        type=float,
        default=1.0,
        help="Unsharp mask blur radius",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG output quality (default: 95)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process subdirectories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing files",
    )
    return parser.parse_args()


def save_image(img: Image.Image, output_path: Path, quality: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
    elif output_path.suffix.lower() == ".png":
        save_kwargs["compress_level"] = 3
    img.save(output_path, **save_kwargs)


def is_single_file_output(input_root: Path, output_root: Path, num_images: int) -> bool:
    return (
        num_images == 1
        and input_root.is_file()
        and output_root.suffix.lower() in IMAGE_EXTS
    )


def main():
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    images = list_images(input_root, recursive=args.recursive)
    if not images:
        raise SystemExit(f"No images found under: {input_root}")

    if input_root.is_dir() and output_root.suffix.lower() in IMAGE_EXTS:
        raise SystemExit("Output must be a directory when input is a directory.")

    single_file_output = is_single_file_output(input_root, output_root, len(images))

    processed = 0
    skipped = 0
    for image_path in images:
        out_path = resolve_output_path(image_path, input_root, output_root)
        in_place = out_path.resolve() == image_path.resolve()
        allow_overwrite = args.overwrite or in_place or single_file_output
        if out_path.exists() and not allow_overwrite:
            print(f"Skipped (exists): {out_path}  [add --overwrite to replace]")
            skipped += 1
            continue

        with Image.open(image_path) as img:
            img.load()
            target_size = compute_target_size(img.width, img.height, args)
            if target_size == (img.width, img.height):
                print(
                    f"Skipped (already target size): {image_path.name} "
                    f"{img.width}x{img.height}"
                )
                skipped += 1
                continue
            result = downsample_image(img, target_size, args)

        scale = min(target_size[0] / img.width, target_size[1] / img.height)
        msg = (
            f"{image_path.name}: {img.width}x{img.height} -> "
            f"{target_size[0]}x{target_size[1]} ({scale:.3f}x, {args.method})"
        )
        if args.dry_run:
            print(f"[dry-run] {msg} -> {out_path}")
            processed += 1
            continue

        save_image(result, out_path, args.quality)
        print(msg)
        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, total={len(images)}")


if __name__ == "__main__":
    main()
