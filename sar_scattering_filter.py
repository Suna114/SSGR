"""兼容入口：实现见 sar.scattering_filter。"""
from pathlib import Path
import runpy

from sar.scattering_filter import *  # noqa: F403

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "sar" / "scattering_filter.py"), run_name="__main__")
