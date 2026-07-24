"""兼容入口：实现见 sar.mpa_scattering。"""
from pathlib import Path
import runpy

from sar.mpa_scattering import *  # noqa: F403

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "sar" / "mpa_scattering.py"), run_name="__main__")
