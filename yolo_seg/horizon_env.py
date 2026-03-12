from __future__ import annotations

from pathlib import Path


DEFAULT_OPEN_EXPLORER_ROOT = Path(
    "/home/supernova/wujinlong/RDK_X5_tools/horizon_x5_open_explorer_v1.2.8-py310_20240926"
)


def ai_toolchain_dir(open_explorer_root=None):
    root = Path(open_explorer_root or DEFAULT_OPEN_EXPLORER_ROOT).resolve()
    return root / "package" / "host" / "ai_toolchain"


def find_single_wheel(wheel_dir, pattern):
    matches = sorted(Path(wheel_dir).glob(pattern))
    if not matches:
        raise SystemExit(f"wheel not found for pattern: {pattern} under {wheel_dir}")
    return matches[0]


def locate_horizon_wheels(open_explorer_root=None):
    wheel_dir = ai_toolchain_dir(open_explorer_root)
    if not wheel_dir.exists():
        raise SystemExit(f"ai_toolchain wheel directory not found: {wheel_dir}")

    return {
        "wheel_dir": wheel_dir,
        "horizon_plugin_pytorch": find_single_wheel(wheel_dir, "horizon_plugin_pytorch-*.whl"),
        "horizon_tc_ui": find_single_wheel(wheel_dir, "horizon_tc_ui-*.whl"),
        "hbdk": find_single_wheel(wheel_dir, "hbdk-*.whl"),
        "horizon_nn": find_single_wheel(wheel_dir, "horizon_nn-*.whl"),
        "horizon_plugin_profiler": find_single_wheel(wheel_dir, "horizon_plugin_profiler-*.whl"),
        "hbdk_model_verifier": find_single_wheel(wheel_dir, "hbdk_model_verifier-*.whl"),
    }
