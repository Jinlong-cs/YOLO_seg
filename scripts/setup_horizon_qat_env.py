#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_seg.horizon_env import DEFAULT_OPEN_EXPLORER_ROOT, locate_horizon_wheels


def build_parser():
    parser = argparse.ArgumentParser(description="Create a dedicated Horizon QAT Python environment.")
    parser.add_argument("--venv-path", default=str(ROOT / ".venv.horizon_qat"), help="Virtualenv path.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to create the venv.")
    parser.add_argument("--open-explorer-root", default=str(DEFAULT_OPEN_EXPLORER_ROOT), help="OpenExplorer root path.")
    parser.add_argument("--torch-version", default="1.13.0", help="Torch version expected by the Horizon plugin wheel.")
    parser.add_argument("--torchvision-version", default="0.14.0", help="Torchvision version expected by the Horizon plugin wheel.")
    parser.add_argument("--extra-index-url", default="https://download.pytorch.org/whl/cu116", help="Extra index used for torch/torchvision installation.")
    parser.add_argument("--execute", action="store_true", help="Actually create the venv and install packages.")
    return parser


def venv_python(venv_path):
    return Path(venv_path) / "bin" / "python"


def build_commands(args):
    wheels = locate_horizon_wheels(args.open_explorer_root)
    vp = venv_python(args.venv_path)
    commands = [
        [args.python, "-m", "venv", args.venv_path],
        [str(vp), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        [
            str(vp),
            "-m",
            "pip",
            "install",
            f"torch=={args.torch_version}",
            f"torchvision=={args.torchvision_version}",
            "--extra-index-url",
            args.extra_index_url,
        ],
        [str(vp), "-m", "pip", "install", "-r", str(ROOT / "requirements-horizon-qat.txt")],
        [
            str(vp),
            "-m",
            "pip",
            "install",
            str(wheels["horizon_plugin_pytorch"]),
            str(wheels["horizon_tc_ui"]),
            str(wheels["hbdk"]),
            str(wheels["horizon_nn"]),
            str(wheels["horizon_plugin_profiler"]),
            str(wheels["hbdk_model_verifier"]),
        ],
        [
            str(vp),
            "-c",
            "import horizon_plugin_pytorch, horizon_tc_ui, hbdk; print('QAT environment ready')",
        ],
    ]
    return commands


def main(argv=None):
    args = build_parser().parse_args(argv)
    commands = build_commands(args)
    for command in commands:
        print(" ".join(command))
    if args.execute:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
