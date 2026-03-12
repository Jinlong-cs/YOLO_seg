#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_seg.x5_compare import load_config, run_stage, write_manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Build or execute the float/PTQ/QAT comparison workflow.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "x5_compare" / "default.yaml"), help="Experiment config yaml.")
    parser.add_argument(
        "--stage",
        choices=["manifest", "float_train", "float_export_ptq_shape", "ptq_compile", "qat_env_setup", "qat_probe"],
        default="manifest",
        help="Workflow stage to execute.",
    )
    parser.add_argument("--execute", action="store_true", help="Execute the selected stage instead of only generating the manifest.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    manifest_path, shell_path, manifest = write_manifest(config)
    print(f"Manifest: {manifest_path}")
    print(f"Stage shell: {shell_path}")
    if args.stage != "manifest":
        stage_info = manifest["stages"][args.stage]
        print(f"Stage command: {' '.join(stage_info['command'])}")
        if args.execute:
            run_stage(config, args.stage)


if __name__ == "__main__":
    main()
