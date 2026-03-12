#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_seg.qat_probe import run_qat_probe


def build_parser():
    parser = argparse.ArgumentParser(description="Run a Horizon QAT prepare/convert/compile probe on the current YOLO-seg model.")
    parser.add_argument("--pt", required=True, help="Float checkpoint path.")
    parser.add_argument("--imgsz", type=int, nargs=2, required=True, metavar=("H", "W"), help="Probe input size.")
    parser.add_argument("--output-dir", required=True, help="Directory for probe outputs.")
    parser.add_argument("--march", default="BAYES_E", help="Target march.")
    parser.add_argument("--device", default="cpu", help="Torch device for QAT preparation.")
    parser.add_argument("--input-source", default="ddr", help="Compile input_source, usually ddr or pyramid.")
    parser.add_argument("--compile-opt", type=int, default=0, help="Compile optimization level.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_qat_probe(
        pt_path=args.pt,
        imgsz=tuple(args.imgsz),
        output_dir=Path(args.output_dir),
        march=args.march,
        device=args.device,
        input_source=args.input_source,
        compile_opt=args.compile_opt,
    )


if __name__ == "__main__":
    main()
