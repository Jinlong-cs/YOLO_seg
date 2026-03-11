import argparse
import shutil
from pathlib import Path

from yolo_seg.ultralytics_rdk import patch_model_for_rdk


def build_parser():
    parser = argparse.ArgumentParser(description="Export an RDK X5 friendly ONNX from a YOLO seg checkpoint.")
    parser.add_argument("--pt", required=True, help="Path to a trained .pt checkpoint.")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset.")
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Export image size. Use one value for square input or two values for h w.",
    )
    parser.add_argument("--output", help="Optional output ONNX path.")
    return parser


def normalize_imgsz(imgsz_values):
    if len(imgsz_values) == 1:
        return imgsz_values[0]
    if len(imgsz_values) == 2:
        return tuple(imgsz_values)
    raise SystemExit("--imgsz expects one value or two values: h w")


def export_checkpoint(pt_path, opset=11, imgsz=640, output=None):
    from ultralytics import YOLO

    model = YOLO(pt_path)
    patch_model_for_rdk(model.model.model)
    exported_path = Path(model.export(format="onnx", simplify=False, opset=opset, imgsz=imgsz))

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported_path, output_path)
        return output_path.resolve()

    return exported_path.resolve()


def main(argv=None):
    args = build_parser().parse_args(argv)
    exported_path = export_checkpoint(
        pt_path=args.pt,
        opset=args.opset,
        imgsz=normalize_imgsz(args.imgsz),
        output=args.output,
    )
    print(f"Exported ONNX: {exported_path}")


if __name__ == "__main__":
    main()
