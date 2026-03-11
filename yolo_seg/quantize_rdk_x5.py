import argparse
import subprocess
from pathlib import Path


DEFAULT_DOCKER_IMAGE = "registry.d-robotics.cc/deliver/hub.hobot.cc/aitools/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8"
DEFAULT_MAPPER_SCRIPT = "rdk_model_zoo/samples/vision/ultralytics_yolo/x86/mapper.py"


def build_parser():
    parser = argparse.ArgumentParser(description="Run RDK X5 PTQ quantization and compile a .bin model.")
    parser.add_argument("--onnx", required=True, help="Path to input ONNX model.")
    parser.add_argument("--cal-images", required=True, help="Calibration image directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for mapper outputs.")
    parser.add_argument("--workspace", default=".", help="Workspace mounted into Docker.")
    parser.add_argument("--mapper-script", default=DEFAULT_MAPPER_SCRIPT, help="Mapper script path inside the workspace.")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE, help="Docker image for the Horizon toolchain.")
    parser.add_argument("--cal-sample-num", type=int, default=50, help="Calibration sample count.")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel jobs for mapper/compile.")
    parser.add_argument("--optimize-level", default="O3", help="Mapper optimize level.")
    return parser


def to_workspace_relative(path_value, workspace):
    path = Path(path_value)
    if not path.is_absolute():
        return path

    resolved_path = path.resolve()
    resolved_workspace = workspace.resolve()
    try:
        return resolved_path.relative_to(resolved_workspace)
    except ValueError as exc:
        raise SystemExit(f"path must stay under workspace: {resolved_path}") from exc


def run_quantize(args):
    workspace = Path(args.workspace).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = to_workspace_relative(args.onnx, workspace)
    cal_images = to_workspace_relative(args.cal_images, workspace)
    mapper_script = to_workspace_relative(args.mapper_script, workspace)
    output_dir_rel = to_workspace_relative(args.output_dir, workspace)

    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workspace}:/workspace",
        "-w",
        "/workspace",
        args.docker_image,
        "python3",
        str(Path("/workspace") / mapper_script),
        "--onnx",
        str(onnx_path),
        "--cal-images",
        str(cal_images),
        "--output-dir",
        str(output_dir_rel),
        "--cal-sample-num",
        str(args.cal_sample_num),
        "--jobs",
        str(args.jobs),
        "--optimize-level",
        args.optimize_level,
    ]

    print("Running:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_quantize(args)


if __name__ == "__main__":
    main()
