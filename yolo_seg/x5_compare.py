from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


def load_config(config_path):
    config_path = Path(config_path).resolve()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["_config_path"] = config_path
    return data


def resolve_under_workspace(workspace_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(workspace_root) / path).resolve()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def expected_float_best_pt(config):
    float_cfg = config["float"]
    return resolve_under_workspace(
        config["workspace"]["root"],
        Path(float_cfg["project"]) / float_cfg["name"] / "weights" / "best.pt",
    )


def expected_ptq_onnx(config):
    ptq_cfg = config["ptq"]
    dims = ptq_cfg["export_imgsz"]
    height, width = dims
    return resolve_under_workspace(
        config["workspace"]["root"],
        f"data_loop/runs/segment/runs/seg/full_dataset/weights/best_{height}x{width}.onnx",
    )


def build_stage_commands(config):
    ws_root = Path(config["workspace"]["root"]).resolve()
    yolo_root = Path(config["workspace"]["yolo_seg_root"]).resolve()
    data_yaml = resolve_under_workspace(ws_root, config["dataset"]["data_yaml"])
    base_checkpoint = resolve_under_workspace(ws_root, config["model"]["base_checkpoint"])
    float_best = resolve_under_workspace(ws_root, config["model"]["float_best_pt"])

    float_cfg = config["float"]
    ptq_cfg = config["ptq"]
    qat_cfg = config["qat"]

    ptq_onnx = expected_ptq_onnx(config)
    ptq_output_dir = resolve_under_workspace(ws_root, ptq_cfg["output_dir"])
    qat_output_dir = resolve_under_workspace(ws_root, qat_cfg["output_dir"])

    return {
        "float_train": [
            "python3",
            str(yolo_root / "scripts" / "train_yolo_seg.py"),
            "--data",
            str(data_yaml),
            "--model",
            str(base_checkpoint),
            "--epochs",
            str(float_cfg["epochs"]),
            "--imgsz",
            str(float_cfg["imgsz"]),
            "--batch",
            str(float_cfg["batch"]),
            "--device",
            str(float_cfg["device"]),
            "--workers",
            str(float_cfg["workers"]),
            "--patience",
            str(float_cfg["patience"]),
            "--project",
            str(resolve_under_workspace(ws_root, float_cfg["project"])),
            "--name",
            str(float_cfg["name"]),
        ],
        "float_export_ptq_shape": [
            "python3",
            str(yolo_root / "scripts" / "export_rdk_onnx.py"),
            "--pt",
            str(float_best),
            "--opset",
            "11",
            "--imgsz",
            str(ptq_cfg["export_imgsz"][0]),
            str(ptq_cfg["export_imgsz"][1]),
            "--output",
            str(ptq_onnx),
        ],
        "ptq_compile": [
            "python3",
            str(yolo_root / "scripts" / "quantize_rdk_x5.py"),
            "--workspace",
            str(ws_root),
            "--onnx",
            str(ptq_onnx),
            "--data-yaml",
            str(data_yaml),
            "--cal-split",
            str(ptq_cfg["cal_split"]),
            "--output-dir",
            str(ptq_output_dir),
            "--docker-image",
            str(ptq_cfg["docker_image"]),
            "--cal-sample-num",
            str(ptq_cfg["cal_sample_num"]),
            "--cal-seed",
            str(ptq_cfg["cal_seed"]),
            "--preprocess",
            str(ptq_cfg["preprocess"]),
            "--jobs",
            str(ptq_cfg["jobs"]),
            "--optimize-level",
            str(ptq_cfg["optimize_level"]),
            "--quantized",
            str(ptq_cfg["quantized"]),
        ],
        "qat_env_setup": [
            "python3",
            str(yolo_root / "scripts" / "setup_horizon_qat_env.py"),
            "--venv-path",
            str(resolve_under_workspace(yolo_root, qat_cfg["venv_path"])),
            "--open-explorer-root",
            str(qat_cfg["open_explorer_root"]),
        ],
        "qat_probe": [
            "python3",
            str(yolo_root / "scripts" / "qat_probe_yolo.py"),
            "--pt",
            str(float_best),
            "--imgsz",
            str(qat_cfg["probe_imgsz"][0]),
            str(qat_cfg["probe_imgsz"][1]),
            "--march",
            str(qat_cfg["march"]),
            "--device",
            str(qat_cfg["device"]),
            "--input-source",
            str(qat_cfg["input_source"]),
            "--compile-opt",
            str(qat_cfg["compile_opt"]),
            "--output-dir",
            str(qat_output_dir),
        ],
    }


def manifest_dict(config):
    commands = build_stage_commands(config)
    ws_root = Path(config["workspace"]["root"]).resolve()
    output_root = ensure_dir(resolve_under_workspace(ws_root, config["experiment"]["output_root"]))
    return {
        "experiment": config["experiment"]["name"],
        "config": str(config["_config_path"]),
        "workspace_root": str(ws_root),
        "output_root": str(output_root),
        "stages": {name: {"command": command} for name, command in commands.items()},
    }


def write_manifest(config):
    manifest = manifest_dict(config)
    output_root = Path(manifest["output_root"])
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shell_path = output_root / "run_stages.sh"
    shell_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for name, info in manifest["stages"].items():
        shell_lines.append(f"# {name}")
        shell_lines.append(" ".join(info["command"]))
        shell_lines.append("")
    shell_path.write_text("\n".join(shell_lines), encoding="utf-8")
    shell_path.chmod(0o755)
    return manifest_path, shell_path, manifest


def run_stage(config, stage):
    commands = build_stage_commands(config)
    command = commands[stage]
    print("Running stage:", stage)
    print(" ".join(command))
    subprocess.run(command, check=True)
