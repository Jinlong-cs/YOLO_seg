# YOLO_seg

`YOLO_seg` is the model-side workflow for an Ultralytics YOLO segmentation project.

It is intentionally small and only keeps the parts that are useful for training and model conversion:

- train a YOLO segmentation model
- export an RDK X5 friendly ONNX
- run PTQ quantization and compile a `.bin` model in Docker

It does **not** include:

- data annotation tools
- LabelMe to YOLO dataset conversion
- pseudo-label generation
- board-side deployment scripts
- board-side comparison or visualization scripts

The goal is simple:

- keep the current workspace usable as-is
- prepare a clean subdirectory that can later become an independent GitHub repo

## What This Directory Is For

Use `YOLO_seg` when you want to work on the model itself:

- model training
- model export
- model quantization / compilation

Do not use `YOLO_seg` for:

- annotation
- dataset curation
- pushing models to the board
- ROS2 runtime integration

Those parts still belong to the current main workspace.

## Layout

```text
YOLO_seg/
  .gitignore
  pyproject.toml
  README.md
  requirements.txt
  scripts/
    train_yolo_seg.py
    export_rdk_onnx.py
    quantize_rdk_x5.py
  yolo_seg/
    __init__.py
    train.py
    ultralytics_rdk.py
    export_rdk_onnx.py
    quantize_rdk_x5.py
```

Meaning of each part:

- `yolo_seg/`
  - reusable Python package code
- `scripts/`
  - thin CLI entry points
- `README.md`
  - usage notes and workflow
- `pyproject.toml`
  - packaging metadata for a future standalone repo

## Design Notes

This directory is organized to make later extraction easier.

A few deliberate choices:

- package code and CLI entry points are separated
- training / export / PTQ are split into small files
- imports are kept late where possible, so help messages work even if heavy packages are not installed
- there is no board-specific runtime code here

The current implementation is still pragmatic rather than polished. The goal right now is portability and low coupling with the rest of the workspace.

## Environment

### Python

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Current Python dependencies are intentionally minimal:

- `ultralytics`
- `torch`
- `onnx`

### PTQ / Docker

For PTQ you also need:

- Docker
- an RDK X5 OpenExplore image
- a Horizon mapper script, for example:
  `rdk_model_zoo/samples/vision/ultralytics_yolo/x86/mapper.py`

## CLI Entry Points

You can run the package in two ways:

### Option 1: run the local scripts

```bash
python YOLO_seg/scripts/train_yolo_seg.py -h
python YOLO_seg/scripts/export_rdk_onnx.py -h
python YOLO_seg/scripts/quantize_rdk_x5.py -h
```

### Option 2: install as a package later

After packaging or `pip install -e .`, the declared console entry points are:

```bash
yolo-seg-train
yolo-seg-export-rdk-onnx
yolo-seg-quantize-rdk-x5
```

## Usage In The Current Workspace

Right now this directory lives inside a larger workspace, so dataset paths and output paths may still point outside `YOLO_seg/`.

That is acceptable for now.

Typical current usage from the workspace root:

### 1. Train

```bash
python YOLO_seg/scripts/train_yolo_seg.py \
  --data data_loop/dataset_full/data.yaml \
  --model data_loop/yolo11n-seg.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --patience 50 \
  --name full_dataset
```

Default outputs follow Ultralytics behavior, for example:

- `runs/seg/<run_name>/weights/best.pt`
- `runs/seg/<run_name>/weights/last.pt`

In the current workspace these are typically under:

- `data_loop/runs/...`

### 2. Export ONNX For RDK X5

Square input:

```bash
python YOLO_seg/scripts/export_rdk_onnx.py \
  --pt data_loop/runs/segment/runs/seg/full_dataset/weights/best.pt \
  --opset 11 \
  --imgsz 640 \
  --output data_loop/runs/segment/runs/seg/full_dataset/weights/best_640x640.onnx
```

Native rectangular input:

```bash
python YOLO_seg/scripts/export_rdk_onnx.py \
  --pt data_loop/runs/segment/runs/seg/full_dataset/weights/best.pt \
  --opset 11 \
  --imgsz 352 640 \
  --output data_loop/runs/segment/runs/seg/full_dataset/weights/best_352x640.onnx
```

### 3. PTQ Quantization + Compile

If you run PTQ from the current workspace, the Docker mount root usually should be the workspace root rather than `YOLO_seg/`.

Example:

```bash
python YOLO_seg/scripts/quantize_rdk_x5.py \
  --workspace . \
  --onnx data_loop/runs/segment/runs/seg/full_dataset/weights/best_352x640.onnx \
  --cal-images data_loop/data_full_labeled \
  --output-dir data_loop/artifacts/rdk_x5_352x640 \
  --mapper-script data_loop/rdk_model_zoo/samples/vision/ultralytics_yolo/x86/mapper.py
```

This is important because:

- the quantization script mounts `--workspace` into Docker as `/workspace`
- all paths passed to the mapper must stay under that mounted root
- in this workspace, the data-loop side now lives under `data_loop/`

## Usage After Standalone Extraction

After `YOLO_seg` becomes its own repo, the intended structure is:

- datasets live inside that repo or are referenced by explicit paths
- model outputs live inside that repo
- the Horizon mapper script is vendored or documented as an external dependency

At that point the commands should look more self-contained, for example:

```bash
python scripts/train_yolo_seg.py \
  --data data/my_dataset.yaml \
  --model yolo11n-seg.pt \
  --name baseline
```

and:

```bash
python scripts/quantize_rdk_x5.py \
  --workspace . \
  --onnx runs/seg/baseline/weights/best.onnx \
  --cal-images data/calibration \
  --output-dir artifacts/rdk_x5
```

## File-Level Notes

### `yolo_seg/train.py`

Responsible for:

- wrapping Ultralytics training arguments
- returning a compact report with `run_dir`, `best.pt`, and `last.pt`

### `yolo_seg/export_rdk_onnx.py`

Responsible for:

- loading a trained checkpoint
- applying an Ultralytics patch for RDK-compatible export
- exporting raw segmentation outputs to ONNX
- supporting both square and rectangular image sizes

### `yolo_seg/ultralytics_rdk.py`

Responsible for:

- patching Ultralytics module forwards so the exported graph exposes raw outputs compatible with the RDK toolchain

### `yolo_seg/quantize_rdk_x5.py`

Responsible for:

- launching the Horizon mapper inside Docker
- mounting a chosen workspace root
- converting user paths to paths valid inside the Docker workspace

## Current Limitations

- dataset conversion is still outside this directory
- stage reporting is still outside this directory
- runtime validation on x86 / board is still outside this directory
- PTQ still depends on an external mapper script path

These are fine for now. The current goal is to isolate the model lifecycle first.

## What To Move Later

When this becomes a standalone repo, the next reasonable moves are:

- move stage reporting into `YOLO_seg`
- decide whether dataset conversion should also move in
- decide whether to vendor the required mapper helper or keep it external
- add a small `examples/` folder with one full train-export-quantize workflow

## Summary

`YOLO_seg` is the clean model workflow boundary:

- train
- export
- PTQ compile

Everything else stays in the current workspace for now.
