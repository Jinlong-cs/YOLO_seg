# YOLO_seg

Minimal training/export/PTQ workflow for an Ultralytics YOLO segmentation model.

This directory is intended to be split into a standalone repository later.
It only keeps the model-side workflow:

- train a YOLO segmentation model
- export an RDK X5 friendly ONNX
- run PTQ quantization and compile a `.bin` model in Docker

It does not include:

- data annotation tools
- label conversion from LabelMe to YOLO format
- board-side deployment or runtime comparison scripts

## Layout

```text
YOLO_seg/
  pyproject.toml
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

## Environment

Python packages:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

System requirements for PTQ:

- Docker
- an RDK X5 OpenExplore image
- Horizon mapper script, for example:
  `rdk_model_zoo/samples/vision/ultralytics_yolo/x86/mapper.py`

## Usage

### 1. Train

```bash
python scripts/train_yolo_seg.py \
  --data dataset_full/data.yaml \
  --model yolo11n-seg.pt \
  --epochs 150 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --patience 50 \
  --name full_dataset
```

### 2. Export ONNX for RDK X5

Square input:

```bash
python scripts/export_rdk_onnx.py \
  --pt runs/segment/runs/seg/full_dataset/weights/best.pt \
  --opset 11 \
  --imgsz 640 \
  --output runs/segment/runs/seg/full_dataset/weights/best_640x640.onnx
```

Native rectangular input:

```bash
python scripts/export_rdk_onnx.py \
  --pt runs/segment/runs/seg/full_dataset/weights/best.pt \
  --opset 11 \
  --imgsz 352 640 \
  --output runs/segment/runs/seg/full_dataset/weights/best_352x640.onnx
```

### 3. PTQ Quantization + Compile

```bash
python scripts/quantize_rdk_x5.py \
  --onnx runs/segment/runs/seg/full_dataset/weights/best_352x640.onnx \
  --cal-images data_full_labeled \
  --output-dir artifacts/rdk_x5_352x640 \
  --mapper-script rdk_model_zoo/samples/vision/ultralytics_yolo/x86/mapper.py
```

## Notes

- The export path applies a small Ultralytics model patch so the segmentation head is exported as raw outputs compatible with the RDK toolchain.
- The quantization script assumes all user paths are under the chosen workspace root, which defaults to the current working directory.
- This directory intentionally keeps the workflow small so it can be open-sourced cleanly later.
