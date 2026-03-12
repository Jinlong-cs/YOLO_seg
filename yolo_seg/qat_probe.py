from __future__ import annotations

import json
import traceback
import types
from pathlib import Path

import torch
import torch.nn as nn

from yolo_seg.ultralytics_rdk import patch_model_for_rdk


class HorizonQATReadyYOLOSeg(nn.Module):
    def __init__(self, core_model, quant_stub_cls, dequant_stub_cls):
        super().__init__()
        self.quant = quant_stub_cls(scale=1 / 128)
        self.model = core_model
        self.dequant = dequant_stub_cls()

    def forward(self, x):
        x = self.quant(x)
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            outputs = list(outputs)
        if isinstance(outputs, list):
            return [self.dequant(out) for out in outputs]
        return self.dequant(outputs)


def c2f_forward_traceable(self, x):
    y0, y1 = self.cv1(x).chunk(2, 1)
    outputs = [y0, y1]
    last = y1
    for module in self.m:
        last = module(last)
        outputs.append(last)
    return self.cv2(torch.cat(outputs, 1))


def sppf_forward_traceable(self, x):
    y0 = self.cv1(x)
    y1 = self.m(y0)
    y2 = self.m(y1)
    y3 = self.m(y2)
    y = self.cv2(torch.cat((y0, y1, y2, y3), 1))
    return y + x if getattr(self, "add", False) else y


def patch_model_for_qat_trace(model):
    from ultralytics.nn.modules.block import C2f, C3k2, SPPF

    for child in model.children():
        if isinstance(child, (C2f, C3k2)):
            child.forward = types.MethodType(c2f_forward_traceable, child)
        elif isinstance(child, SPPF):
            child.forward = types.MethodType(sppf_forward_traceable, child)
        patch_model_for_qat_trace(child)


def march_from_name(name, march_enum):
    if hasattr(march_enum, name):
        return getattr(march_enum, name)
    if hasattr(march_enum, name.upper()):
        return getattr(march_enum, name.upper())
    raise SystemExit(f"Unsupported march: {name}")


def build_example_input(imgsz, device):
    height, width = imgsz
    return torch.randn(1, 3, height, width, device=device)


def run_qat_probe(pt_path, imgsz, output_dir, march="BAYES_E", device="cpu", input_source="ddr", compile_opt=0):
    from ultralytics import YOLO
    from horizon_plugin_pytorch.march import March, set_march
    from horizon_plugin_pytorch.quantization import (
        QuantStub,
        check_model,
        compile_model,
        convert_fx,
        perf_model,
        prepare_qat_fx,
        visualize_model,
    )
    from horizon_plugin_pytorch.quantization.qconfig import default_qat_8bit_fake_quant_qconfig
    from torch.quantization import DeQuantStub

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "pt_path": str(Path(pt_path).resolve()),
        "imgsz": list(imgsz),
        "march": march,
        "device": device,
        "input_source": input_source,
        "compile_opt": compile_opt,
        "success": False,
    }

    try:
        torch_device = torch.device(device)
        set_march(march_from_name(march, March))

        yolo = YOLO(pt_path)
        patch_model_for_rdk(yolo.model.model)
        patch_model_for_qat_trace(yolo.model.model)
        core_model = yolo.model.model
        wrapped = HorizonQATReadyYOLOSeg(core_model, QuantStub, DeQuantStub).to(torch_device).eval()

        example_input = build_example_input(imgsz, torch_device)
        qat_model = prepare_qat_fx(wrapped, {"": default_qat_8bit_fake_quant_qconfig}).to(torch_device).eval()
        quantized_model = convert_fx(qat_model).to(torch_device).eval()

        script_model = torch.jit.trace(quantized_model.cpu(), example_input.cpu(), strict=False)
        int_model_path = output_dir / "int_model.pt"
        torch.jit.save(script_model, int_model_path)

        check_model(script_model.cpu(), [example_input.cpu()], advice=1)

        hbm_path = output_dir / "model.hbm"
        compile_model(
            script_model,
            [example_input.cpu()],
            hbm=str(hbm_path),
            input_source=input_source,
            opt=compile_opt,
        )
        perf_model(
            script_model,
            [example_input.cpu()],
            out_dir=str(output_dir / "perf_out"),
            input_source=input_source,
            opt=compile_opt,
            layer_details=True,
        )
        visualize_model(
            script_model,
            [example_input.cpu()],
            save_path=str(output_dir / "model.svg"),
            show=False,
        )

        report["success"] = True
        report["artifacts"] = {
            "int_model_pt": str(int_model_path),
            "model_hbm": str(hbm_path),
            "perf_dir": str(output_dir / "perf_out"),
            "model_svg": str(output_dir / "model.svg"),
        }
    except Exception as exc:
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()

    report_path = output_dir / "qat_probe_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["success"]:
        raise SystemExit(1)
    return report_path
