from __future__ import annotations

import json
import traceback
import types
from pathlib import Path

import torch
import torch.nn as nn

from yolo_seg.ultralytics_rdk import aattn_forward, attention_forward, patch_model_for_rdk


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
    return self.cv2(self.cat_ff.cat(tuple(outputs), dim=1))


def sppf_forward_traceable(self, x):
    y0 = self.cv1(x)
    y1 = self.m(y0)
    y2 = self.m(y1)
    y3 = self.m(y2)
    y = self.cv2(self.cat_ff.cat((y0, y1, y2, y3), dim=1))
    return y + x if getattr(self, "add", False) else y


def bottleneck_forward_qat(self, x):
    y = self.cv2(self.cv1(x))
    return self.skip_add.add(x, y) if self.add else y


def psablock_forward_qat(self, x):
    x = self.attn_add.add(x, self.attn(x)) if self.add else self.attn(x)
    x = self.ffn_add.add(x, self.ffn(x)) if self.add else self.ffn(x)
    return x


def c3_forward_qat(self, x):
    return self.cv3(self.cat_ff.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


def c2psa_forward_qat(self, x):
    a, b = self.cv1(x).split((self.c, self.c), dim=1)
    b = self.m(b)
    return self.cv2(self.cat_ff.cat((a, b), dim=1))


def concat_forward_qat(self, x):
    return self.cat_ff.cat(tuple(x), dim=self.d)


def attention_forward_qat(self, x):
    x = self.dequant_qat(x)
    x = attention_forward(self, x)
    return self.quant_qat(x)


def aattn_forward_qat(self, x):
    x = self.dequant_qat(x)
    x = aattn_forward(self, x)
    return self.quant_qat(x)


def patch_model_for_qat_trace(model):
    from horizon_plugin_pytorch.nn.quantized import FloatFunctional
    from ultralytics.nn.modules.block import C2f, C3k2, SPPF

    for child in model.children():
        if isinstance(child, (C2f, C3k2)):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(c2f_forward_traceable, child)
        elif isinstance(child, SPPF):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(sppf_forward_traceable, child)
        patch_model_for_qat_trace(child)


def patch_model_for_qat_eager(model):
    from horizon_plugin_pytorch.nn.quantized import FloatFunctional
    from horizon_plugin_pytorch.quantization import QuantStub
    from torch.quantization import DeQuantStub
    from ultralytics.nn.modules.block import AAttn, Attention, Bottleneck, C2PSA, C3, C3k, C2f, C3k2, PSABlock, SPPF
    from ultralytics.nn.modules.conv import Concat

    for child in model.children():
        if isinstance(child, Bottleneck) and child.add:
            if not hasattr(child, "skip_add"):
                child.skip_add = FloatFunctional()
            child.forward = types.MethodType(bottleneck_forward_qat, child)
        elif isinstance(child, PSABlock) and child.add:
            if not hasattr(child, "attn_add"):
                child.attn_add = FloatFunctional()
            if not hasattr(child, "ffn_add"):
                child.ffn_add = FloatFunctional()
            child.forward = types.MethodType(psablock_forward_qat, child)
        elif isinstance(child, (C2f, C3k2)):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(c2f_forward_traceable, child)
        elif isinstance(child, SPPF):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(sppf_forward_traceable, child)
        elif isinstance(child, (C3, C3k)):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(c3_forward_qat, child)
        elif isinstance(child, C2PSA):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(c2psa_forward_qat, child)
        elif isinstance(child, Concat):
            if not hasattr(child, "cat_ff"):
                child.cat_ff = FloatFunctional()
            child.forward = types.MethodType(concat_forward_qat, child)
        elif isinstance(child, Attention):
            if not hasattr(child, "dequant_qat"):
                child.dequant_qat = DeQuantStub()
            if not hasattr(child, "quant_qat"):
                child.quant_qat = QuantStub(scale=1 / 128)
            child.forward = types.MethodType(attention_forward_qat, child)
        elif isinstance(child, AAttn):
            if not hasattr(child, "dequant_qat"):
                child.dequant_qat = DeQuantStub()
            if not hasattr(child, "quant_qat"):
                child.quant_qat = QuantStub(scale=1 / 128)
            child.forward = types.MethodType(aattn_forward_qat, child)
        patch_model_for_qat_eager(child)


def march_from_name(name, march_enum):
    if hasattr(march_enum, name):
        return getattr(march_enum, name)
    if hasattr(march_enum, name.upper()):
        return getattr(march_enum, name.upper())
    raise SystemExit(f"Unsupported march: {name}")


def build_example_input(imgsz, device):
    height, width = imgsz
    return torch.randn(1, 3, height, width, device=device)


def try_fx_probe(core_model, example_input, torch_device):
    from horizon_plugin_pytorch.quantization import convert_fx, prepare_qat_fx
    from horizon_plugin_pytorch.quantization.qconfig import default_qat_8bit_fake_quant_qconfig
    from torch.quantization import DeQuantStub
    from horizon_plugin_pytorch.quantization import QuantStub

    wrapped = HorizonQATReadyYOLOSeg(core_model, QuantStub, DeQuantStub).to(torch_device).eval()
    qat_model = prepare_qat_fx(wrapped, {"": default_qat_8bit_fake_quant_qconfig}).to(torch_device).eval()
    quantized_model = convert_fx(qat_model).to(torch_device).eval()
    return quantized_model


def try_eager_probe(core_model, example_input, torch_device):
    from horizon_plugin_pytorch.quantization import convert, prepare_qat
    from horizon_plugin_pytorch.quantization.qconfig import default_qat_8bit_fake_quant_qconfig
    from torch.quantization import DeQuantStub
    from horizon_plugin_pytorch.quantization import QuantStub

    wrapped = HorizonQATReadyYOLOSeg(core_model, QuantStub, DeQuantStub).to(torch_device).eval()
    wrapped.qconfig = default_qat_8bit_fake_quant_qconfig
    for module in wrapped.modules():
        if hasattr(module, "qconfig") and getattr(module, "qconfig", None) is None:
            if module.__class__.__name__ == "DeQuantStub":
                continue
            module.qconfig = default_qat_8bit_fake_quant_qconfig

    qat_model = prepare_qat(
        wrapped,
        inplace=False,
        example_inputs=(example_input,),
        verbose=0,
    ).to(torch_device).eval()
    quantized_model = convert(qat_model, inplace=False).to(torch_device).eval()
    return quantized_model


def run_qat_probe(pt_path, imgsz, output_dir, march="BAYES_E", device="cpu", input_source="ddr", compile_opt=0):
    from ultralytics import YOLO
    from horizon_plugin_pytorch.march import March, set_march
    from horizon_plugin_pytorch.quantization import (
        check_model,
        compile_model,
        perf_model,
        visualize_model,
    )

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
        patch_model_for_qat_eager(yolo.model.model)
        core_model = yolo.model.model
        example_input = build_example_input(imgsz, torch_device)

        fx_error = None
        eager_error = None
        quantized_model = None
        probe_mode = None

        try:
            quantized_model = try_fx_probe(core_model, example_input, torch_device)
            probe_mode = "fx"
        except Exception as exc:
            fx_error = {"error": str(exc), "traceback": traceback.format_exc()}

        if quantized_model is None:
            try:
                quantized_model = try_eager_probe(core_model, example_input, torch_device)
                probe_mode = "eager"
            except Exception as exc:
                eager_error = {"error": str(exc), "traceback": traceback.format_exc()}

        if quantized_model is None:
            report["fx_probe"] = fx_error
            report["eager_probe"] = eager_error
            raise RuntimeError("Both fx and eager QAT probe paths failed")

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
        report["probe_mode"] = probe_mode
        report["fx_probe"] = {"status": "ok"} if fx_error is None else fx_error
        report["eager_probe"] = {"status": "ok"} if eager_error is None else eager_error
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
