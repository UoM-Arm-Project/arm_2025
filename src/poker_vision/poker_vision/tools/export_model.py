"""export_model: convert a YOLO .pt to ONNX or TensorRT for faster inference.

    ros2 run poker_vision export_model ~/poker_arm_ws/models/cards.pt --format onnx
    ros2 run poker_vision export_model ~/poker_arm_ws/models/cards.pt --format engine --half

Then point ``weights`` in vision.yaml at the new file and set ``task``
(detect / segment). A TensorRT engine is tied to the GPU + TensorRT version it
was built on: build it on the machine that will run it.
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "torchscript"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (GPU engines)")
    ap.add_argument("--device", default=None, help="0 for the first GPU (required for engine)")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--workspace", type=float, default=2.0, help="TensorRT workspace GB (keep small on a 4 GB GPU)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    kwargs = dict(format=a.format, imgsz=a.imgsz, half=a.half, dynamic=a.dynamic)
    if a.format == "engine":
        kwargs["device"] = a.device or 0
        kwargs["workspace"] = a.workspace
    elif a.device is not None:
        kwargs["device"] = a.device
    out = model.export(**kwargs)
    print(f"\nexported: {out}\nset in vision.yaml:  weights: {out}   task: {model.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
