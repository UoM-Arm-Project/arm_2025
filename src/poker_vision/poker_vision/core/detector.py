"""Model backends behind one interface: ``backend(image_bgr) -> [Detection]``.

``ultralytics`` loads ``.pt``, ``.onnx`` and TensorRT ``.engine`` files with the
same API, so switching CPU -> CUDA -> ONNX -> TensorRT is a YAML change:

    weights: models/cards.pt       device: cpu
    weights: models/cards.pt       device: cuda:0   half: true
    weights: models/cards.onnx     device: cuda:0   (needs onnxruntime-gpu)
    weights: models/cards.engine   device: cuda:0   (needs tensorrt; build on the target GPU)

``mock`` returns fixed detections from the YAML, so the whole ROS pipeline can
be exercised before the real weights exist.
"""
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    name: str                       # class name from the model (e.g. "AC", "chip_25")
    conf: float
    xyxy: np.ndarray                # pixel box x1, y1, x2, y2
    polygon: Optional[np.ndarray] = None   # Nx2 pixel outline (segmentation models)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.xyxy[0] + self.xyxy[2]) / 2.0, (self.xyxy[1] + self.xyxy[3]) / 2.0])


class UltralyticsBackend:
    def __init__(self, cfg: dict, base_dir: str = ""):
        from ultralytics import YOLO   # imported lazily: heavy, and optional for tests
        weights = os.path.expanduser(os.path.expandvars(str(cfg["weights"])))
        if base_dir and not os.path.isabs(weights):
            weights = os.path.join(base_dir, weights)
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Model weights not found: {weights}")
        self.weights = weights
        self.task = cfg.get("task")                 # detect | segment (required for .onnx/.engine)
        self.model = YOLO(weights, task=self.task) if self.task else YOLO(weights)
        self.device = str(cfg.get("device", "cpu"))
        self.half = bool(cfg.get("half", False)) and self.device != "cpu"
        self.imgsz = int(cfg.get("imgsz", 640))
        self.conf = float(cfg.get("conf", 0.5))
        self.iou = float(cfg.get("iou", 0.6))
        self.max_det = int(cfg.get("max_det", 100))
        self.last_latency_ms = 0.0

    def warmup(self, shape=(480, 640, 3)) -> None:
        self(np.zeros(shape, dtype=np.uint8))

    def __call__(self, image: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        res = self.model.predict(image, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                                 half=self.half, max_det=self.max_det, verbose=False)[0]
        out: List[Detection] = []
        names = res.names
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            polys = res.masks.xy if getattr(res, "masks", None) is not None else None
            for i in range(len(boxes)):
                poly = np.asarray(polys[i]) if polys is not None and len(polys[i]) else None
                out.append(Detection(str(names[classes[i]]), float(confs[i]), boxes[i], poly))
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return out


class MockBackend:
    """Fixed detections from YAML: [{name, xyxy: [x1,y1,x2,y2], conf, polygon: [[x,y],...]}]."""

    def __init__(self, cfg: dict, base_dir: str = ""):
        self.items = []
        for d in cfg.get("mock_detections", []):
            poly = np.asarray(d["polygon"], dtype=np.float64) if d.get("polygon") else None
            self.items.append(Detection(str(d["name"]), float(d.get("conf", 0.9)),
                                        np.asarray(d["xyxy"], dtype=np.float64), poly))
        self.last_latency_ms = 0.0

    def warmup(self, shape=None) -> None:
        pass

    def __call__(self, image: np.ndarray) -> List[Detection]:
        return list(self.items)


def make_backend(cfg: dict, base_dir: str = ""):
    kind = cfg.get("backend", "ultralytics")
    if kind == "ultralytics":
        return UltralyticsBackend(cfg, base_dir)
    if kind == "mock":
        return MockBackend(cfg, base_dir)
    raise ValueError(f"Unknown model backend {kind!r} (ultralytics | mock)")
