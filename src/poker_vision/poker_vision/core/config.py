"""Load vision.yaml (one file configures both cameras, both models and the table frame)."""
import os

import yaml


def load_config(path: str) -> dict:
    path = os.path.expanduser(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_base_dir"] = os.path.dirname(os.path.realpath(path))
    return cfg
