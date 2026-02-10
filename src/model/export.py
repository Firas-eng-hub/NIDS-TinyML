"""Model export utilities."""

from typing import Dict

import json
import os
from datetime import datetime

import joblib


def export_model(model, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)


def export_metadata(metadata: Dict[str, object], metadata_path: str) -> None:
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    metadata = dict(metadata)
    metadata["created_at"] = datetime.utcnow().isoformat() + "Z"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
