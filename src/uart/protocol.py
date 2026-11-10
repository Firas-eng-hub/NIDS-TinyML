"""ASCII UART protocol shared by replay and evaluation commands."""

import json
import math


def encode_row(row_id, feature_values):
    if not all(math.isfinite(float(v)) for v in feature_values):
        raise ValueError("UART features must be finite")
    return (
        ",".join([str(row_id)] + [f"{v:.9g}" for v in feature_values]) + "\n"
    ).encode("ascii")


def parse_response(line):
    line = line.strip()
    if not line:
        raise ValueError("Empty UART response")
    if line.startswith("{"):
        payload = json.loads(line)
        row_id = payload.get("row_id")
        pred_id = int(payload["pred_id"])
        confidence = float(payload.get("confidence", 0.0))
    else:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            row_id, pred_id, confidence = None, int(parts[0]), float(parts[1])
        elif len(parts) == 3:
            row_id, pred_id, confidence = int(parts[0]), int(parts[1]), float(parts[2])
        else:
            raise ValueError("Unsupported UART response")
    if confidence > 1.0:
        confidence /= 1_000_000.0
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid UART confidence")
    return None if row_id is None else int(row_id), pred_id, confidence
