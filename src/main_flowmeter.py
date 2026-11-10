"""Run model inference on CICFlowMeter-style flow CSVs (pcap -> flows -> model)."""

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
import requests

from .config import load_config, resolve_artifacts
from .features.preprocess import load_preprocessor, transform


def _latest_csv_in_dir(directory: Path) -> Optional[Path]:
    csvs = sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    return csvs[-1] if csvs else None


def run_cic_flowmeter(cic_cmd: str, pcap_path: Path, out_dir: Path) -> Path:
    """Run external CICFlowMeter command and return the newest CSV in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = cic_cmd.format(pcap=str(pcap_path), out=str(out_dir))
    print(f"Running CICFlowMeter cmd: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    # Give filesystem a moment to flush output (especially on Windows).
    time.sleep(0.25)
    flows_csv = _latest_csv_in_dir(out_dir)
    if flows_csv is None:
        raise FileNotFoundError(f"No CSV produced under: {out_dir}")
    return flows_csv


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _maybe_get_col(df: pd.DataFrame, target: str) -> Optional[str]:
    target_l = target.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == target_l:
            return c
    return None


def _build_flow_id(row: pd.Series, fallback_idx: int) -> str:
    # Prefer common CICFlowMeter fields if available.
    candidates = [
        ("Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"),
        ("Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol"),
        ("Dst Port", "Protocol", "Timestamp"),
        ("Dst Port", "Protocol"),
        ("Timestamp",),
    ]
    for fields in candidates:
        ok = True
        parts = []
        for f in fields:
            if f not in row.index:
                ok = False
                break
            parts.append(str(row[f]))
        if ok and parts:
            return "|".join(parts)
    return f"flow_{fallback_idx}"


def _iter_flow_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(path, chunksize=chunksize, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Use artifacts from a specific run (e.g., rf_run_03)",
    )

    # Input options.
    parser.add_argument(
        "--flows_csv",
        default=None,
        help="Path to CICFlowMeter flow CSV to run inference on",
    )
    parser.add_argument(
        "--pcap",
        default=None,
        help="Optional pcap path (if you want this script to run CICFlowMeter)",
    )
    parser.add_argument(
        "--cic_cmd",
        default=None,
        help="Command template to run CICFlowMeter. Use {pcap} and {out} placeholders.",
    )
    parser.add_argument(
        "--cic_out_dir",
        default="data/processed/cicflowmeter_out",
        help="Output dir for CICFlowMeter",
    )

    # Output controls.
    parser.add_argument("--min_confidence", type=float, default=0.0)
    parser.add_argument("--only_attacks", action="store_true")
    parser.add_argument(
        "--dashboard", action="store_true", help="Post events to Flask dashboard API"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable per-flow console output"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max rows to process (0 means unlimited)"
    )
    args = parser.parse_args()

    cfg, root = load_config(args.config)

    model_path, preproc_path = resolve_artifacts(cfg, root, args.run_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preproc_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preproc_path}")

    model = joblib.load(model_path)
    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
    preprocessor = load_preprocessor(str(preproc_path))
    feature_order: List[str] = list(preprocessor["feature_order"])

    flows_csv: Optional[Path] = Path(args.flows_csv) if args.flows_csv else None
    if flows_csv is None:
        if not args.pcap or not args.cic_cmd:
            raise ValueError("Provide --flows_csv OR (--pcap and --cic_cmd).")
        pcap_path = Path(args.pcap)
        if not pcap_path.exists():
            raise FileNotFoundError(f"Pcap not found: {pcap_path}")
        out_dir = root / args.cic_out_dir
        flows_csv = run_cic_flowmeter(args.cic_cmd, pcap_path, out_dir)

    if not flows_csv.exists():
        raise FileNotFoundError(f"Flow CSV not found: {flows_csv}")

    dashboard_url = None
    if args.dashboard:
        host = cfg["dashboard"]["host"]
        port = int(cfg["dashboard"]["port"])
        dashboard_url = f"http://{host}:{port}/api/events"
        try:
            requests.get(f"http://{host}:{port}/api/stats", timeout=3)
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Dashboard backend is unreachable at http://{host}:{port}. "
                "Start it first with: python -m src.main_serve"
            ) from exc

    chunksize = int(cfg["data"].get("chunksize", 200000))
    min_conf = float(args.min_confidence)
    only_attacks = bool(args.only_attacks)
    limit = int(args.limit)

    print(f"Flow CSV: {flows_csv}")
    print(f"Model: {model_path}")
    print(f"Preprocessor: {preproc_path}")

    seen = 0
    emitted = 0
    dropped_nan = 0

    for chunk in _iter_flow_chunks(flows_csv, chunksize=chunksize):
        chunk = _normalize_columns(chunk)

        missing = [f for f in feature_order if f not in chunk.columns]
        if missing:
            raise ValueError(f"Missing required features in flow CSV: {missing}")

        for feat in feature_order:
            chunk[feat] = pd.to_numeric(chunk[feat], errors="coerce")
        finite_mask = np.isfinite(chunk[feature_order].values).all(axis=1)
        dropped_nan += int((~finite_mask).sum())
        chunk = chunk[finite_mask].copy()

        if chunk.empty:
            continue

        X = transform(chunk, preprocessor)
        probs = model.predict_proba(X)
        classes = [str(c) for c in model.classes_]
        best_idx = np.argmax(probs, axis=1)
        preds = np.asarray([classes[int(i)] for i in best_idx], dtype=object)
        confs = probs[np.arange(len(best_idx)), best_idx].astype(float)

        ts_col = _maybe_get_col(chunk, "Timestamp")
        events: List[Dict[str, object]] = []
        for i in range(len(chunk)):
            seen += 1
            if limit > 0 and seen > limit:
                break

            pred = str(preds[i])
            conf = float(confs[i])
            if only_attacks and pred == "Normal":
                continue
            if conf < min_conf:
                continue

            row = chunk.iloc[i]
            ts = str(row[ts_col]) if ts_col else datetime.now(timezone.utc).isoformat()
            feature_values = {f: float(row[f]) for f in feature_order}
            top_features = dict(
                sorted(
                    feature_values.items(), key=lambda item: abs(item[1]), reverse=True
                )[:5]
            )

            event = {
                "ts": ts,
                "predicted_class": pred,
                "confidence": conf,
                "flow_id": _build_flow_id(row, fallback_idx=seen - 1),
                "top_feature_values": top_features,
                "scaled_feature_preview": X[i][:5].tolist(),
            }
            events.append(event)
            emitted += 1

            if not args.quiet:
                print(f"flow={event['flow_id']} pred={pred} conf={conf:.4f}")

        if dashboard_url and events:
            try:
                requests.post(dashboard_url, json=events, timeout=10)
            except requests.RequestException as exc:
                print(f"Failed to post events: {exc}")

        if limit > 0 and seen >= limit:
            break

    print(f"Done. rows_seen={seen} events_emitted={emitted} dropped_nan={dropped_nan}")


if __name__ == "__main__":
    main()
