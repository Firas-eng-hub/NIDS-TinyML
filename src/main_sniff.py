"""Sniff live/offline traffic with Scapy and run flow-based model inference."""

import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from scapy.all import sniff

from .config import load_config, resolve_artifacts
from .features.preprocess import load_preprocessor, transform
from .flows import FlowStats, FlowTable, _build_offline_lfilter


def build_event(
    flow: FlowStats,
    feature_values: Dict[str, float],
    scaled_row: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> Dict[str, object]:
    top_features = dict(
        sorted(feature_values.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    )
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "flow_id": flow.flow_id(),
        "top_feature_values": top_features,
        "scaled_feature_preview": scaled_row[:5].tolist(),
    }


def infer_flow(
    flow: FlowStats,
    model,
    preprocessor: Dict[str, object],
    feature_order: List[str],
) -> Tuple[str, float, Dict[str, float], np.ndarray]:
    feature_values = flow.to_feature_dict()
    row_df = pd.DataFrame([{name: feature_values[name] for name in feature_order}])
    scaled = transform(row_df, preprocessor)
    probs = model.predict_proba(scaled)[0]
    class_names = [str(c) for c in model.classes_]
    best_idx = int(np.argmax(probs))
    predicted = class_names[best_idx]
    confidence = float(probs[best_idx])
    return predicted, confidence, feature_values, scaled[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Use artifacts from a specific run (e.g., rf_run_03)",
    )
    parser.add_argument(
        "--iface", default=None, help="Network interface for live sniffing"
    )
    parser.add_argument(
        "--pcap",
        default=None,
        help="Read packets from a pcap file instead of live interface",
    )
    parser.add_argument("--bpf", default="ip", help="BPF filter (e.g., 'tcp or udp')")
    parser.add_argument(
        "--packet_count",
        type=int,
        default=0,
        help="Stop after N packets (0 means unlimited)",
    )
    parser.add_argument(
        "--sniff_timeout",
        type=float,
        default=0.0,
        help="Sniff duration in seconds (0 means none)",
    )
    parser.add_argument(
        "--flow_timeout", type=float, default=2.0, help="Flow idle timeout in seconds"
    )
    parser.add_argument(
        "--max_flow_duration",
        type=float,
        default=0.0,
        help="If >0, force-rotate flows after this many seconds (active timeout).",
    )
    parser.add_argument(
        "--min_packets",
        type=int,
        default=2,
        help="Minimum packets before flow inference",
    )
    parser.add_argument(
        "--active_infer_interval",
        type=float,
        default=0.0,
        help="If >0, also run inference periodically on active flows every N seconds.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Drop predictions below this confidence",
    )
    parser.add_argument(
        "--only_attacks", action="store_true", help="Skip Normal predictions"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Post events to Flask dashboard API"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable per-flow console output"
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
        except ValueError:
            pass

    preprocessor = load_preprocessor(str(preproc_path))
    feature_order = list(preprocessor["feature_order"])

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

    if float(args.flow_timeout) < 0:
        raise ValueError("--flow_timeout must be >= 0")
    if float(args.max_flow_duration) < 0:
        raise ValueError("--max_flow_duration must be >= 0")
    if float(args.active_infer_interval) < 0:
        raise ValueError("--active_infer_interval must be >= 0")

    flow_table = FlowTable(
        timeout_seconds=float(args.flow_timeout),
        max_duration_seconds=float(args.max_flow_duration),
    )
    active_infer_interval = float(args.active_infer_interval)

    counters = {
        "packets_seen": 0,
        "flows_inferred": 0,
        "events_emitted": 0,
    }

    def process_flows(flows: List[FlowStats], mark_emit_ts: bool = False) -> None:
        events: List[Dict[str, object]] = []
        for flow in flows:
            if flow.total_packets() < int(args.min_packets):
                continue

            predicted, confidence, feature_values, scaled_row = infer_flow(
                flow=flow,
                model=model,
                preprocessor=preprocessor,
                feature_order=feature_order,
            )
            counters["flows_inferred"] += 1
            if mark_emit_ts:
                flow.last_emit_ts = float(flow.last_ts)

            if args.only_attacks and predicted == "Normal":
                continue
            if confidence < float(args.min_confidence):
                continue

            event = build_event(
                flow=flow,
                feature_values=feature_values,
                scaled_row=scaled_row,
                predicted_class=predicted,
                confidence=confidence,
            )
            events.append(event)
            counters["events_emitted"] += 1

            if not args.quiet:
                print(
                    f"flow={event['flow_id']} pred={predicted} "
                    f"conf={confidence:.4f} duration_us={feature_values['Flow Duration']:.0f} "
                    f"pkts={int(feature_values['Tot Fwd Pkts'] + feature_values['Tot Bwd Pkts'])}"
                )

        if dashboard_url and events:
            try:
                requests.post(dashboard_url, json=events, timeout=5)
            except requests.RequestException as exc:
                print(f"Failed to post events: {exc}")

    def on_packet(packet) -> None:
        counters["packets_seen"] += 1
        updated_flow, expired = flow_table.update_with_packet(packet)
        if expired:
            process_flows(expired)

        if (
            active_infer_interval > 0.0
            and updated_flow is not None
            and updated_flow.total_packets() >= int(args.min_packets)
        ):
            now_ts = float(updated_flow.last_ts)
            last_emit = updated_flow.last_emit_ts
            if (
                last_emit is None
                or (now_ts - float(last_emit)) >= active_infer_interval
            ):
                process_flows([updated_flow], mark_emit_ts=True)

    sniff_kwargs = {"prn": on_packet, "store": False}
    if args.packet_count and args.packet_count > 0:
        sniff_kwargs["count"] = int(args.packet_count)
    if args.sniff_timeout and args.sniff_timeout > 0:
        sniff_kwargs["timeout"] = float(args.sniff_timeout)
    if args.pcap:
        sniff_kwargs["offline"] = str(args.pcap)
        lfilter = _build_offline_lfilter(args.bpf)
        if lfilter is not None:
            sniff_kwargs["lfilter"] = lfilter
        else:
            if args.bpf and str(args.bpf).strip().lower() not in ("", "ip"):
                print(
                    "Note: Offline PCAP BPF filtering requires tcpdump. "
                    "tcpdump is not available; processing all packets (best-effort)."
                )
    elif args.iface:
        sniff_kwargs["iface"] = args.iface
        sniff_kwargs["filter"] = args.bpf

    source_desc = (
        f"pcap={args.pcap}"
        if args.pcap
        else (f"iface={args.iface}" if args.iface else "default iface")
    )
    print(f"Sniffing source: {source_desc}")
    print(f"Model: {model_path}")
    print(f"Preprocessor: {preproc_path}")
    print(
        "Flow idle timeout: "
        f"{float(args.flow_timeout)}s, "
        f"max_flow_duration: {float(args.max_flow_duration)}s, "
        f"active_infer_interval: {active_infer_interval}s, "
        f"min_packets: {int(args.min_packets)}"
    )

    try:
        sniff(**sniff_kwargs)
    except KeyboardInterrupt:
        print("Interrupted by user, flushing remaining flows...")

    process_flows(flow_table.flush_all())

    print(
        "Done. "
        f"packets_seen={counters['packets_seen']} "
        f"flows_inferred={counters['flows_inferred']} "
        f"events_emitted={counters['events_emitted']}"
    )


if __name__ == "__main__":
    main()
