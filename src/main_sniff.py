"""Sniff live/offline traffic with Scapy and run flow-based model inference."""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import yaml
from scapy.all import IP, TCP, UDP, sniff

from .features.preprocess import load_preprocessor, transform


def _build_offline_lfilter(bpf: str):
    """Best-effort Python filter for offline pcaps (avoids tcpdump dependency on Windows).

    Only supports a few common patterns used in this project.
    """
    if not bpf:
        return None

    expr = str(bpf).strip().lower()
    if expr == "ip":
        return lambda p: IP in p
    if expr == "tcp":
        return lambda p: TCP in p
    if expr == "udp":
        return lambda p: UDP in p
    if expr in ("tcp or udp", "udp or tcp"):
        return lambda p: (TCP in p) or (UDP in p)

    return None


@dataclass
class FlowStats:
    """Aggregated flow stats for a single 5-tuple direction."""

    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    start_ts: float
    last_ts: float
    # Used to rate-limit periodic inference on active flows (see --active_infer_interval).
    last_emit_ts: Optional[float] = None
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_len_sum: int = 0
    bwd_len_sum: int = 0
    fwd_len_max: int = 0
    bwd_len_max: int = 0
    pkt_len_min: int = 2**31 - 1
    pkt_len_max: int = 0
    syn_cnt: int = 0
    rst_cnt: int = 0
    ack_cnt: int = 0
    psh_cnt: int = 0

    def update(self, direction: str, pkt_len: int, flags: int, ts: float) -> None:
        """Update counters with one packet."""
        self.last_ts = ts

        if direction == "fwd":
            self.fwd_pkts += 1
            self.fwd_len_sum += pkt_len
            if pkt_len > self.fwd_len_max:
                self.fwd_len_max = pkt_len
        else:
            self.bwd_pkts += 1
            self.bwd_len_sum += pkt_len
            if pkt_len > self.bwd_len_max:
                self.bwd_len_max = pkt_len

        if pkt_len < self.pkt_len_min:
            self.pkt_len_min = pkt_len
        if pkt_len > self.pkt_len_max:
            self.pkt_len_max = pkt_len

        if flags & 0x02:
            self.syn_cnt += 1
        if flags & 0x04:
            self.rst_cnt += 1
        if flags & 0x10:
            self.ack_cnt += 1
        if flags & 0x08:
            self.psh_cnt += 1

    def total_packets(self) -> int:
        return self.fwd_pkts + self.bwd_pkts

    def duration_seconds(self) -> float:
        return max(0.0, self.last_ts - self.start_ts)

    def duration_microseconds(self) -> float:
        return float(self.duration_seconds() * 1_000_000.0)

    def flow_id(self) -> str:
        start_ms = int(self.start_ts * 1000)
        return f"{self.src}:{self.sport}->{self.dst}:{self.dport}/{self.proto}@{start_ms}"

    def to_feature_dict(self) -> Dict[str, float]:
        duration_s = self.duration_seconds()
        flow_pkts_per_s = 0.0 if duration_s <= 1e-9 else self.total_packets() / duration_s
        down_up_ratio = 0.0 if self.fwd_pkts == 0 else self.bwd_pkts / float(self.fwd_pkts)
        pkt_len_min = 0.0 if self.pkt_len_min == 2**31 - 1 else float(self.pkt_len_min)

        return {
            "Flow Duration": float(self.duration_microseconds()),
            "Tot Fwd Pkts": float(self.fwd_pkts),
            "Tot Bwd Pkts": float(self.bwd_pkts),
            "TotLen Fwd Pkts": float(self.fwd_len_sum),
            "TotLen Bwd Pkts": float(self.bwd_len_sum),
            "Fwd Pkt Len Max": float(self.fwd_len_max),
            "Bwd Pkt Len Max": float(self.bwd_len_max),
            "Pkt Len Min": pkt_len_min,
            "Pkt Len Max": float(self.pkt_len_max),
            "SYN Flag Cnt": float(self.syn_cnt),
            "RST Flag Cnt": float(self.rst_cnt),
            "ACK Flag Cnt": float(self.ack_cnt),
            "PSH Flag Cnt": float(self.psh_cnt),
            "Down/Up Ratio": float(down_up_ratio),
            "Flow Pkts/s": float(flow_pkts_per_s),
        }


class FlowTable:
    """Track active flows and emit expired ones."""

    def __init__(self, timeout_seconds: float, max_duration_seconds: float = 0.0):
        self.timeout_seconds = timeout_seconds
        self.max_duration_seconds = float(max_duration_seconds)
        self.table: Dict[Tuple[str, str, int, int, int], FlowStats] = {}

    @staticmethod
    def _extract_packet(packet) -> Optional[Tuple[Tuple[str, str, int, int, int], int, int, float]]:
        if IP not in packet:
            return None

        ip = packet[IP]
        src = str(ip.src)
        dst = str(ip.dst)
        proto = int(ip.proto)

        sport = 0
        dport = 0
        flags = 0
        if TCP in packet:
            sport = int(packet[TCP].sport)
            dport = int(packet[TCP].dport)
            flags = int(packet[TCP].flags)
        elif UDP in packet:
            sport = int(packet[UDP].sport)
            dport = int(packet[UDP].dport)

        key = (src, dst, sport, dport, proto)
        pkt_len = int(len(packet))
        ts = float(packet.time)
        return key, pkt_len, flags, ts

    def update_with_packet(self, packet) -> Tuple[Optional[FlowStats], List[FlowStats]]:
        parsed = self._extract_packet(packet)
        if parsed is None:
            return None, []

        key, pkt_len, flags, ts = parsed
        rev_key = (key[1], key[0], key[3], key[2], key[4])

        expired: List[FlowStats] = []

        if key in self.table:
            canonical_key = key
            flow = self.table[canonical_key]
            direction = "fwd"
        elif rev_key in self.table:
            canonical_key = rev_key
            flow = self.table[canonical_key]
            direction = "bwd"
        else:
            canonical_key = key
            flow = FlowStats(
                src=key[0],
                dst=key[1],
                sport=key[2],
                dport=key[3],
                proto=key[4],
                start_ts=ts,
                last_ts=ts,
            )
            self.table[canonical_key] = flow
            direction = "fwd"

        # If configured, rotate long-lived flows (active timeout) so we don't wait
        # for idle gaps to emit an inference event.
        if (
            self.max_duration_seconds > 0.0
            and (ts - flow.start_ts) >= self.max_duration_seconds
            and canonical_key in self.table
        ):
            expired.append(self.table.pop(canonical_key))
            flow = FlowStats(
                src=canonical_key[0],
                dst=canonical_key[1],
                sport=canonical_key[2],
                dport=canonical_key[3],
                proto=canonical_key[4],
                start_ts=ts,
                last_ts=ts,
            )
            self.table[canonical_key] = flow

        flow.update(direction, pkt_len, flags, ts)
        expired.extend(self.pop_expired(ts))
        return flow, expired

    def pop_expired(self, now_ts: float) -> List[FlowStats]:
        expired_keys = [
            key for key, flow in self.table.items() if (now_ts - flow.last_ts) >= self.timeout_seconds
        ]
        expired: List[FlowStats] = []
        for key in expired_keys:
            expired.append(self.table.pop(key))
        return expired

    def flush_all(self) -> List[FlowStats]:
        all_flows = list(self.table.values())
        self.table.clear()
        return all_flows


def load_config(config_path: Optional[str] = None):
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, root


def resolve_artifacts(cfg: dict, root: Path, run_name: Optional[str]) -> Tuple[Path, Path]:
    models_dir = root / cfg["artifacts"]["models_dir"]
    scalers_dir = root / cfg["artifacts"]["scalers_dir"]
    if run_name:
        model_path = models_dir / run_name / "model.joblib"
        preproc_path = scalers_dir / run_name / "preprocessor.json"
    else:
        model_path = models_dir / "model.joblib"
        preproc_path = scalers_dir / "preprocessor.json"
    return model_path, preproc_path


def build_event(
    flow: FlowStats,
    feature_values: Dict[str, float],
    scaled_row: np.ndarray,
    predicted_class: str,
    confidence: float,
) -> Dict[str, object]:
    top_features = dict(sorted(feature_values.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
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
    parser.add_argument("--run_name", default=None, help="Use artifacts from a specific run (e.g., rf_run_03)")
    parser.add_argument("--iface", default=None, help="Network interface for live sniffing")
    parser.add_argument("--pcap", default=None, help="Read packets from a pcap file instead of live interface")
    parser.add_argument("--bpf", default="ip", help="BPF filter (e.g., 'tcp or udp')")
    parser.add_argument("--packet_count", type=int, default=0, help="Stop after N packets (0 means unlimited)")
    parser.add_argument("--sniff_timeout", type=float, default=0.0, help="Sniff duration in seconds (0 means none)")
    parser.add_argument("--flow_timeout", type=float, default=2.0, help="Flow idle timeout in seconds")
    parser.add_argument(
        "--max_flow_duration",
        type=float,
        default=0.0,
        help="If >0, force-rotate flows after this many seconds (active timeout).",
    )
    parser.add_argument("--min_packets", type=int, default=2, help="Minimum packets before flow inference")
    parser.add_argument(
        "--active_infer_interval",
        type=float,
        default=0.0,
        help="If >0, also run inference periodically on active flows every N seconds.",
    )
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Drop predictions below this confidence")
    parser.add_argument("--only_attacks", action="store_true", help="Skip Normal predictions")
    parser.add_argument("--dashboard", action="store_true", help="Post events to Flask dashboard API")
    parser.add_argument("--quiet", action="store_true", help="Disable per-flow console output")
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
            if last_emit is None or (now_ts - float(last_emit)) >= active_infer_interval:
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

    source_desc = f"pcap={args.pcap}" if args.pcap else (f"iface={args.iface}" if args.iface else "default iface")
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
