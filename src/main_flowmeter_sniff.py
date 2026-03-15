"""Scapy-based live/offline flowmeter that exports 15 flow features to CSV."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from scapy.all import IP, TCP, UDP, sniff

from .features.feature_list import FEATURE_LIST


def _build_offline_lfilter(bpf: str):
    """Best-effort Python filter for offline pcaps (avoids tcpdump dependency on Windows)."""
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

    def update_with_packet(self, packet) -> List[FlowStats]:
        parsed = self._extract_packet(packet)
        if parsed is None:
            return []

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
        return expired

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


class FlowCsvWriter:
    """Append per-flow feature rows to CSV in real time."""

    def __init__(self, csv_path: Path, feature_order: List[str]) -> None:
        self.csv_path = csv_path
        self.feature_order = list(feature_order)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.csv_path.open("a", encoding="utf-8", newline="")
        base_fields = [
            "Timestamp",
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Dst Port",
            "Protocol",
        ]
        self.fieldnames = base_fields + self.feature_order
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def write_flow(self, flow: FlowStats) -> None:
        feature_values = flow.to_feature_dict()
        missing = [f for f in self.feature_order if f not in feature_values]
        if missing:
            raise ValueError(f"Missing feature values while writing CSV: {missing}")

        row = {
            "Timestamp": datetime.now(timezone.utc).isoformat(),
            "Flow ID": flow.flow_id(),
            "Src IP": flow.src,
            "Src Port": int(flow.sport),
            "Dst IP": flow.dst,
            "Dst Port": int(flow.dport),
            "Protocol": int(flow.proto),
        }
        for feat in self.feature_order:
            row[feat] = float(feature_values[feat])

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


def load_config(config_path: Optional[str] = None):
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
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
        help="If >0, force-rotate long-lived flows after this many seconds.",
    )
    parser.add_argument("--min_packets", type=int, default=2, help="Minimum packets before exporting a flow row")
    parser.add_argument(
        "--csv_out",
        default="data/processed/live_flows_15.csv",
        help="CSV file to append exported flows",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable per-flow console output")
    args = parser.parse_args()

    cfg, root = load_config(args.config)
    configured_feature_order = list(cfg["features"]["fixed_list"])
    if configured_feature_order != list(FEATURE_LIST):
        raise ValueError(
            "config.yaml features.fixed_list does not match src/features/feature_list.py. "
            "Keep a single canonical 15-feature order."
        )

    if float(args.flow_timeout) < 0:
        raise ValueError("--flow_timeout must be >= 0")
    if float(args.max_flow_duration) < 0:
        raise ValueError("--max_flow_duration must be >= 0")

    csv_path = Path(args.csv_out)
    if not csv_path.is_absolute():
        csv_path = root / csv_path

    flow_table = FlowTable(
        timeout_seconds=float(args.flow_timeout),
        max_duration_seconds=float(args.max_flow_duration),
    )
    csv_writer = FlowCsvWriter(csv_path=csv_path, feature_order=configured_feature_order)

    counters = {
        "packets_seen": 0,
        "flows_written": 0,
    }

    def write_flows(flows: List[FlowStats]) -> None:
        for flow in flows:
            if flow.total_packets() < int(args.min_packets):
                continue
            csv_writer.write_flow(flow)
            counters["flows_written"] += 1

            if not args.quiet:
                features = flow.to_feature_dict()
                print(
                    f"flow={flow.flow_id()} duration_us={features['Flow Duration']:.0f} "
                    f"pkts={int(features['Tot Fwd Pkts'] + features['Tot Bwd Pkts'])}"
                )

    def on_packet(packet) -> None:
        counters["packets_seen"] += 1
        expired = flow_table.update_with_packet(packet)
        if expired:
            write_flows(expired)

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
        elif args.bpf and str(args.bpf).strip().lower() not in ("", "ip"):
            print(
                "Note: Offline PCAP BPF filtering requires tcpdump. "
                "tcpdump is not available; processing all packets (best-effort)."
            )
    elif args.iface:
        sniff_kwargs["iface"] = args.iface
        sniff_kwargs["filter"] = args.bpf

    source_desc = f"pcap={args.pcap}" if args.pcap else (f"iface={args.iface}" if args.iface else "default iface")
    print(f"Sniffing source: {source_desc}")
    print(f"CSV out: {csv_path}")
    print(
        "Flow idle timeout: "
        f"{float(args.flow_timeout)}s, "
        f"max_flow_duration: {float(args.max_flow_duration)}s, "
        f"min_packets: {int(args.min_packets)}"
    )

    try:
        sniff(**sniff_kwargs)
    except KeyboardInterrupt:
        print("Interrupted by user, flushing remaining flows...")

    write_flows(flow_table.flush_all())
    csv_writer.close()

    print(
        "Done. "
        f"packets_seen={counters['packets_seen']} "
        f"flows_written={counters['flows_written']}"
    )


if __name__ == "__main__":
    main()
