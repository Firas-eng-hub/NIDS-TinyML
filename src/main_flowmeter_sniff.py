"""Scapy-based live/offline flowmeter that exports 15 flow features to CSV."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from scapy.all import sniff

from .config import load_config
from .features.feature_list import FEATURE_LIST
from .flows import FlowStats, FlowTable, _build_offline_lfilter


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
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
        help="If >0, force-rotate long-lived flows after this many seconds.",
    )
    parser.add_argument(
        "--min_packets",
        type=int,
        default=2,
        help="Minimum packets before exporting a flow row",
    )
    parser.add_argument(
        "--csv_out",
        default="data/processed/live_flows_15.csv",
        help="CSV file to append exported flows",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable per-flow console output"
    )
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
    csv_writer = FlowCsvWriter(
        csv_path=csv_path, feature_order=configured_feature_order
    )

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
        _, expired = flow_table.update_with_packet(packet)
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

    source_desc = (
        f"pcap={args.pcap}"
        if args.pcap
        else (f"iface={args.iface}" if args.iface else "default iface")
    )
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
