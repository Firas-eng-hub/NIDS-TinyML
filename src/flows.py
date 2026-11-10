"""Shared IPv4 flow aggregation for capture and inference."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scapy.all import IP, TCP, UDP


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
        return (
            f"{self.src}:{self.sport}->{self.dst}:{self.dport}/{self.proto}@{start_ms}"
        )

    def to_feature_dict(self) -> Dict[str, float]:
        duration_s = self.duration_seconds()
        flow_pkts_per_s = (
            0.0 if duration_s <= 1e-9 else self.total_packets() / duration_s
        )
        down_up_ratio = (
            0.0 if self.fwd_pkts == 0 else self.bwd_pkts / float(self.fwd_pkts)
        )
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
    def _extract_packet(
        packet,
    ) -> Optional[Tuple[Tuple[str, str, int, int, int], int, int, float]]:
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
            key
            for key, flow in self.table.items()
            if (now_ts - flow.last_ts) >= self.timeout_seconds
        ]
        expired: List[FlowStats] = []
        for key in expired_keys:
            expired.append(self.table.pop(key))
        return expired

    def flush_all(self) -> List[FlowStats]:
        all_flows = list(self.table.values())
        self.table.clear()
        return all_flows
