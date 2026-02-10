"""Label mapping utilities for IDS2018."""

from typing import Optional

TARGET_CLASSES = ["Normal", "DoS", "BruteForce", "Botnet", "WebAttack"]

LABEL_MAPPING_INFO = {
    "Normal": "BENIGN -> Normal",
    "DoS": "Contains DoS, DDoS, HOIC, or LOIC",
    "BruteForce": "Contains Brute Force, BruteForce, SSH-Bruteforce, or FTP-Bruteforce",
    "Botnet": "Contains Bot or Botnet",
    "WebAttack": "Contains Web Attack, SQL Injection, or XSS",
}


def map_label(raw_label: str) -> Optional[str]:
    """Map a raw label string into one of the 5 target classes.

    Returns None if the label should be dropped.
    """
    if raw_label is None:
        return None

    label = str(raw_label).strip()
    lower = label.lower()

    if lower == "benign":
        return "Normal"

    if "dos" in lower or "ddos" in lower or "hoic" in lower or "loic" in lower:
        return "DoS"

    if (
        "brute force" in lower
        or "bruteforce" in lower
        or "ssh-bruteforce" in lower
        or "ftp-bruteforce" in lower
    ):
        return "BruteForce"

    if "bot" in lower or "botnet" in lower:
        return "Botnet"

    if "web attack" in lower or "sql injection" in lower or "xss" in lower:
        return "WebAttack"

    return None
