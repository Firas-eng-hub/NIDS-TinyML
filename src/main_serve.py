"""Run the Flask dashboard server."""

import argparse
from pathlib import Path

import yaml

from .dashboard.app import create_app


def load_config(config_path: str = None):
    root = Path(__file__).resolve().parents[1]
    if config_path is None:
        config_path = root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = cfg["dashboard"]["host"]
    port = int(cfg["dashboard"]["port"])

    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
