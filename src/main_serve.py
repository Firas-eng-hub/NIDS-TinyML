"""Run the Flask dashboard server."""

import argparse

from .config import load_config
from .dashboard.app import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg, _ = load_config(args.config)
    host = cfg["dashboard"]["host"]
    port = int(cfg["dashboard"]["port"])

    app = create_app()
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
