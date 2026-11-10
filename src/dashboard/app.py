"""Flask dashboard app."""

from datetime import datetime, timezone
from typing import List

from flask import Flask, jsonify, render_template, request

from .storage import EventStore

store = EventStore()


def _normalize_event(event: dict) -> dict:
    if "ts" not in event:
        event["ts"] = datetime.now(timezone.utc).isoformat()
    return event


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/events")
    def ingest_events():
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Invalid JSON"}), 400

        if isinstance(data, list):
            events: List[dict] = [_normalize_event(e) for e in data]
            store.add_events(events)
            return jsonify({"status": "ok", "count": len(events)})

        if isinstance(data, dict):
            event = _normalize_event(data)
            store.add_event(event)
            return jsonify({"status": "ok", "count": 1})

        return jsonify({"error": "Invalid payload"}), 400

    @app.get("/api/events")
    def get_events():
        limit = int(request.args.get("limit", 100))
        class_filter = request.args.get("class")
        min_confidence = float(request.args.get("min_confidence", 0.0))
        events = store.query_events(
            limit=limit, class_filter=class_filter, min_confidence=min_confidence
        )
        return jsonify({"events": events})

    @app.get("/api/stats")
    def get_stats():
        return jsonify(store.stats())

    return app


def main() -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
