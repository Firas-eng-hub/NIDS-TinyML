"""In-memory event storage."""

from threading import Lock
from typing import Dict, List, Optional


class EventStore:
    def __init__(self):
        self._lock = Lock()
        self._events: List[Dict[str, object]] = []

    def add_event(self, event: Dict[str, object]) -> None:
        with self._lock:
            self._events.append(event)

    def add_events(self, events: List[Dict[str, object]]) -> None:
        with self._lock:
            self._events.extend(events)

    def query_events(
        self,
        limit: int = 100,
        class_filter: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, object]]:
        with self._lock:
            events = list(self._events)

        if class_filter:
            events = [e for e in events if e.get("predicted_class") == class_filter]

        if min_confidence > 0:
            events = [
                e for e in events if float(e.get("confidence", 0.0)) >= min_confidence
            ]

        events = sorted(events, key=lambda e: e.get("ts", ""), reverse=True)
        return events[:limit]

    def stats(self) -> Dict[str, object]:
        with self._lock:
            events = list(self._events)

        total = len(events)
        by_class: Dict[str, int] = {}
        for e in events:
            cls = e.get("predicted_class", "Unknown")
            by_class[cls] = by_class.get(cls, 0) + 1

        return {"total": total, "by_class": by_class}
