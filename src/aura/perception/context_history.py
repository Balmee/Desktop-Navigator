
from collections import deque
from datetime import datetime, timezone


class ContextHistory:
    def __init__(self, max_entries: int = 10):
        self.entries = deque(maxlen=max_entries)

    def add(self, context: dict):
        entry = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "application": context.get("application"),
            "application_context": context.get("context", {}),
            "text": context.get("ocr", {}).get("lines", []),
            "screen_change": context.get("screen_change"),
        }

        self.entries.append(entry)
        return entry

    def get_recent(self, limit: int = 5) -> list:
        return list(self.entries)[-limit:]

    def clear(self):
        self.entries.clear()
