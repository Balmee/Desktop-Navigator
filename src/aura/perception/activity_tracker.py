
import re


ACTIVITY_KEYWORDS = {
    "debugging": [
        "traceback", "exception", "error", "debug",
        "breakpoint", "stack trace", "failed",
    ],
    "coding": [
        "def ", "class ", "import ", "function",
        "return ", "const ", "public static",
    ],
    "terminal_work": [
        "powershell", "cmd.exe", "pip install",
        "python ", "npm ", "git ", "terminal",
    ],
    "research": [
        "search", "documentation", "references",
        "research", "results", "article",
    ],
    "document_writing": [
        "word", "document", "paragraph",
        "heading", "page layout",
    ],
    "game_development": [
        "hierarchy", "inspector", "scene",
        "gameobject", "blueprint", "unreal",
        "unity",
    ],
}


class ActivityTracker:
    def __init__(self, confirmation_count: int = 2):
        self.confirmation_count = confirmation_count
        self.current_activity = "unknown"
        self.candidate_activity = None
        self.candidate_count = 0

    def classify(self, application: dict, ocr: dict) -> dict:
        process = (
            application.get("process") or ""
        ).lower()

        title = (
            application.get("title") or ""
        ).lower()

        lines = ocr.get("lines", [])
        screen_text = " ".join(lines).lower()

        combined_text = f"{title} {screen_text}"

        scores = {}

        for activity, keywords in ACTIVITY_KEYWORDS.items():
            scores[activity] = sum(
                1 for keyword in keywords
                if keyword in combined_text
            )

        # Application identity provides additional context,
        # but does not establish a specific task by itself.
        if process in {"code.exe", "devenv.exe"}:
            scores["coding"] += 1

        if process == "unity.exe":
            scores["game_development"] += 2

        if process == "unrealeditor.exe":
            scores["game_development"] += 2

        if process in {"chrome.exe", "msedge.exe", "firefox.exe"}:
            scores["research"] += 1

        if scores and max(scores.values()) > 0:
            activity = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[activity] / max(total, 1)
        else:
            activity = "unknown"
            confidence = 0.0

        return {
            "activity": activity,
            "confidence": round(confidence, 2),
            "scores": scores,
        }

    def update(self, application: dict, ocr: dict) -> dict:
        result = self.classify(application, ocr)
        observed = result["activity"]

        if observed == self.candidate_activity:
            self.candidate_count += 1
        else:
            self.candidate_activity = observed
            self.candidate_count = 1

        changed = False

        if (
            self.candidate_count >= self.confirmation_count
            and observed != self.current_activity
        ):
            self.current_activity = observed
            changed = True

        return {
            **result,
            "current_activity": self.current_activity,
            "confirmed": (
                self.candidate_count >= self.confirmation_count
            ),
            "activity_changed": changed,
            "observations": self.candidate_count,
        }

    def reset(self):
        self.current_activity = "unknown"
        self.candidate_activity = None
        self.candidate_count = 0
