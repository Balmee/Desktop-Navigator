
from aura.perception.screen_capture import (
    capture_primary_monitor_image,
)
from aura.perception.application_detector import (
    get_active_application,
)
from aura.perception.screen_analyser import (
    ScreenChangeDetector,
)
from aura.perception.ocr import extract_text
from aura.perception.context_history import ContextHistory
from aura.perception.application_profiles import (
    get_application_profile,
)


SENSITIVE_PROCESSES = {
    "1password.exe",
    "keepass.exe",
    "keepassxc.exe",
    "bitwarden.exe",
    "lastpass.exe",
    "logonui.exe",
    "credentialui.exe",
}


class ContextEngine:

    def __init__(self):
        self.screen_detector = ScreenChangeDetector()
        self.history = ContextHistory(max_entries=10)

    def inspect(self) -> dict:
        # 1. Identify the foreground application.
        application = get_active_application()

        process_name = (
            application.get("process") or ""
        ).lower()

        # 2. Apply the initial sensitive-application filter.
        if process_name in SENSITIVE_PROCESSES:
            return {
                "application": application,
                "ocr": {
                    "lines": [],
                    "word_count": 0,
                    "line_count": 0,
                },
                "screen_change": None,
                "skipped": True,
                "reason": "sensitive_application",
            }

        # 3. Identify the application category.
        profile = get_application_profile(
            application.get("process")
        )

        # 4. Capture the current screen.
        image = capture_primary_monitor_image()

        # 5. Detect visual changes.
        screen_result = self.screen_detector.compare(
            image
        )

        # 6. Extract visible text.
        ocr_result = extract_text(image)

        # 7. Construct structured context.
        context = {
            "application": application,
            "context": {
                "application_category": profile["category"],
                "application_description": profile["description"],
                "possible_tasks": profile["task_hints"],
            },
            "ocr": ocr_result,
            "screen_change": screen_result,
            "skipped": False,
        }

        # 8. Retain this observation temporarily.
        self.history.add(context)

        return context

    def recent_context(self, limit: int = 5):
        return self.history.get_recent(limit)

