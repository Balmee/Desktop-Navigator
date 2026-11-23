
import time

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
from aura.perception.application_profiles import (
    get_application_profile,
)
from aura.perception.activity_tracker import (
    ActivityTracker,
)
from aura.perception.relevance_engine import (
    RelevanceEngine,
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


class AdaptiveObserver:
    def __init__(
        self,
        interval: float = 2.0,
        threshold: float = 0.035,
        ocr_cooldown: float = 5.0,
    ):
        if interval <= 0:
            raise ValueError("Interval must be positive.")

        if ocr_cooldown < 0:
            raise ValueError("OCR cooldown cannot be negative.")

        self.interval = interval
        self.ocr_cooldown = ocr_cooldown

        self.screen_detector = ScreenChangeDetector(
            threshold=threshold
        )

        self.activity_tracker = ActivityTracker()
        self.relevance_engine = RelevanceEngine()

        self.previous_application = None
        self.last_ocr_at = 0.0
        self.running = False

        self.last_ocr = {
            "lines": [],
            "word_count": 0,
            "line_count": 0,
        }

    def observe_once(self):
        application = get_active_application()

        process = (
            application.get("process") or ""
        ).lower()

        application_changed = (
            application != self.previous_application
        )

        self.previous_application = application

        if process in SENSITIVE_PROCESSES:
            self.activity_tracker.reset()

            return {
                "skipped": True,
                "reason": "sensitive_application",
                "application": application,
            }

        image = capture_primary_monitor_image()

        screen_result = self.screen_detector.compare(
            image
        )

        screen_changed = screen_result["changed"]

        should_run_ocr = (
            application_changed or screen_changed
        ) and (
            time.monotonic() - self.last_ocr_at
            >= self.ocr_cooldown
        )

        if should_run_ocr:
            self.last_ocr = extract_text(image)
            self.last_ocr_at = time.monotonic()

        profile = get_application_profile(
            application.get("process")
        )

        activity = self.activity_tracker.update(
            application,
            self.last_ocr,
        )

        relevance = self.relevance_engine.evaluate(
            application_changed=application_changed,
            screen_changed=screen_changed,
            activity_changed=activity["activity_changed"],
            activity=activity,
            ocr=self.last_ocr,
        )

        return {
            "timestamp": time.time(),
            "skipped": False,
            "application": application,
            "application_profile": profile,
            "screen": screen_result,
            "ocr": self.last_ocr,
            "ocr_updated": should_run_ocr,
            "activity": activity,
            "relevance": relevance,
        }

    def start(self):
        self.running = True

        print("\nAURA Adaptive Observer started.")
        print("Screen observation is active.")
        print("No screenshots are saved or sent to an LLM.")
        print("Press CTRL+C to stop.\n")

        try:
            while self.running:
                started = time.monotonic()

                try:
                    result = self.observe_once()

                    if result.get("skipped"):
                        print(
                            "\nObservation skipped:",
                            result["reason"],
                        )
                    elif (
                        result["application"] !=
                        getattr(self, "_displayed_application", None)
                        or result["activity"]["activity_changed"]
                        or result["relevance"]["decision"]
                        == "candidate_assistance"
                    ):
                        print("\n" + "=" * 60)
                        print("AURA CONTEXT UPDATE")
                        print(
                            "Application:",
                            result["application"].get("process"),
                        )
                        print(
                            "Window:",
                            result["application"].get("title"),
                        )
                        print(
                            "Category:",
                            result["application_profile"]["category"],
                        )
                        print(
                            "Activity:",
                            result["activity"]["current_activity"],
                        )
                        print(
                            "Activity confidence:",
                            result["activity"]["confidence"],
                        )
                        print(
                            "Relevance:",
                            result["relevance"]["score"],
                        )
                        print(
                            "Decision:",
                            result["relevance"]["decision"],
                        )
                        print(
                            "Reasons:",
                            result["relevance"]["reasons"],
                        )
                        print(
                            "OCR updated:",
                            result["ocr_updated"],
                        )
                        print("=" * 60)

                    self._displayed_application = (
                        result.get("application")
                    )

                except Exception as error:
                    print("Observation error:", error)

                elapsed = time.monotonic() - started
                time.sleep(max(0.0, self.interval - elapsed))

        except KeyboardInterrupt:
            print("\nAURA Adaptive Observer stopped.")

        finally:
            self.running = False
