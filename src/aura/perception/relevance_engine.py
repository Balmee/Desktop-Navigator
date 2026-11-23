
class RelevanceEngine:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def evaluate(
        self,
        application_changed: bool,
        screen_changed: bool,
        activity_changed: bool,
        activity: dict,
        ocr: dict,
    ) -> dict:

        score = 0.0
        reasons = []

        if application_changed:
            score += 0.25
            reasons.append("application_changed")

        if screen_changed:
            score += 0.20
            reasons.append("screen_changed")

        if activity_changed:
            score += 0.35
            reasons.append("activity_changed")

        text = " ".join(
            ocr.get("lines", [])
        ).lower()

        error_signals = [
            "traceback",
            "exception",
            "fatal error",
            "build failed",
            "compilation failed",
            "error:",
        ]

        if any(signal in text for signal in error_signals):
            score += 0.25
            reasons.append("possible_error_detected")

        score = min(score, 1.0)

        candidate = (
            score >= self.threshold
            and activity.get("confirmed", False)
        )

        return {
            "score": round(score, 2),
            "threshold": self.threshold,
            "decision": (
                "candidate_assistance"
                if candidate
                else "stay_quiet"
            ),
            "reasons": reasons,
        }
