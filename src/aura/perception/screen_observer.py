import time

from aura.perception.screen_capture import (
    capture_primary_monitor_image
)

from aura.perception.application_detector import (
    get_active_application
)

from aura.perception.screen_analyser import (
    ScreenChangeDetector
)


class ScreenObserver:

    def __init__(
        self,
        interval: float = 2.0,
        threshold: float = 0.035
    ):

        self.interval = interval

        self.detector = ScreenChangeDetector(
            threshold=threshold
        )

        self.previous_application = None
        self.running = False

    def observe_once(self):

        image = capture_primary_monitor_image()

        application = get_active_application()

        screen_result = self.detector.compare(image)

        application_changed = (
            application != self.previous_application
        )

        self.previous_application = application

        return {
            "application": application,
            "screen": screen_result,
            "application_changed": application_changed,
            "timestamp": time.time()
        }

    def start(self):

        self.running = True

        print("\nAURA Screen Observer started.")
        print("Press CTRL+C to stop.\n")

        try:

            while self.running:

                result = self.observe_once()

                if (
                    result["screen"]["changed"]
                    or result["application_changed"]
                ):

                    print("=" * 55)

                    print("SCREEN EVENT DETECTED")

                    print(
                        "Application:",
                        result["application"]["process"]
                    )

                    print(
                        "Window:",
                        result["application"]["title"]
                    )

                    print(
                        "PID:",
                        result["application"]["pid"]
                    )

                    print(
                        "Visual difference:",
                        result["screen"]["difference"]
                    )

                    print(
                        "Application changed:",
                        result["application_changed"]
                    )

                    print("=" * 55)

                time.sleep(self.interval)

        except KeyboardInterrupt:

            print("\nAURA Screen Observer stopped.")

        finally:

            self.running = False