
import psutil

from aura.perception.application_detector import get_active_application


class DeadlockDetector:
    """Detects the Deadlock game process and foreground window."""

    PROCESS_NAME = "deadlock.exe"

    def find_game_processes(self) -> list[dict]:
        """Find all running Deadlock processes."""

        processes = []

        for process in psutil.process_iter(["pid", "name"]):
            try:
                name = process.info["name"]

                if name and name.lower() == self.PROCESS_NAME:
                    processes.append({
                        "pid": process.info["pid"],
                        "process": name,
                    })

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return processes

    def is_running(self) -> bool:
        """Check whether Deadlock is running."""

        return len(self.find_game_processes()) > 0

    def is_foreground(self) -> bool:
        """Check whether Deadlock is the active foreground application."""

        application = get_active_application()

        process_name = application.get("process")

        return (
            process_name is not None
            and process_name.lower() == self.PROCESS_NAME
        )

    def get_status(self) -> dict:
        """Return the current Deadlock detection state."""

        processes = self.find_game_processes()
        application = get_active_application()

        foreground = (
            application.get("process") is not None
            and application["process"].lower() == self.PROCESS_NAME
        )

        return {
            "running": len(processes) > 0,
            "foreground": foreground,
            "processes": processes,
            "active_window": application,
            "ready_for_observation": (
                len(processes) > 0 and foreground
            ),
        }
