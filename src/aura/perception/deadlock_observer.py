
from __future__ import annotations

import ctypes
from ctypes import wintypes

import mss
from PIL import Image

from aura.perception.deadlock_detector import DeadlockDetector


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class DeadlockObserver:
    """
    Captures the visible Deadlock window when it is foregrounded.

    Screenshots are returned as in-memory PIL images and are not
    automatically written to disk.
    """

    def __init__(self):
        self.detector = DeadlockDetector()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)

        self.user32.GetWindowRect.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(RECT),
        ]
        self.user32.GetWindowRect.restype = wintypes.BOOL

    def capture_game_window(self) -> Image.Image | None:
        """
        Capture the Deadlock window.

        Returns:
            PIL Image if Deadlock is foregrounded.
            None if Deadlock is not ready for observation.
        """

        status = self.detector.get_status()

        if not status["ready_for_observation"]:
            return None

        hwnd = status["active_window"]["hwnd"]

        rect = RECT()

        if not self.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise ctypes.WinError(ctypes.get_last_error())

        width = rect.right - rect.left
        height = rect.bottom - rect.top

        if width <= 0 or height <= 0:
            raise RuntimeError("Deadlock window has invalid dimensions.")

        # Capture the virtual desktop so that windows on secondary
        # monitors and monitors with negative coordinates are supported.
        with mss.mss() as sct:
            virtual_monitor = sct.monitors[0]
            screenshot = sct.grab(virtual_monitor)

            desktop_image = Image.frombytes(
                "RGB",
                screenshot.size,
                screenshot.rgb,
            )

        # Convert Windows screen coordinates into screenshot coordinates.
        left = rect.left - virtual_monitor["left"]
        top = rect.top - virtual_monitor["top"]
        right = rect.right - virtual_monitor["left"]
        bottom = rect.bottom - virtual_monitor["top"]

        # Keep the crop within the captured desktop.
        left = max(0, left)
        top = max(0, top)
        right = min(desktop_image.width, right)
        bottom = min(desktop_image.height, bottom)

        if right <= left or bottom <= top:
            raise RuntimeError(
                "Deadlock window is outside the captured desktop."
            )

        game_image = desktop_image.crop((left, top, right, bottom))

        if game_image.width < 100 or game_image.height < 100:
            raise RuntimeError(
                "Captured Deadlock image is unexpectedly small."
            )

        return game_image
