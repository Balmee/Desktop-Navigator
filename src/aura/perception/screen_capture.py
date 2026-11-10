import mss

from PIL import Image


def capture_primary_monitor_image() -> Image.Image:
    """
    Capture the primary monitor and return a PIL image.

    The image remains in memory and is not automatically
    saved to disk.
    """

    with mss.mss() as sct:

        if len(sct.monitors) < 2:
            raise RuntimeError("No monitor detected.")

        monitor = sct.monitors[1]

        screenshot = sct.grab(monitor)

        image = Image.frombytes(
            "RGB",
            screenshot.size,
            screenshot.rgb
        )

        return image