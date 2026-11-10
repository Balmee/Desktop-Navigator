from pathlib import Path
from tempfile import NamedTemporaryFile
import mss
from PIL import Image

def capture_primary_monitor() -> Path:
    # Capture the primary monitor to a temporary PNG; caller must delete it.
    with mss.mss() as sct:
        if len(sct.monitors) < 2:
            raise RuntimeError("No display monitor was detected.")
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.rgb)

    temp = NamedTemporaryFile(prefix="aura_screen_", suffix=".png", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    image.save(temp_path, format="PNG")
    return temp_path
