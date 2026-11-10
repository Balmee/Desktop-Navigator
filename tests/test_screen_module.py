from pathlib import Path
from aura.screen import capture_primary_monitor

def test_capture_returns_existing_png():
    path = capture_primary_monitor()
    try:
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix.lower() == ".png"
        assert path.stat().st_size > 0
    finally:
        path.unlink(missing_ok=True)
