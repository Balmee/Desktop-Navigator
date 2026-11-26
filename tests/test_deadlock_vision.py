
import json

from aura.perception.deadlock_observer import DeadlockObserver
from aura.perception.screen_analyser import DeadlockScreenAnalyser


def main():
    observer = DeadlockObserver()

    print("Checking Deadlock foreground status...")

    image = observer.capture_game_window()

    if image is None:
        print("Deadlock is not foregrounded. Test cancelled.")
        return

    print(f"Captured image: {image.width}x{image.height}")

    analyser = DeadlockScreenAnalyser()

    print("\nSending screenshot to local vision model...")
    result = analyser.analyse(image)

    print("\n========== AURA VISUAL ANALYSIS ==========\n")
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print("\n==========================================")

    print("\nVision analysis completed successfully.")


if __name__ == "__main__":
    main()
