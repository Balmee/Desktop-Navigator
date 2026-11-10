import argparse
from aura.config import settings
from aura.llm import AuraLLM
from aura.screen import capture_primary_monitor
from aura.perception.screen_observer import ScreenObserver

def command_doctor():
    print("AURA diagnostics")
    print(f"Ollama endpoint: {settings.ollama_host}")
    print(f"Selected model:  {settings.model}")
    try:
        result = AuraLLM().check_connection()
        models = getattr(result, "models", [])
        names = [getattr(m, "model", getattr(m, "name", "unknown")) for m in models]
        print("[OK] Connected to Ollama.")
        if names:
            print("Installed models:")
            for name in names:
                print(f"  - {name}")
        if not any(settings.model in name for name in names):
            print(f"[WARN] The selected model may not be installed. Run: ollama pull {settings.model}")
    except Exception as exc:
        print("[ERROR] Could not connect to Ollama.")
        print(f"Details: {exc}")
        print("Check that Ollama is installed and running.")
        raise SystemExit(1)

def command_chat():
    try:
        assistant = AuraLLM()
        assistant.check_connection()
    except Exception as exc:
        print(f"Cannot reach Ollama at {settings.ollama_host}: {exc}")
        raise SystemExit(1)

    print(f"AURA is ready. Model: {settings.model}")
    print("Type 'exit' or 'quit' to end the session.\n")
    while True:
        try:
            prompt = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAURA session ended.")
            break
        if prompt.lower() in {"exit", "quit"}:
            print("AURA > Goodbye.")
            break
        if not prompt:
            continue
        try:
            print(f"\nAURA > {assistant.chat(prompt)}\n")
        except Exception as exc:
            print(f"\n[LLM error] {exc}\n")

def command_ask_screen(prompt: str):
    print("This captures the PRIMARY MONITOR once and sends it to your configured Ollama endpoint.")
    consent = input("Proceed with screenshot capture? [y/N]: ").strip().lower()
    if consent not in {"y", "yes"}:
        print("Screen capture cancelled.")
        return

    image_path = None
    try:
        image_path = capture_primary_monitor()
        assistant = AuraLLM()
        print("\nAURA is analysing the screenshot...\n")
        answer = assistant.ask_about_image(prompt, str(image_path))
        print(f"AURA > {answer}")
    except Exception as exc:
        print(f"[Screen analysis error] {exc}")
        raise SystemExit(1)
    finally:
        if image_path and image_path.exists():
            image_path.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(prog="aura", description="AURA local desktop AI companion")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Ollama connectivity and model availability")
    sub.add_parser("chat", help="Start a local text conversation")
    screen_parser = sub.add_parser("ask-screen", help="Ask a question about one confirmed screenshot")
    screen_parser.add_argument(
        "prompt",
        nargs="?",
        default="Describe what is visible on my screen. Separate observations from assumptions and suggest one useful next step."

    )
    observe_parser = sub.add_parser(
        "observe-screen",
        help="Monitor desktop changes and active applications"
    )

    observe_parser.add_argument(
        "--interval",
        type=float,
        default=2.0
    )

    observe_parser.add_argument(
        "--threshold",
        type=float,
        default=0.035
    )

    args = parser.parse_args()
    if args.command == "doctor":
        command_doctor()
    elif args.command == "chat":
        command_chat()
    elif args.command == "ask-screen":
        command_ask_screen(args.prompt)
    elif args.command == "observe-screen":

        observer = ScreenObserver(
            interval=args.interval,
            threshold=args.threshold
        )

        observer.start()

if __name__ == "__main__":
    main()
