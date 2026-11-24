
from aura.intelligence.coach import GameplayCoach
from aura.voice.speech_synthesis import PiperSpeechSynthesizer


class GameplayCoachingSession:
    """Connects generative gameplay reasoning to spoken output."""

    def __init__(self):
        print("Initialising AURA gameplay coach...")

        self.coach = GameplayCoach()

        print("Initialising AURA speech synthesis...")

        self.speech = PiperSpeechSynthesizer()

        print("AURA coaching session ready.")

    def coach_and_speak(self, context: str) -> str:
        """Generate original advice and speak it using Piper."""

        print("\nAnalysing gameplay situation...")

        advice = self.coach.generate_advice(context)

        if not advice:
            raise RuntimeError("The language model returned no advice.")

        print("\nGenerated coaching advice:")
        print(advice)

        print("\nSpeaking response...")

        self.speech.speak(advice)

        return advice

    def stop_speaking(self):
        self.speech.stop()
