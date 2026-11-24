
from aura.voice.audio_manager import AudioManager
from aura.voice.speech_synthesis import SpeechRecognizer


class VoiceInterface:

    def __init__(self):
        self.audio_manager = AudioManager()
        self.recognizer = SpeechRecognizer(
            model_size="small.en",
            device="cpu",
            compute_type="int8",
        )

    def listen(self, duration: float = 6.0) -> dict:

        audio = self.audio_manager.record(
            duration=duration
        )

        print("\nProcessing speech...")

        result = self.recognizer.transcribe(audio)

        print("\nAURA heard:")
        print(result["text"] or "[No speech recognised]")

        return result
