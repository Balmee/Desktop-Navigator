
import numpy as np
import sounddevice as sd


class AudioManager:

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def record(self, duration: float = 6.0) -> np.ndarray:
        if duration <= 0:
            raise ValueError("Recording duration must be positive.")

        print(f"\nRecording for {duration:.1f} seconds...")
        print("Speak now.")

        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )

        sd.wait()

        audio = recording.flatten()

        if audio.size == 0:
            raise RuntimeError("No audio was recorded.")

        if np.max(np.abs(audio)) < 0.001:
            raise RuntimeError(
                "No meaningful microphone audio detected. "
                "Check your microphone and input device."
            )

        return audio
