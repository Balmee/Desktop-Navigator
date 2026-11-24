
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig


class PiperSpeechSynthesizer:
    """Local speech synthesis and playback using Piper TTS."""

    def __init__(
        self,
        model_path: str | Path = "models/voices/en_GB-alan-medium.onnx",
        volume: float = 1.0,
        speed: float = 1.0,
        output_device: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.volume = volume
        self.speed = speed
        self.output_device = output_device

        if not 0.0 <= volume <= 1.0:
            raise ValueError("Volume must be between 0.0 and 1.0.")

        if speed <= 0:
            raise ValueError("Speed must be greater than zero.")

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Piper voice model not found: {self.model_path.resolve()}"
            )

        print(f"Loading Piper voice: {self.model_path.name}")

        self.voice = PiperVoice.load(str(self.model_path))

        print("Piper voice loaded successfully.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Generate audio in memory and return audio samples and sample rate."""

        text = text.strip()

        if not text:
            raise ValueError("Speech text cannot be empty.")

        config = SynthesisConfig(
            volume=self.volume,
            length_scale=1.0 / self.speed,
            normalize_audio=True,
        )

        audio_buffer = io.BytesIO()

        with wave.open(audio_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=config,
            )

        audio_buffer.seek(0)

        with wave.open(audio_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        if sample_width != 2:
            raise RuntimeError(
                f"Expected 16-bit PCM audio, received {sample_width * 8}-bit."
            )

        audio = np.frombuffer(frames, dtype=np.int16)

        if channels > 1:
            audio = audio.reshape(-1, channels)

        return audio, sample_rate

    def speak(self, text: str) -> None:
        """Generate and play a spoken response."""

        audio, sample_rate = self.synthesize(text)

        if audio.size == 0:
            raise RuntimeError("Piper generated empty audio.")

        print(f"AURA: {text}")

        sd.play(
            audio,
            samplerate=sample_rate,
            device=self.output_device,
            blocking=True,
        )

    def stop(self) -> None:
        """Stop current audio playback."""

        sd.stop()
