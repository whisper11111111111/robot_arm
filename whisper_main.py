import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from faster_whisper import WhisperModel


class RobotEar:
    """Speech recognition module backed by faster-whisper."""

    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.fs = 16000

    def get_text(self, audio_data):
        """Transcribe audio frames to text.

        Args:
            audio_data: list of numpy arrays captured from sounddevice InputStream.

        Returns:
            Transcribed string (stripped).
        """
        temp_file = "temp_voice.wav"
        audio_np = np.concatenate(audio_data, axis=0)
        wav.write(temp_file, self.fs, (audio_np * 32767).astype(np.int16))
        segments, _ = self.model.transcribe(temp_file, beam_size=5, language="zh")
        return "".join(s.text for s in segments).strip()
