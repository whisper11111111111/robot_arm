import re

import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

import config


class RobotEar:
    """Speech recognition module backed by faster-whisper."""

    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.fs = 16000

    def get_text(self, audio_frames):
        """Transcribe audio frames to text with silence trimming and duration guards.

        Args:
            audio_frames: list of numpy arrays captured from sounddevice InputStream.

        Returns:
            Transcribed string (stripped), or "" if audio is empty/too short/silent.
        """
        if not audio_frames:
            return ""

        audio_data = np.concatenate(audio_frames, axis=0)
        audio_flat = audio_data.flatten()

        # Trim leading/trailing silence to reduce Whisper hallucinations
        nonzero = np.where(np.abs(audio_flat) > config.AUDIO_SILENCE_THRESHOLD)[0]
        if len(nonzero) == 0:
            print("[Audio] No speech detected")
            return ""

        margin = int(self.fs * config.AUDIO_SILENCE_MARGIN)
        start = max(0, nonzero[0] - margin)
        end = min(len(audio_flat), nonzero[-1] + margin)
        audio_trimmed = audio_flat[start:end]

        duration = len(audio_trimmed) / self.fs
        if duration < config.AUDIO_MIN_DURATION:
            print(f"[Audio] Too short ({duration:.1f}s), skipping")
            return ""
        if duration > config.AUDIO_MAX_DURATION:
            print(f"[Audio] Too long ({duration:.1f}s), truncating to {config.AUDIO_MAX_DURATION:.0f}s")
            audio_trimmed = audio_trimmed[:int(self.fs * config.AUDIO_MAX_DURATION)]

        temp_file = "temp_voice.wav"
        wav.write(temp_file, self.fs, (audio_trimmed * 32767).astype(np.int16))

        segments, _ = self.model.transcribe(
            temp_file,
            beam_size=5,
            language="zh",
            no_speech_threshold=0.5,
            condition_on_previous_text=False,  # prevents "向右向右向右..." hallucination loop
            # i18n: domain hint for Whisper — Chinese robot command vocabulary
            initial_prompt="机械臂控制指令：抓取,抬起,放下,松开,复位,点头,摇头,削笔刀,盒子,零件,瓶子,厘米,毫米,向上,向下,向左,向右,向前,向后"
        )

        text = "".join(s.text for s in segments)
        return self._fix_recognition(text.strip())

    def _fix_recognition(self, text):
        """Post-process ASR output: punctuation removal, homophone correction, dedup."""
        if not text:
            return text

        text = re.sub(r'[,，。！？!?、;；]', '', text)

        # i18n: Chinese homophone correction table (Whisper mishearings → correct words)
        replacements = {
            '小笔刀': '削笔刀', '消笔刀': '削笔刀', '销笔刀': '削笔刀',
            '零米': '厘米', '里米': '厘米', '黎米': '厘米', '离米': '厘米',
            '公分': '厘米', '利米': '厘米',
            '电头': '点头', '点投': '点头', '店头': '点头', '垫头': '点头',
            '药头': '摇头', '要头': '摇头', '右头': '摇头', '咬头': '摇头', '摇土': '摇头',
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)

        # Detect and strip repeated-phrase hallucinations like "向右向右向右..."
        dedup_match = re.match(r'^(.{2,8}?)(.{2,8}?)\2{2,}', text)
        if dedup_match:
            text = dedup_match.group(1)
            print(f"[Dedup] Repeated hallucination stripped, kept: {text}")

        if len(text) > 30:
            words = re.findall(r'向[上下左右前后]', text)
            if len(words) > 3:
                first_match = re.search(r'(.*?向[上下左右前后].*?\d+.*?厘米)', text)
                text = first_match.group(1) if first_match else text[:20]
                print(f"[Dedup] Overlong text truncated to: {text}")

        return text.strip()
