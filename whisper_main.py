"""
whisper_main.py
===============
语音识别模块。

基于 Faster-Whisper 封装本地语音识别引擎，提供将
音频数据转化为中文文字的统一接口。
"""

# ── 标准库 ──
import os

# ── 第三方库 ──
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

# ── 项目配置 ──
from config import WHISPER_MODEL_SIZE, AUDIO_SAMPLE_RATE, TEMP_AUDIO_FILE


class RobotEar:
    """
    语音识别封装器（"耳朵"）。

    负责将外部传入的音频数据写为临时 WAV 文件，
    并调用 Faster-Whisper 进行离线中文转录。

    Args:
        model_size (str): Whisper 模型规格，默认读取 config.WHISPER_MODEL_SIZE。
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE) -> None:
        print(f">>> [耳朵] 正在加载语音模型 ({model_size})…")
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.sample_rate = AUDIO_SAMPLE_RATE

    def get_text(self, audio_data: np.ndarray, initial_prompt: str = "") -> str:
        """将音频数组转化为中文文字。

        Args:
            audio_data:     归一化到 [-1, 1] 的浮点音频数组，形状 (N,) 或 (N, 1)。
            initial_prompt: 传递给 Whisper 的提示词，可改善领域词识别精度。

        Returns:
            识别出的文字内容（已去除首尾空白）。
        """
        audio_flat = audio_data.flatten()
        wav.write(TEMP_AUDIO_FILE, self.sample_rate, (audio_flat * 32767).astype(np.int16))

        segments, _ = self.model.transcribe(
            TEMP_AUDIO_FILE,
            beam_size=5,
            language="zh",
            initial_prompt=initial_prompt,
        )
        text = "".join(s.text for s in segments)

        # 清除临时文件
        try:
            os.remove(TEMP_AUDIO_FILE)
        except OSError:
            pass

        return text.strip()
