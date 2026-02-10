# 文件名: whisper_main.py
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

class RobotEar:
    def __init__(self, model_size="base"):
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.fs = 16000
        self.recording_buffer = []

    def start_recording(self):
        self.recording_buffer = []
        # 开始长录音
        sd.start_stream(samplerate=self.fs, channels=1)
        print(">>> [耳朵] 录音中...")

    def record_callback(self, indata, frames, time, status):
        self.recording_buffer.append(indata.copy())

    def get_text(self, audio_data):
        """将传入的音频数组转为文字"""
        temp_file = "temp_voice.wav"
        # 归一化音频数据
        audio_np = np.concatenate(audio_data, axis=0)
        wav.write(temp_file, self.fs, (audio_np * 32767).astype(np.int16))
        
        segments, info = self.model.transcribe(temp_file, beam_size=5, language="zh")
        text = "".join([s.text for s in segments])
        return text.strip()