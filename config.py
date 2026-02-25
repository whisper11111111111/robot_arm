"""
config.py
=========
项目全局配置文件。

所有硬件参数、路径、标定数据均在此集中管理。
需要调整参数时只需修改本文件，无需改动各功能模块。
"""

import os

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

# 项目根目录（以本文件位置为基准，避免工作目录不一致的问题）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# YOLO 自定义训练模型路径
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# 微调 LLM 模型路径（需替换为实际导出路径）
LLM_MODEL_PATH = r"D:\lora\2"

# Whisper 临时音频文件路径
TEMP_AUDIO_FILE = os.path.join(BASE_DIR, "temp_voice.wav")

# ─────────────────────────────────────────────
# 硬件配置
# ─────────────────────────────────────────────

# ESP32 串口参数
SERIAL_PORT = "COM3"
SERIAL_BAUD = 115200

# 摄像头
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720

# ─────────────────────────────────────────────
# 机械臂运动参数
# ─────────────────────────────────────────────

# 减震参数
ARM_DAMPING_BUFFER_SIZE = 3      # 移动平均窗口大小
ARM_DAMPING_MAX_SPEED   = 25.0   # 舵机最大速度（度/帧）
ARM_DAMPING_FACTOR      = 0.6    # 阻尼系数 [0~1]

# 倾斜修正偏移量（单位：度）
# 若水平移动时末端下沉，可适当增大 OFFSET_Y
ARM_CORRECTION_OFFSET_Y = -10.0
ARM_CORRECTION_OFFSET_Z =   0.0

# 初始待机位置 [x, y, z]，单位：mm
ARM_REST_POSITION = [120.0, 0.0, 60.0]

# ─────────────────────────────────────────────
# 视觉标定参数
# ─────────────────────────────────────────────

# 机器人坐标系下的 4 个标定点，单位：mm
# 顺序：P1(左上) → P2(右上) → P3(右下) → P4(左下)
CALIB_ROBOT_POINTS = [
    [ 90,  90],
    [200,  90],
    [200, -90],
    [ 90, -90],
]

# 对应的初始像素坐标（需随摄像头安装位置调整）
CALIB_IMAGE_POINTS = [
    [ 817,  72],
    [ 433,  79],
    [ 291, 612],
    [1029, 610],
]

# ─────────────────────────────────────────────
# 语音识别参数
# ─────────────────────────────────────────────

WHISPER_MODEL_SIZE   = "base"    # 可选：tiny / base / small / medium
AUDIO_SAMPLE_RATE    = 16000     # 采样率（Hz）
AUDIO_MIN_DURATION   = 0.5       # 低于此时长（秒）的音频直接丢弃
AUDIO_MAX_DURATION   = 15.0      # 超过此时长（秒）的音频截断处理
AUDIO_SILENCE_MARGIN = 0.3       # 首尾静音保留余量（秒）
