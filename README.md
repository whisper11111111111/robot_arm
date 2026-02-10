# 🤖 智能语音具身智能机械臂 (Voice-Controlled Embodied AI Robot Arm)

这是一个基于多模态大模型的全栈具身智能项目，实现了从**语音指令**到**机械臂动作**的端到端控制。项目完全运行在本地消费级硬件（RTX 3060 Laptop）上，集成了自动语音识别（ASR）、大语言模型（LLM）指令跟随、视觉目标检测（CV）与逆运动学控制（IK）。

> **关键词**: `Embodied AI`, `Robot Arm`, `Voice Control`, `LLM`, `YOLOv8`, `Whisper`, `Fine-tuning`, `DeepSeek`, `ESP32`, `Inverse Kinematics`

## ✨ 项目亮点

*   **👂 听 (Listen)**: 使用 **Faster-Whisper** 进行本地语音识别，支持抗噪与谐音纠错。
*   **🧠 想 (Think)**: 基于 **DeepSeek/Qwen** 大模型微调 (LoRA)，将自然语言能够解析为结构化的 JSON 动作指令。
*   **👁️ 看 (Look)**: 结合 **YOLOv8** 视觉识别与手眼标定系统，实现精准的物体定位。
*   **💪 动 (Move)**: 自研 **D-H 逆运动学算法** 与 S-Curve 速度规划，保证机械臂在 ESP32 控制下的平滑运动。

---

## 🛠️ 技术栈总览

| 模块 | 技术方案 | 作用 |
| :--- | :--- | :--- |
| **语音识别** | **Faster-Whisper** (Base) | 离线语音转文本，支持流式输入 |
| **语义理解** | **LLM (DeepSeek/Qwen) + LoRA** | 指令意图解析，泛化复杂语序 |
| **视觉感知** | **YOLOv8s** + OpenCV | 目标检测与坐标映射 (Homography) |
| **运动控制** | **Python (IK)** + **ESP32 (C++)** | 逆解算与底层 PWM 舵机驱动 |
| **训练框架** | **LLaMA-Factory** | 高效微调大模型指令跟随能力 |

---

# 📖 项目复刻指南 (Replication Guide)

本指南详细介绍了如何从零开始复刻本项目，包括硬件准备、环境搭建、以及**最关键的三个AI模型（语音、视觉、大脑）的获取与训练方法**。

## 1. 硬件准备 (Hardware)

*   **机械臂**: 也就是本项目中的 `RobotArmUltimate`。
    *   要求：支持串口通信（Serial），使用标准舵机控制协议。
    *   连接：USB连接电脑，需确认串口号（代码默认为 `COM3`，请在 `arm_main.py` 或 `voice_main.py` 中修改）。
*   **摄像头**: USB免驱网络摄像头。
    *   安装位置：固定在机械臂前方或上方，确保能覆盖工作台面。
*   **麦克风**: 任意USB麦克风或电脑内置麦克风。
*   **计算设备**: 建议配备 NVIDIA 显卡的 Windows/Linux 电脑（用于加速 YOLO 和 LLM 推理）。

### 1.1 固件烧录 (Firmware)
本项目包含下位机控制代码 `main.ino`，适用于 ESP32 开发板。
*   **开发环境**: Arduino IDE 2.x
*   **开发板管理器**: ESP32 by Espressif Systems (建议版本 3.0.0+)
*   **烧录步骤**:
    1.  使用 USB 数据线连接 ESP32 到电脑。
    2.  打开 `main.ino` 文件。
    3.  选择开发板型号（如 "ESP32 Dev Module"）和端口。
    4.  上传代码。
    5.  记下端口号（如 `COM3`），后续需在 `voice_main.py` 中配置。

## 2. 软件环境搭建 (Software)

### 2.1 基础环境
1.  安装 **Python 3.10+**。
2.  安装 **CUDA** (如果你有NVIDIA显卡)，建议版本 11.8 或 12.x，以便使用 `torch` 的 GPU 版本。
3.  克隆本项目代码。

### 2.2 依赖安装
请在终端运行以下命令安装所需库：

```bash
# 基础工具
pip install numpy opencv-python pyserial sounddevice scipy

# AI 模型相关 (PyTorch, Ultralytics, Transformers)
# 注意：PyTorch 请去官网 https://pytorch.org/ 根据你的 CUDA 版本安装对应的命令
pip install torch torchvision torchaudio 

# 视觉与大模型
pip install ultralytics transformers accelerate peft bitsandbytes

# 语音识别
pip install openai-whisper
```

## 3. 三大核心模型获取与训练指南 (Model Training)

本项目包含三个核心 AI 模块，请分别按照以下步骤准备。

### 3.1 👂 语音听觉 (Whisper)
*   **作用**: 将你的语音指令转为文字。
*   **获取方法**: 
    *   **无需训练**。代码使用了 OpenAI 的 `whisper` 模型。
    *   首次运行时，程序会自动下载模型权重（如 `base` 或 `small` 模型）。
    *   代码位置：`whisper_main.py` 中的 `RobotEar` 类。

### 3.2 👁️ 视觉感知 (YOLOv8)
*   **作用**: 识别桌面上的物体（如：削笔刀、盒子、零件）并定位其像素坐标。
*   **获取方法**: **需要训练** (Custom Training)。
*   **详细步骤**:
    1.  **数据采集**: 
        *   打开摄像头，拍摄你的桌面上不同摆放位置的物体图片（建议 100-300 张）。
    2.  **数据标注**:
        *   使用 `LabelImg` 或 `Roboflow` 等工具进行标注。
        *   类别名称必须与可以被语音识别到的名称对应（如：`part`, `box` 等）。
        *   *注意：本项目目前默认将所有目标映射为 `part` 进行抓取，但训练时建议区分不同类别。*
    3.  **模型训练**:
        *   确保你安装了 `ultralytics`。
        *   准备 `data.yaml` 文件，指定 `train` 和 `val` 图片路径及类别名称。
        *   运行训练命令：
            ```bash
            yolo detect train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640
            ```
    4.  **模型部署**:
        *   训练完成后，会在 `runs/detect/train/weights/` 下生成 `best.pt`。
        *   将 `best.pt` 复制到项目根目录，并在 `voice_main.py` 中修改加载路径：`self.model = YOLO('best.pt')`。

### 3.3 🧠 逻辑大脑 (LLM + LoRA)
*   **作用**: 将自然语言（例如“把那个红色的块拿起来”）翻译成机器能读懂的 JSON 指令（`{"action": "pick", ...}`）。
*   **获取方法**: **基于开源大模型进行微调 (Fine-tuning)**。
*   **详细步骤**:
    1.  **基座模型准备**:
        *   建议下载 Qwen1.5-1.8B, Llama-3-8B 或 ChatGLM3-6B 等适合本地运行的模型。
    2.  **构建数据集**:
        *   参考项目中的 `robot_train.json` 文件。
        *   格式（Alpaca 格式）：
            ```json
            [
              {
                "instruction": "向左移动一厘米",
                "input": "",
                "system": "你是机械臂JSON转换器...",
                "output": "[{\"action\": \"move_inc\", \"axis\": \"y\", \"value\": 10}]"
              }
            ]
            ```
        *   你需要编写大量类似的 "中文指令 -> JSON" 对照数据，覆盖抓取、移动、摇头等场景。
    3.  **微调 (Fine-tuning)**:
        *   本项目集成了 **LLaMA-Factory** 框架（见 `LLaMA-Factory/` 目录）。
        *   使用 LLaMA-Factory 进行 LoRA 微调：
            ```bash
            cd LLaMA-Factory
            # 示例微调命令 (需根据实际显存调整参数)
            llamafactory-cli train \
                --stage sft \
                --do_train \
                --model_name_or_path /path/to/base_model \
                --dataset robot_train \
                --template qwen \
                --finetuning_type lora \
                --output_dir ../saves/lora_adapter \
                --per_device_train_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --lr_scheduler_type cosine \
                --logging_steps 10 \
                --save_steps 100 \
                --learning_rate 5e-5 \
                --num_train_epochs 5.0
            ```
    4.  **模型加载**:
        *   训练完成后，你将获得一个 LoRA 权重文件夹（如 `saves/lora_adapter`）。
        *   在 `voice_main.py` 的 `RobotBrain` 类中，将 `model_path` 指向你的 LoRA 文件夹路径（代码中默认为 `D:\lora\2`）。
        *   *代码不仅加载了 LoRA，还通过 `AutoModelForCausalLM` 自动合并加载了基座模型（前提是 LoRA 的配置文件里记录了基座模型路径）。*

## 4. 运行与标定 (Run & Calibration)

1.  **连接硬件**: 插入摄像头和机械臂 USB。
2.  **启动程序**:
    ```bash
    python voice_main.py
    ```
3.  **手眼标定 (Hand-Eye Calibration)**:
    *   **无论摄像头怎么动，都需要重新标定**。
    *   在程序运行画面中，按下键盘 **`C`** 键进入标定模式。
    *   此时画面会提示依次点击 4 个点（左上、右上、右下、左下）。
    *   请用鼠标在画面中点击机械臂实际能够到达的这 4 个对应的矩形区域角点（对应机械臂坐标 `(90,90), (200,90), (200,-90), (90,-90)`）。
    *   点击完第 4 个点后，系统会自动计算变换矩阵，至此标定完成。

## 5. 使用方法
*   按住 **空格键** 说话（如：“把那个零件拿起来”，“向左两厘米”）。
*   松开空格键，机械臂将自动执行动作。
*   更多快捷键和指令说明请参考 `使用说明书.md`。
