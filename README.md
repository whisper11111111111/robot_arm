# 🤖 语音控制具身智能机械臂 (Voice-Controlled Embodied AI Robot Arm)

> 🚀 **桌面级具身智能 (Embodied AI) 最佳实践**：构建“耳-脑-眼-手”全链路闭环控制系统。

本项目实现了一套运行在消费级笔记本（RTX 3060, 6GB）上的**全栈离线具身智能系统**。通过多模态模型融合，打通了从自然语言到物理动作的最后壁垒：
*   **听 (Listen)**：集成 **Faster-Whisper**，支持抗噪中文语音指令输入；
*   **想 (Think)**：部署微调后的 **DeepSeek/Qwen 大语言模型**，具备极强的复杂语义理解与逻辑推理能力；
*   **看 (See)**：结合 **YOLOv8 机器视觉** 与单应性矩阵手眼标定，实现亚毫米级目标定位；
*   **动 (Act)**：自研 **D-H 逆运动学** 求解器与平滑轨迹规划算法，精准驱动机械臂执行抓取、搬运等任务。

无需联网，低成本复刻，满足边缘计算与隐私安全需求。

> **关键词**: `Embodied AI`, `Robot Arm`, `Voice Control`, `LLM`, `DeepSeek`, `YOLOv8`, `Whisper`, `Fine-tuning`, `Inverse Kinematics`

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

## � 项目文件结构

```
.
├── main.py              # 主程序 — 语音/视觉/控制全链路调度
├── arm_main.py          # 机械臂驱动 — D-H 运动学、S-Curve 轨迹、串口通信
├── whisper_main.py      # 语音识别 — Faster-Whisper 封装（RobotEar 类）
├── config.py            # 全局配置 — 路径、串口、标定参数（改这里即可）
├── main.ino             # ESP32 下位机固件 (Arduino C++)
├── best.pt              # YOLOv8 自训练权重（不含于仓库，需自行训练）
├── requirements.txt     # Python 依赖清单
├── LLaMA-Factory/       # 大模型微调框架
├── model/               # YOLO 训练脚本与数据集
├── DESIGN.md            # 系统架构与核心算法技术文档
└── LICENSE
```

> 深度技术细节（架构图、D-H 建模、LLM 微调参数、问题复盘）请参阅 **[DESIGN.md](DESIGN.md)**。

---

## �📖 项目复刻指南 (Replication Guide)

本指南详细介绍了如何从零开始复刻本项目，包括硬件准备、环境搭建、以及**最关键的三个AI模型（语音、视觉、大脑）的获取与训练方法**。

## 1. 硬件准备 (Hardware)

### 1.1 项目物料清单 (BOM) & 成本

本项目硬件成本极低，总花费约 **¥317**。以下是基于实际采购发票的详细清单：

| 序号 | 物品名称 | 规格/型号 | 数量 | 单价 (CNY) | 总费用 (CNY) | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3D打印机械臂 | 教具/机械臂 (散件) | 1 个 | 71.00 | 71.00 | 包含亚克力/PLA结构件 |
| 2 | ESP32开发板 | WiFi+蓝牙双核 MCU | 1 件 | 18.71 | 18.71 | 主控核心 |
| 3 | ESP32配件 | (接插件/扩展板) | 1 件 | 4.55 | 4.55 | 辅助连接 |
| 4 | 工业摄像头 | USB免驱 / 广角 | 1 个 | 61.00 | 61.00 | 机器视觉输入 |
| 5 | 数字舵机 | MG996R (金属齿轮) | 5 个 | 26.54 | 132.70 | 大扭矩驱动 |
| 6 | 稳压电源 | 6V 6A | 1 个 | 29.00 | 29.00 | 舵机供电 |
| **总计** | | | | | **¥316.96** | **高性价比** |

### 1.2 硬件连接说明

*   **机械臂**: 也就是本项目中的 `RobotArmUltimate`。
    *   要求：支持串口通信（Serial），使用标准舵机控制协议。
    *   连接：USB 连接电脑，需确认串口号（默认 `COM3`，在 `config.py` 中修改 `SERIAL_PORT`）。
*   **摄像头**: USB免驱网络摄像头。
    *   安装位置：固定在机械臂前方或上方，确保能覆盖工作台面。
*   **麦克风**: 任意USB麦克风或电脑内置麦克风。
*   **计算设备**: 建议配备 NVIDIA 显卡的 Windows/Linux 电脑（用于加速 YOLO 和 LLM 推理）。

### 1.3 固件烧录 (Firmware)
本项目包含下位机控制代码 `main.ino`，适用于 ESP32 开发板。
*   **开发环境**: Arduino IDE 2.x
*   **开发板管理器**: ESP32 by Espressif Systems (建议版本 3.0.0+)
*   **烧录步骤**:
    1.  使用 USB 数据线连接 ESP32 到电脑。
    2.  打开 `main.ino` 文件。
    3.  选择开发板型号（如 "ESP32 Dev Module"）和端口。
    4.  上传代码。
    5.  记下端口号（如 `COM3`），后续在 `config.py` 中修改 `SERIAL_PORT` 即可。

## 2. 软件环境搭建 (Software)

### 2.1 基础环境
1.  安装 **Python 3.10+**。
2.  安装 **CUDA** (如果你有NVIDIA显卡)，建议版本 11.8 或 12.x，以便使用 `torch` 的 GPU 版本。
3.  克隆本项目代码。

### 2.2 依赖安装

**推荐使用 conda 隔离环境（避免依赖冲突）：**

```bash
conda create -n robot_arm python=3.10 -y
conda activate robot_arm
```

请在终端运行以下命令安装所需库：

```bash
# 基础工具
pip install numpy opencv-python pyserial sounddevice scipy

# AI 模型相关 (PyTorch)
# 注意：PyTorch 请去官网 https://pytorch.org/ 根据你的 CUDA 版本选择对应命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 视觉与大模型
pip install ultralytics transformers accelerate safetensors

# 语音识别（使用 faster-whisper，非 openai-whisper）
pip install faster-whisper
```

或直接使用项目中的 `requirements.txt`：

```bash
pip install -r requirements.txt
```

## 3. 三大核心模型获取与训练指南 (Model Training)

本项目包含三个核心 AI 模块，请分别按照以下步骤准备。

### 3.1 👂 语音听觉 (Whisper)
*   **作用**: 将你的语音指令转为文字。
*   **获取方法**: 
    *   **无需训练**。代码使用 **Faster-Whisper** 进行本地推理。
    *   首次运行时，程序会自动从 HuggingFace 下载对应规格的模型权重（受网络影响，可提前手动下载）。
    *   模型规格在 `config.py` 中通过 `WHISPER_MODEL_SIZE` 设置（默认 `base`，追求速度可改 `tiny`）。

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
        *   将 `best.pt` 复制到项目根目录（默认路径），或在 `config.py` 中修改 `YOLO_MODEL_PATH` 指向实际路径。

### 3.3 🧠 逻辑大脑 (LLM + LoRA)
*   **作用**: 将自然语言（例如“把那个红色的块拿起来”）翻译成机器能读懂的 JSON 指令（`{"action": "pick", ...}`）。
*   **获取方法**: **基于开源大模型进行微调 (Fine-tuning)**。
*   **详细步骤**:
    1.  **基座模型准备**:
        *   本项目使用 **DeepSeek-R1-Distill-Qwen-1.5B** 作为基座（显存占用 ~3.3 GB FP16）。
        *   从 [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) 或 [ModelScope](https://modelscope.cn/) 下载模型权重。
        *   也可替换为同参数量级的其他模型（如 Qwen2.5-1.5B-Instruct），但需重新训练。
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
        *   在 `config.py` 中修改 `LLM_MODEL_PATH` 指向你的 LoRA 导出文件夹（默认为 `D:\lora\2`）。
        *   *代码通过 `AutoModelForCausalLM` 自动合并基座模型与 LoRA 权重，前提是 LoRA 配置文件中记录了基座模型路径。*
        *   **注意**：若系统页面文件不足（Windows），加载 3GB+ 模型时代码会自动切换为无 mmap 的流式加载模式（终端输出"切换直接 GPU 加载模式"属正常现象）。

## 4. 运行与标定 (Run & Calibration)

1.  **连接硬件**: 插入摄像头和机械臂 USB。
2.  **激活环境**:
    ```bash
    conda activate robot_arm
    ```
3.  **修改配置**（可选）：打开 `config.py`，按需修改 `SERIAL_PORT`（串口号）、`LLM_MODEL_PATH`（模型路径）等参数。
4.  **启动程序**:
    ```bash
    python main.py
    ```
5.  **手眼标定**：首次使用请参考 [5.3 节](#53-手眼标定c-键) 完成四点标定（摄像头位置变动后需重新标定）。

## 5. 使用方法

### 5.1 快捷键

| 按键 | 功能 |
| :--- | :--- |
| **SPACE（空格）** | 按住开始录音，松开结束并识别指令 |
| **C** | 进入 / 退出四点手眼标定模式 |
| **R** | 手动复位到初始待机位置 (120, 0, 60) |
| **O** | 手动松开夹爪 |
| **Q** | 安全退出程序（释放摄像头、显存、串口） |

### 5.2 支持的语音指令

| 类型 | 示例 | 说明 |
| :--- | :--- | :--- |
| **抓取** | "把削笔刀抓起来"、"抓住那个盒子" | 视觉定位后自动抓取 |
| **抬起** | "把削笔刀抬起5厘米"、"将零件举高10公分" | "公分"自动换算，支持中文数字 |
| **方向移动** | "向上三厘米"、"往前伸10厘米" | 未指定距离时默认 5 厘米 |
| **点头 / 摇头** | "点头"、"摇头" | 以 3 cm 幅度往复运动三次 |
| **放下** | "放下"、"放到桌面上" | 自动计算高度降落至桌面并松爪 |
| **复位** | "复位"、"回到原点"、"归位" | 返回初始安全姿态 |

> **提示**：系统支持谐音纠错（"零米"→"厘米"、"电头"→"点头"）和防幻觉过滤（自动去除"向右向右向右..."等重复）。

### 5.3 手眼标定（C 键）

#### 原理说明

**为什么需要手眼标定？**

摄像头输出的是**像素坐标系**（以图像左上角为原点，单位：像素），而机械臂运动使用的是**机器人坐标系**（以底座为原点，单位：mm）。两个坐标系之间存在旋转、缩放、平移等复合变换，没有固定公式可以直接转换，因此需要通过"标定"来测量这一变换关系。

**数学原理：单应性矩阵（Homography）**

本项目采用**平面单应性变换**来完成坐标映射。由于工作台面是一个平面，摄像头视角与该平面之间存在一个确定的投影变换，可以用一个 $3 \times 3$ 的齐次矩阵 $H$ 来描述：

$$
\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix} = H \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}, \quad H = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix}
$$

最终机器人坐标为：

$$r_x = \frac{x'}{w'}, \quad r_y^{\text{raw}} = \frac{y'}{w'}$$

由于摄像头画面存在**水平镜像**，还需要对 $r_y$ 做补偿修正：

$$r_y = 2 \times C_{\text{center}} - r_y^{\text{raw}}$$

其中 $C_{\text{center}}$ 是标定区域在机器人坐标系下的 $y$ 轴中心值（由 `config.py` 中 `CALIB_CENTER_Y` 定义）。

**为什么需要 4 个点？**

求解矩阵 $H$ 的 8 个独立参数（$h_{33}$ 归一化为 1）至少需要 **4 组对应点**，每对点提供 2 个方程，恰好凑够 8 个约束。本项目使用 `cv2.findHomography` 由 4 个角点自动求解。

**完整的标定流程（代码层面）**：

```
用户点击 4 个像素点 → image_points (4×2)
对应已知机器人坐标 → robot_points (4×2)
        ↓
cv2.findHomography(image_points, robot_points)
        ↓
生成 H 矩阵 (3×3)，存入 self.H
        ↓
检测到目标像素坐标 (u, v)
        ↓
pixel_to_robot(u, v)  →  机器人坐标 (rx, ry)
```

#### 完整操作流程

手眼标定分为 **三个阶段**，首次部署时必须按顺序全部执行；摄像头位置变动后仅需重做第三阶段。

---

##### 阶段一：物理标记（一次性，离线完成）

1. 在工作台面上用记号笔、彩色胶带或贴纸**标出 4 个点**，建议选取桌面的 4 个角落，面积越大映射精度越高。
2. 用直尺粗略确认 4 个点的相对位置关系（不需要精确量距，只需清晰可见）。
3. 记下你计划赋予这 4 个点的**机器人坐标**（单位 mm，以机械臂底座为原点，X 前，Y 左，Z 上），建议设置为矩形区域四角，例如：
   ```
   P1（左上）: x=90,  y=90
   P2（右上）: x=200, y=90
   P3（右下）: x=200, y=-90
   P4（左下）: x=90,  y=-90
   ```
4. 将这 4 组坐标填入 `config.py` 的 `CALIB_ROBOT_POINTS`：
   ```python
   CALIB_ROBOT_POINTS = [
       [90,  90],   # P1 左上
       [200, 90],   # P2 右上
       [200, -90],  # P3 右下
       [90,  -90],  # P4 左下
   ]
   ```

---

##### 阶段二：机械臂到位并记录坐标（使用 `calibrate.py`）

此阶段目的是验证第一阶段设定的坐标**与机械臂实际能到达的位置是否吻合**，并在必要时修正。

```bash
python calibrate.py
```

打开 GUI 后，对每个物理标记点执行以下操作：

1. **手动控制机械臂移动**：使用界面上的方向按钮（或键盘快捷键）将夹爪尖端**精确悬停在台面标记点的正上方**（Z 轴降至接近桌面，使尖端刚好触碰或接近标记点）。

   | 界面按钮 / 快捷键 | 动作 |
   | :--- | :--- |
   | ▲ / ▼ 或 **W / S** | X 轴 前进 / 后退 |
   | ◀ / ▶ 或 **A / D** | Y 轴 左移 / 右移 |
   | ↑ / ↓ 或 **Q / E** | Z 轴 上升 / 下降 |
   | 步长输入框 | 调整单次移动步长（默认 5 mm） |
   | **移动到该坐标** 按钮 | 直接跳转至输入的 XYZ 坐标 |

2. **按下"记录当前位置"按钮**：程序将此时的 `(x, y)` 坐标追加到右侧列表，共记录 4 次。

3. 全部记录后点击**"完成"**，终端会输出 4 组实测坐标，将其**更新到 `config.py` 的 `CALIB_ROBOT_POINTS`** 中（如与预设值偏差不大可不做修改）。

> **提示**：标定点分布越均匀、覆盖面积越大，全区域的映射精度越高。

---

##### 阶段三：相机像素点匹配（在 `main.py` 中进行）

此步骤将第一阶段的物理标记点与摄像头拍到的像素坐标进行对应，建立 $H$ 矩阵。

1. 启动主程序：
   ```bash
   python main.py
   ```
2. 确认摄像头画面正常显示，且台面上 4 个物理标记点**清晰可见**。
3. 按 **C 键**，画面左上角出现 `CALIBRATION MODE`，此时等待鼠标输入。
4. 按照 `CALIB_ROBOT_POINTS` 中**相同的顺序**（P1 → P2 → P3 → P4），依次用鼠标**左键单击**摄像头画面中各标记点的像素位置。
5. 点完第 4 个点后，程序自动调用 `cv2.findHomography` 计算矩阵，**立即生效**（仅本次运行有效，不写入 config.py）。

> **注意**：若需要永久保存像素坐标，可在终端中查看程序输出的 `新标定点` 坐标，手动复制到 `config.py` 的 `CALIB_IMAGE_POINTS`。之后摄像头未移动时，重启程序无需重新点击。

---

## ⚙️ 配置说明

### config.py — 运行参数

所有可调参数均集中在 `config.py`，**无需修改其他代码文件**：

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `LLM_MODEL_PATH` | `D:\lora\2` | 微调后的大模型路径（**必改为实际路径**） |
| `SERIAL_PORT` | `COM3` | ESP32 串口号 |
| `YOLO_MODEL_PATH` | `best.pt` | 目标检测模型路径 |
| `CAMERA_INDEX` | `0` | 摄像头编号（多摄像头时可改为 1、2） |
| `WHISPER_MODEL_SIZE` | `base` | Whisper 规格（`tiny` / `base` / `small`） |
| `ARM_REST_POSITION` | `[120, 0, 60]` | 机械臂初始待机位置，单位 mm |
| `CALIB_ROBOT_POINTS` | 见文件 | 4 个标定点的机器人坐标，手眼标定后更新 |
| `CALIB_IMAGE_POINTS` | 见文件 | 4 个标定点的像素坐标，手眼标定后更新 |

### arm_main.py — DH 连杆参数（⚠️ 必须按实际测量修改）

`arm_main.py` 第 42 行硬编码了机械臂的 **D-H 连杆长度**，这是逆运动学求解的基础参数，**数值必须与你的实际机械臂完全一致**，否则末端执行器的实际位置将与计算位置产生偏差，导致抓取失败。

```python
# arm_main.py  第 42 行
self.L1, self.L2, self.L3, self.L4 = 70.0, 75.0, 50.0, 130.0
```

各参数的物理含义：

| 参数 | 默认值 (mm) | 物理含义 |
| :--- | :--- | :--- |
| `L1` | 70.0 | 底部升降关节到肩关节的**垂直高度**（基座高度） |
| `L2` | 75.0 | 肩关节到肘关节的**大臂长度** |
| `L3` | 50.0 | 肘关节到腕关节的**小臂长度** |
| `L4` | 130.0 | 腕关节到夹爪尖端的**末端执行器长度**（含夹爪） |

**测量方法**：在机械臂伸直（各关节角度为 0）的状态下，用卡尺或直尺逐段测量相邻舵机转轴之间的**轴心距离**。末端 `L4` 需量到夹爪闭合时尖端的位置。

---

## ❗ 常见问题排查

| 问题现象 | 解决方法 |
| :--- | :--- |
| 按空格没反应 | 点击摄像头画面窗口，确保程序焦点在窗口上 |
| 语音识别乱码 | 在安静环境下说话，按住空格等 0.5s 再开口，语速适中 |
| 「未找到目标」 | 调整物体摆放角度，确保光照充足；或该物体不在 YOLO 训练类别中 |
| 抓取位置偏离 | 摄像头被移动过，按 **C 键**重新四点标定 |
| `No module named 'sounddevice'` | 确认已激活 `robot_arm` conda 环境后运行 |
| Windows 弹窗「Python 已停止工作」 | 系统页面文件不足；代码已内置自动回退，重试即可；如持续出现请将页面文件扩大至 8 GB+ |
| 终端输出「切换直接 GPU 加载模式」 | 正常现象，表示绕过了内存映射限制，加载约 30s，不影响功能 |
| 串口连接失败 | 在设备管理器确认 COM 端口号，修改 `config.py` 中的 `SERIAL_PORT` |
| 摄像头画面打不开 | 将 `config.py` 中 `CAMERA_INDEX` 改为 `1` 或 `2` |