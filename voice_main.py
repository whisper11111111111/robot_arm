import cv2
import numpy as np
import time
import os
import re
import json
import torch
import sounddevice as sd
from transformers import AutoModelForCausalLM, AutoTokenizer
from ultralytics import YOLO  
from arm_main import RobotArmUltimate 
from whisper_main import RobotEar 

# 禁用代理
os.environ["no_proxy"] = "localhost,127.0.0.1"

# =========================================================
# 1. 标定数据（与 yolo_main.py 保持一致）
# =========================================================
robot_points = np.array([
    [90, 90], [200, 90], [200, -90], [90, -90]
], dtype=np.float32)
initial_image_points = [[817, 72], [433, 79], [291, 612], [1029, 610]]
CALIB_CENTER_Y = 0.0

# =========================================================
# 2. 大模型指令解析器（优化提示词）
# =========================================================
class RobotBrain:
    def __init__(self, model_path=r"D:\lora\2"):
        self.model_path = model_path
        print(f">>> [大脑] 正在加载模型: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        # 【关键】system prompt 必须和训练数据一致
        self.system_prompt = "你是机械臂JSON转换器。一个指令=一个动作！\n规则：\n1. 单位：厘米×10=毫米\n2. 轴向：上/下=z, 左/右=y, 前/后=x\n3. 正负：上/左/前=正, 下/右/后=负\n4. 目标：所有物体映射为 \"part\"\n5. 只输出JSON数组。\n6. 示例：摇头即输出 [{\"action\": \"shake_head\"}]"

    def think(self, user_text):
        """将自然语言转换为JSON指令列表"""
        # 【优化】简单方向/松开/复位指令，直接走可靠的规则解析器，不经过大模型
        # 避免模型把"向下三厘米"误判为 lift
        simple_result = self._try_simple_parse(user_text)
        if simple_result:
            print(f">>> [规则解析] 命中简单指令: {simple_result}")
            return simple_result
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            text = f"{text}<｜Assistant｜>"

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f">>> [大模型原始输出] {content}")
            
            # 【修复】更精确的正则：只匹配第一个完整的JSON数组或对象
            match = re.search(r'\[\s*\{[^[\]]*\}\s*\]', content, re.DOTALL)
            if not match:
                match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', content, re.DOTALL)
            if not match:
                match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            
            if match:
                json_str = match.group()
                
                # 清理JSON字符串
                json_str = re.sub(r'[。！？\.\!\?]\s*$', '', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])', r':"\1"\2', json_str)
                json_str = json_str.strip()
                
                print(f">>> [清理后JSON] {json_str}")
                
                res = json.loads(json_str)
                result = res if isinstance(res, list) else [res]
                
                # 【新增】过滤无效动作，并强制只保留一个动作
                valid_actions = ["pick", "lift", "move_inc", "release", "drop", "reset"]
                filtered = [cmd for cmd in result if cmd.get("action") in valid_actions]

                if filtered:
                    # 若包含 lift，始终只保留 lift
                    for cmd in filtered:
                        if cmd.get("action") == "lift":
                            print(f">>> [过滤] 仅保留 lift 动作: {cmd}")
                            return [cmd]

                    # 否则只保留第一个有效动作
                    print(f">>> [过滤] 仅保留第一个有效动作: {filtered[0]}")
                    return [filtered[0]]
                
        except json.JSONDecodeError as e:
            print(f">>> [JSON解析错误] {e}")
        except Exception as e:
            print(f">>> [大脑错误] {e}")
        
        # 备用解析
        return self._fallback_parse(user_text)
    
    def _try_simple_parse(self, text):
        """【优化】对简单的方向/松开/复位指令直接规则匹配，不经过大模型
        返回 None 表示不是简单指令，需要交给大模型处理"""
        
        # 松开/释放
        if re.search(r'(松开|放开|释放|松手)', text):
            return [{"action": "release"}]
        
        # 复位
        if re.search(r'(复位|回到原位|归位|回原点|回原位|回到原点)', text):
            return [{"action": "reset"}]
        
        # 放下 (【优先处理】避免被误判为"向下")
        if re.search(r'(放下|放到|放置)', text):
            return [{"action": "drop"}]

        # 点头/摇头
        if re.search(r'(点头)', text):
            return [{"action": "nod"}]
        if re.search(r'(摇头)', text):
            return [{"action": "shake_head"}]
        
        # 方向+数值 的纯移动指令（不含物体名称时才走这里）
        has_object = re.search(r'(削笔刀|盒子|物块|零件|瓶子|part)', text)
        if has_object:
            return None  # 含物体名，交给大模型判断 lift/pick
        
        # 提取数值和单位（支持中文数字）
        def extract_value(match, num_group=1, unit_group=2):
            val_str = match.group(num_group)
            
            # 中文数字映射
            cn_map = {'一':1, '二':2, '两':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}
            if val_str in cn_map:
                value = cn_map[val_str]
            else:
                try:
                    value = int(val_str)
                except:
                    value = 5 # 默认值
            
            unit = match.group(unit_group) or '厘米'
            if '毫米' in unit or 'mm' in unit or '米米' in unit: # 兼容"米米"听写错误
                return value
            return value * 10  # 厘米转毫米
        
        # 正则表达式：匹配数字 OR 中文数字 [0-9一二...]
        num_pattern = r'(\d+|[一二两三四五六七八九十])'
        
        # 上下
        m = re.search(f'向?上.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "z", "value": extract_value(m)}]
        m = re.search(f'向?下.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "z", "value": -extract_value(m)}]
        
        # 左右
        m = re.search(f'向?左.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "y", "value": extract_value(m)}]
        m = re.search(f'向?右.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "y", "value": -extract_value(m)}]
        
        # 前后
        m = re.search(f'(向?前|往前).{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "x", "value": extract_value(m, 2, 3)}]
        m = re.search(f'(向?后|往后).{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "x", "value": -extract_value(m, 2, 3)}]

        
        # ==========================================
        # 【新增】空手模糊指令（无数值，使用默认值 5cm）
        # ==========================================
        DEFAULT_MOVE = 50.0  # 默认移动 50mm
        
        # 向上/抬起
        if re.search(r'(向?上|抬起|举起|往上)', text):
            return [{"action": "move_inc", "axis": "z", "value": DEFAULT_MOVE}]
        # 向下
        if re.search(r'(向?下|往下)', text):
            return [{"action": "move_inc", "axis": "z", "value": -DEFAULT_MOVE}]
            
        # 向左
        if re.search(r'(向?左|往左)', text):
            return [{"action": "move_inc", "axis": "y", "value": DEFAULT_MOVE}]
        # 向右
        if re.search(r'(向?右|往右)', text):
            return [{"action": "move_inc", "axis": "y", "value": -DEFAULT_MOVE}]
            
        # 向前
        if re.search(r'(向?前|往前)', text):
            return [{"action": "move_inc", "axis": "x", "value": DEFAULT_MOVE}]
        # 向后
        if re.search(r'(向?后|往后)', text):
            return [{"action": "move_inc", "axis": "x", "value": -DEFAULT_MOVE}]
        
        return None  # 不是简单指令

    def _fallback_parse(self, text):
        """备用解析：当大模型输出无法解析时，尝试直接从原文提取意图"""
        print(">>> [启用备用解析器]")
        cmds = []
        
        # 【优先1】检测纯方向移动指令（无目标物体）
        # 向上/向下
        up_match = re.search(r'向?上.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        down_match = re.search(r'向?下.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        # 向左/向右
        left_match = re.search(r'向?左.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        right_match = re.search(r'向?右.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        # 向前/向后
        forward_match = re.search(r'(向?前|往前).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        backward_match = re.search(r'(向?后|往后).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        
        # 处理上下移动
        if up_match:
            value = int(up_match.group(1))
            unit = up_match.group(2) or '厘米'
            if '毫米' not in unit and 'mm' not in unit:
                value *= 10
            cmds.append({"action": "move_inc", "axis": "z", "value": value})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        if down_match:
            value = int(down_match.group(1))
            unit = down_match.group(2) or '厘米'
            if '毫米' not in unit and 'mm' not in unit:
                value *= 10
            cmds.append({"action": "move_inc", "axis": "z", "value": -value})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 处理左右移动
        if left_match:
            value = int(left_match.group(1))
            unit = left_match.group(2) or '厘米'
            if '毫米' not in unit and 'mm' not in unit:
                value *= 10
            cmds.append({"action": "move_inc", "axis": "y", "value": value})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        if right_match:
            value = int(right_match.group(1))
            unit = right_match.group(2) or '厘米'
            if '毫米' not in unit and 'mm' not in unit:
                value *= 10
            cmds.append({"action": "move_inc", "axis": "y", "value": -value})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 处理前后移动
        if forward_match:
            value = int(forward_match.group(2))
            unit = forward_match.group(3) or '厘米'
            if '毫米' not in unit and 'mm' not in unit:
                value *= 10
            cmds.append({"action": "move_inc", "axis": "x", "value": value})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        if backward_match:
            value = int(backward_match.group(2))
            unit = backward_match.group(3) or '厘米'
            if '毫米' not in unit and 'mm' not in unit:
                value *= 10
            cmds.append({"action": "move_inc", "axis": "x", "value": -value})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 【优先2】检测"抬起X物体N厘米"（需要目标）
        lift_pattern1 = re.search(r'(把|将)?.{0,6}?(削笔刀|盒子|物块|零件|part).{0,4}?(抬起|拿起|举起).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)', text)
        lift_pattern2 = re.search(r'(抬起|拿起|举起).{0,6}?(削笔刀|盒子|物块|零件|part).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)', text)
        
        if lift_pattern1:
            height = int(lift_pattern1.group(4))
            unit = lift_pattern1.group(5)
            if '厘米' in unit or 'cm' in unit:
                height *= 10
            cmds.append({"action": "lift", "target": "part", "height": height})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        if lift_pattern2:
            height = int(lift_pattern2.group(3))
            unit = lift_pattern2.group(4)
            if '厘米' in unit or 'cm' in unit:
                height *= 10
            cmds.append({"action": "lift", "target": "part", "height": height})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 检测单纯抓取（需要目标）
        if re.search(r'(抓住|抓取|拿住|夹住).{0,6}?(削笔刀|盒子|物块|零件|part)', text):
            cmds.append({"action": "pick", "target": "part"})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 检测松开
        if re.search(r'(松开|放开|释放|松手)', text):
            cmds.append({"action": "release"})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 检测放下
        if re.search(r'(放下|放到|放置)', text):
            cmds.append({"action": "drop"})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        # 检测复位
        if re.search(r'(复位|回到原位|归位|回原点|回原位)', text):
            cmds.append({"action": "reset"})
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        
        if cmds:
            print(f">>> [备用解析结果] {cmds}")
            return cmds
        return None


# =========================================================
# 3. 视觉抓取系统（支持目标筛选 + 标定功能）
# =========================================================
class AutoGraspSystem:
    def __init__(self):
        print("\n>>> [1/3] 正在启动机械臂...")
        self.arm = RobotArmUltimate(port='COM3')
        self.arm.set_damping_params(buffer_size=3, max_speed=25.0, damping_factor=0.6)
        self.arm.set_correction(offset_y=-10.0, offset_z=0.0)
        
        print(">>> [2/3] 正在载入YOLO模型...")
        self.model = YOLO('best.pt')
        
        print(">>> [3/3] 正在初始化摄像头...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # =========== 标定相关变量 ===========
        self.image_points = np.array(initial_image_points, dtype=np.float32)
        self.temp_points = []           # 存储重标定时的临时点击
        self.is_calibrating = False     # 标定状态开关
        self.update_homography()        # 初始计算一次矩阵
        
        # 当前检测到的所有目标 {类名: (rx, ry, 置信度)}
        self.detected_targets = {}
        
        # 夹爪状态
        self.gripper_closed = False
        
        print(">>> [完成] 系统就绪！")

    def update_homography(self):
        """重新计算单应性矩阵H和画面中心显示坐标"""
        self.H, _ = cv2.findHomography(self.image_points, robot_points)
        self.pixel_center_u = int(np.mean(self.image_points[:, 0]))
        self.pixel_center_v = int(np.mean(self.image_points[:, 1]))
        print(f">>> [标定] 矩阵已更新，中心点: ({self.pixel_center_u}, {self.pixel_center_v})")

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击处理：仅在标定模式下记录点"""
        if self.is_calibrating and event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append([x, y])
            print(f" -> 已记录标定点 P{len(self.temp_points)}: [{x}, {y}]")
            
            if len(self.temp_points) == 4:
                self.image_points = np.array(self.temp_points, dtype=np.float32)
                self.update_homography()
                self.is_calibrating = False
                print(">>> [成功] 标定已更新，切回正常模式。")
                print(f">>> 新标定点: {self.image_points.tolist()}")

    def start_calibration(self):
        """进入标定模式"""
        print("\n>>> 进入重标定模式：请按顺序点击4个标定点")
        print("    P1(左上) -> P2(右上) -> P3(右下) -> P4(左下)")
        print("    对应机器人坐标: (90,90) -> (200,90) -> (200,-90) -> (90,-90)")
        self.temp_points = []
        self.is_calibrating = True

    def pixel_to_robot(self, u, v):
        """像素坐标 → 机器人坐标"""
        vec = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)
        res = np.dot(self.H, vec)
        rx = float(res[0, 0] / res[2, 0])
        ry = (CALIB_CENTER_Y * 2) - float(res[1, 0] / res[2, 0])
        return rx, ry

    def update_detections(self, frame):
        """更新检测结果，返回标注后的帧"""
        # 如果在标定模式，不进行YOLO检测
        if self.is_calibrating:
            return frame
            
        results = self.model.predict(frame, conf=0.3, verbose=False)
        self.detected_targets = {}
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                u, v = (x1 + x2) // 2, (y1 + y2) // 2
                rx, ry = self.pixel_to_robot(u, v)
                
                if cls_name not in self.detected_targets or conf > self.detected_targets[cls_name][2]:
                    self.detected_targets[cls_name] = (rx, ry, conf)
                
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} ({rx:.0f},{ry:.0f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
        
        return frame

    def draw_calibration_ui(self, frame):
        """绘制标定相关的UI元素"""
        if self.is_calibrating:
            # --- 标定模式界面 ---
            cv2.putText(frame, f"CALIBRATION MODE: Click Point P{len(self.temp_points)+1}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 绘制已点击的临时标定点
            for i, pt in enumerate(self.temp_points):
                cv2.circle(frame, tuple(pt), 8, (0, 255, 255), -1)
                cv2.putText(frame, f"P{i+1}", (pt[0]+10, pt[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 提示信息
            hints = [
                "P1: Top-Left (90, 90)",
                "P2: Top-Right (200, 90)",
                "P3: Bottom-Right (200, -90)",
                "P4: Bottom-Left (90, -90)"
            ]
            for i, hint in enumerate(hints):
                color = (0, 255, 0) if i < len(self.temp_points) else (128, 128, 128)
                cv2.putText(frame, hint, (20, 80 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            # --- 正常模式：显示当前标定点 ---
            # 绘制标定区域中心点
            cv2.drawMarker(frame, (self.pixel_center_u, self.pixel_center_v), (0, 255, 255), 
                          markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)
            
            # 绘制4个标定参考点
            for i, pt in enumerate(self.image_points):
                u, v = int(pt[0]), int(pt[1])
                cv2.circle(frame, (u, v), 8, (0, 0, 255), 2)
                cv2.putText(frame, f"P{i+1}", (u + 10, v - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # 绘制标定区域边框（连接4个点）
            pts = self.image_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        
        return frame

    def find_target(self, target_name):
        """根据目标名称查找坐标"""
        if target_name in self.detected_targets:
            rx, ry, _ = self.detected_targets[target_name]
            return rx, ry
        return None

    def execute_pick(self, rx, ry):
        """执行抓取动作"""
        print(f"\n>>> 执行抓取: ({rx:.1f}, {ry:.1f})")
        Z_HOVER = 120.0
        Z_GRAB = -15.0
        Z_AFTER_PICK = 50.0
        
        p_up = np.array([rx, ry, Z_HOVER])
        p_down = np.array([rx, ry, Z_GRAB])
        p_after = np.array([rx, ry, Z_AFTER_PICK])
        
        # 张开夹爪
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.3)
        
        # 【修复】清空缓冲区，确保新动作不受之前状态影响
        self.arm.servo_buffer = []
        
        # 【修复】使用 move_line 从当前位置移动到目标上方（更可靠）
        # 先获取当前位置的近似值
        current_approx = np.array([120.0, 0.0, 60.0])  # 默认休息位置
        self.arm.move_line(current_approx, p_up, p_start=-60, p_end=-60, duration=1.5)
        
        # 下降抓取
        self.arm.move_line(p_up, p_down, p_start=-60, p_end=-90, duration=1.5)
        
        # 闭合夹爪
        self.arm.gripper_control(120)
        self.gripper_closed = True
        time.sleep(0.5)
        
        # 抬起
        self.arm.move_line(p_down, p_after, p_start=-90, p_end=-60, duration=1.0)
        
        return p_after

    def execute_lift(self, rx, ry, height):
        """执行抓取并抬起到指定高度"""
        print(f"\n>>> 执行抬起: 目标({rx:.1f}, {ry:.1f}), 高度{height}mm")
        Z_HOVER = 120.0
        Z_GRAB = -15.0
        
        p_up = np.array([rx, ry, Z_HOVER])
        p_down = np.array([rx, ry, Z_GRAB])
        p_lifted = np.array([rx, ry, Z_GRAB + height])
        p_lifted[2] = np.clip(p_lifted[2], -20, 200)
        
        # 张开夹爪
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.3)
        
        # 【修复】清空缓冲区
        self.arm.servo_buffer = []
        
        # 【修复】使用 move_line 移动到目标上方
        current_approx = np.array([120.0, 0.0, 60.0])
        self.arm.move_line(current_approx, p_up, p_start=-60, p_end=-60, duration=1.5)
        
        # 下降抓取
        self.arm.move_line(p_up, p_down, p_start=-60, p_end=-90, duration=1.5)
        
        # 闭合夹爪
        self.arm.gripper_control(120)
        self.gripper_closed = True
        time.sleep(0.5)
        
        # 抬起到指定高度
        self.arm.move_line(p_down, p_lifted, p_start=-90, p_end=-60, duration=1.5)
        
        print(f"✓ 已抬起到高度: Z={p_lifted[2]:.1f}mm")
        return p_lifted

    def execute_release(self):
        """只松开夹爪"""
        print("\n>>> 执行松开夹爪")
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.5)
        print("✓ 夹爪已松开")

    def execute_drop(self, current_pos):
        """执行放下动作 - 智能高度判断"""
        print("\n>>> 执行动作: 放下 (智能高度控制)")
        
        # 定义桌面高度 (参考 execute_pick 中的 Z_GRAB)
        Z_TABLE = -15.0
        
        # 计算下移距离
        drop_height = current_pos[2] - Z_TABLE
        print(f">>> [高度数据记录] 起始高度: {current_pos[2]:.1f}mm | 目标高度: {Z_TABLE}mm")
        print(f">>> [自动计算] 下移距离: {drop_height:.1f}mm")
        
        # 目标位置：当前XY，但是Z轴降到桌面
        p_drop = np.array([current_pos[0], current_pos[1], Z_TABLE])
        
        # 【修复】清空缓冲区
        self.arm.servo_buffer = []
        
        # 移动到放置点
        # 注意：如果当前已经在桌面高度以下，不要强行移动，或者先抬升? 
        # 这里假设是从上方放下，直接直线移动
        self.arm.move_line(current_pos, p_drop, p_start=-60, p_end=-90, duration=2.0)
        
        # 松开夹爪
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.5)
        
        print("✓ 物体已平稳放置到桌面")
        return p_drop

    def execute_reset(self):
        """复位到初始位置"""
        print("\n>>> 执行复位")
        p_rest = np.array([120, 0, 60])
        
        # 松开夹爪
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.3)
        
        # 【修复】清空缓冲区
        self.arm.servo_buffer = []
        
        # 【修复】使用 move_line 移动（更可靠）
        # 从一个安全的中间位置开始
        p_safe = np.array([120, 0, 100])
        s_safe = self.arm.inverse_kinematics(p_safe, target_pitch=-60)
        self.arm._send_and_audit(s_safe, p_safe)
        time.sleep(0.5)
        
        self.arm.move_line(p_safe, p_rest, p_start=-60, p_end=-60, duration=1.5)
        
        print("✓ 已复位")
        return p_rest


# =========================================================
# 4. 主应用程序
# =========================================================
class RobotApp:
    def __init__(self):
        self.grasp_sys = AutoGraspSystem()
        self.brain = RobotBrain()
        self.ear = RobotEar()
        
        self.current_pos = np.array([120.0, 0.0, 60.0])
        self.is_recording = False
        self.audio_frames = []

    def audio_callback(self, indata, frames, time_info, status):
        """音频流回调函数"""
        if self.is_recording:
            self.audio_frames.append(indata.copy())

    def get_audio_text(self):
        """将录制的音频转换为文本"""
        if not self.audio_frames:
            return ""
        
        audio_data = np.concatenate(self.audio_frames, axis=0)
        
        # 【优化1】裁剪首尾静音，减少 Whisper 幻觉
        audio_flat = audio_data.flatten()
        threshold = 0.01
        nonzero = np.where(np.abs(audio_flat) > threshold)[0]
        if len(nonzero) == 0:
            print(">>> [语音] 未检测到有效音频")
            return ""
        # 前后各留 0.3 秒余量
        margin = int(16000 * 0.3)
        start = max(0, nonzero[0] - margin)
        end = min(len(audio_flat), nonzero[-1] + margin)
        audio_trimmed = audio_flat[start:end]
        
        # 【优化2】音频太短（<0.5秒）或太长（>15秒）直接跳过
        duration = len(audio_trimmed) / 16000
        if duration < 0.5:
            print(f">>> [语音] 音频太短({duration:.1f}s)，跳过")
            return ""
        if duration > 15.0:
            print(f">>> [语音] 音频太长({duration:.1f}s)，截断到15秒")
            audio_trimmed = audio_trimmed[:16000 * 15]
        
        import scipy.io.wavfile as wav
        temp_file = "temp_voice.wav"
        wav.write(temp_file, 16000, (audio_trimmed * 32767).astype(np.int16))
        
        segments, _ = self.ear.model.transcribe(
            temp_file,
            beam_size=5,
            language="zh",
            no_speech_threshold=0.5,        # 静音段置信度阈值
            condition_on_previous_text=False, # 不依赖前文，防止重复幻觉
            initial_prompt="机械臂控制指令：抓取,抬起,放下,松开,复位,点头,摇头,削笔刀,盒子,零件,瓶子,厘米,毫米,向上,向下,向左,向右,向前,向后"
        )
        
        text = "".join([s.text for s in segments])
        text = self._fix_recognition(text.strip())
        return text
    
    def _fix_recognition(self, text):
        """【优化6】语音识别后处理：纠正常见错误 + 去重复幻觉"""
        if not text:
            return text
        
        # 去除标点
        text = re.sub(r'[,，。！？!?、;；]', '', text)
        
        # 纠正常见谐音误识别
        replacements = {
            '小笔刀': '削笔刀', '消笔刀': '削笔刀', '销笔刀': '削笔刀',
            '零米': '厘米', '里米': '厘米', '黎米': '厘米', '离米': '厘米',
            '公分': '厘米', '利米': '厘米',
            # 新增 点头/摇头 增强
            '电头': '点头', '点投': '点头', '店头': '点头', '垫头': '点头',
            '药头': '摇头', '要头': '摇头', '右头': '摇头', '咬头': '摇头', '摇土': '摇头',
        }
        for wrong, right in replacements.items():
            text = text.replace(wrong, right)
        
        # 【关键】检测并清除重复幻觉：如 "向右向右向右..." 
        # 查找连续重复的2-4字模式
        dedup_match = re.match(r'^(.{2,8}?)(.{2,8}?)\2{2,}', text)
        if dedup_match:
            # 只保留重复前的有效部分
            text = dedup_match.group(1)
            print(f">>> [去幻觉] 检测到重复，保留: {text}")
        
        # 另一种检测：整句过长且包含大量重复词
        if len(text) > 30:
            words = re.findall(r'向[上下左右前后]', text)
            if len(words) > 3:
                # 取第一个方向词及其前面的内容
                first_match = re.search(r'(.*?向[上下左右前后].*?\d+.*?厘米)', text)
                if first_match:
                    text = first_match.group(1)
                else:
                    text = text[:20]  # 强制截断
                print(f">>> [去幻觉] 文本过长，截断为: {text}")
        
        return text.strip()

    def execute_command(self, cmd):
        """执行单条指令"""
        action = cmd.get("action")
        print(f"\n>>> 解析指令: {cmd}")
        
        if action == "lift":
            target = cmd.get("target", "part")
            height = float(cmd.get("height", 50))
            
            pos = self.grasp_sys.find_target(target)
            if pos:
                rx, ry = pos
                # 【修复】使用正确的方法名
                self.current_pos = self.grasp_sys.execute_lift(rx, ry, height)
                print(f"✓ 已抬起 {target}，当前位置: {self.current_pos}")
            else:
                print(f"✗ 未找到目标: {target}")
                print(f"  当前可见目标: {list(self.grasp_sys.detected_targets.keys())}")
                
        elif action == "pick":
            target = cmd.get("target", "part")
            
            pos = self.grasp_sys.find_target(target)
            if pos:
                rx, ry = pos
                # 【修复】使用正确的方法名
                self.current_pos = self.grasp_sys.execute_pick(rx, ry)
                print(f"✓ 已抓取 {target}，当前位置: {self.current_pos}")
            else:
                print(f"✗ 未找到目标: {target}")
                print(f"  当前可见目标: {list(self.grasp_sys.detected_targets.keys())}")
                
        elif action == "move_inc":
            axis = cmd.get("axis", "z")
            value = float(cmd.get("value", 0))
            
            new_pos = self.current_pos.copy()
            if axis == 'x':
                new_pos[0] += value
            elif axis == 'y':
                new_pos[1] += value
            elif axis == 'z':
                new_pos[2] += value
            
            new_pos[0] = np.clip(new_pos[0], 80, 250)
            new_pos[1] = np.clip(new_pos[1], -120, 120)
            new_pos[2] = np.clip(new_pos[2], -20, 200)
            
            print(f">>> 增量移动: {axis}轴 {value}mm")
            print(f"    {self.current_pos} → {new_pos}")
            self.grasp_sys.arm.move_line(self.current_pos, new_pos, duration=1.5)
            self.current_pos = new_pos
            print(f"✓ 移动完成，当前位置: {self.current_pos}")
            
        elif action == "release":
            self.grasp_sys.execute_release()
            
        elif action == "drop":
            self.current_pos = self.grasp_sys.execute_drop(self.current_pos)
            
        elif action == "reset":
            self.current_pos = self.grasp_sys.execute_reset()
        
        elif action == "nod":
            print(">>> 执行点头动作")
            # 记录当前位置
            base_pos = self.current_pos.copy()
            # 动作幅度 3cm
            dist = 30.0
            
            # 清空缓冲，确保动作连贯
            self.grasp_sys.arm.servo_buffer = []
            
            for i in range(3):
                # 向上
                up_pos = base_pos.copy()
                up_pos[2] += dist
                self.grasp_sys.arm.move_line(base_pos, up_pos, duration=0.5)
                # 向下
                down_pos = base_pos.copy()
                down_pos[2] -= dist
                self.grasp_sys.arm.move_line(up_pos, down_pos, duration=0.8)
                # 回中
                self.grasp_sys.arm.move_line(down_pos, base_pos, duration=0.5)
                
            self.current_pos = base_pos
            print("✓ 点头完成")

        elif action == "shake_head":
            print(">>> 执行摇头动作")
            # 记录当前位置
            base_pos = self.current_pos.copy()
            # 动作幅度 3cm
            dist = 30.0
            
            # 清空缓冲
            self.grasp_sys.arm.servo_buffer = []
            
            for i in range(3):
                # 向左 (y增加是左)
                left_pos = base_pos.copy()
                left_pos[1] += dist
                self.grasp_sys.arm.move_line(base_pos, left_pos, duration=0.5)
                # 向右
                right_pos = base_pos.copy()
                right_pos[1] -= dist
                self.grasp_sys.arm.move_line(left_pos, right_pos, duration=0.8)
                # 回中
                self.grasp_sys.arm.move_line(right_pos, base_pos, duration=0.5)
                
            self.current_pos = base_pos
            print("✓ 摇头完成")

        else:
            print(f"✗ 未知动作: {action}")

    def run(self):
        """主循环"""
        window_name = "Voice Robot Control"
        cv2.namedWindow(window_name)
        # 【关键】注册鼠标回调函数
        cv2.setMouseCallback(window_name, self.grasp_sys.mouse_callback)
        
        stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=16000
        )
        stream.start()
        
        print("\n" + "="*50)
        print("  语音控制机械臂系统")
        print("="*50)
        print(" [SPACE] : 按住录音，松开识别")
        print(" [C]     : 进入标定模式 (点击4个点)")
        print(" [R]     : 手动复位")
        print(" [O]     : 手动松开夹爪")
        print(" [Q]     : 退出程序")
        print("="*50)
        
        while True:
            ret, frame = self.grasp_sys.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # 更新YOLO检测（标定模式下会跳过）
            frame = self.grasp_sys.update_detections(frame)
            
            # 绘制标定UI（标定点、标定框等）
            frame = self.grasp_sys.draw_calibration_ui(frame)
            
            # 显示录音状态
            if self.is_recording:
                cv2.putText(frame, "● RECORDING...", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示夹爪状态
            gripper_status = "CLOSED" if self.grasp_sys.gripper_closed else "OPEN"
            gripper_color = (0, 0, 255) if self.grasp_sys.gripper_closed else (0, 255, 0)
            cv2.putText(frame, f"Gripper: {gripper_status}", (1050, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, gripper_color, 2)
            
            # 显示当前位置
            pos_text = f"Pos: X={self.current_pos[0]:.0f} Y={self.current_pos[1]:.0f} Z={self.current_pos[2]:.0f}"
            cv2.putText(frame, pos_text, (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 显示检测到的目标（非标定模式）
            if not self.grasp_sys.is_calibrating:
                targets_text = f"Targets: {list(self.grasp_sys.detected_targets.keys())}"
                cv2.putText(frame, targets_text, (50, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('c'):
                # 进入/退出标定模式
                if not self.grasp_sys.is_calibrating:
                    self.grasp_sys.start_calibration()
                else:
                    # 如果已经在标定模式，按C取消
                    print(">>> [取消] 退出标定模式")
                    self.grasp_sys.is_calibrating = False
                    self.grasp_sys.temp_points = []
                
            elif key == ord(' '):
                # 标定模式下禁用录音
                if self.grasp_sys.is_calibrating:
                    print(">>> [提示] 请先完成标定或按C取消")
                    continue
                    
                if not self.is_recording:
                    self.is_recording = True
                    self.audio_frames = []
                    print("\n🎤 录音开始... (松开空格键结束)")
                else:
                    self.is_recording = False
                    print("🧠 正在识别语音...")
                    
                    text = self.get_audio_text()
                    print(f"✅ 你说: \"{text}\"")
                    
                    if text:
                        print("🤖 正在解析指令...")
                        cmds = self.brain.think(text)
                        
                        if cmds:
                            print(f"📋 解析结果: {cmds}")
                            for cmd in cmds:
                                self.execute_command(cmd)
                            print("\n>>> 所有指令执行完毕，等待下一条语音...")
                        else:
                            print("✗ 无法解析指令")
                            
            elif key == ord('r'):
                if not self.grasp_sys.is_calibrating:
                    self.current_pos = self.grasp_sys.execute_reset()
                
            elif key == ord('o'):
                if not self.grasp_sys.is_calibrating:
                    self.grasp_sys.execute_release()
        
        stream.stop()
        self.grasp_sys.cap.release()
        cv2.destroyAllWindows()
        self.grasp_sys.arm.close()
        print("\n>>> 程序已退出")


# =========================================================
# 5. 程序入口
# =========================================================
if __name__ == "__main__":
    app = RobotApp()
    app.run()