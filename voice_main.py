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

# Disable proxy for local serial/model communication
os.environ["no_proxy"] = "localhost,127.0.0.1"

# =========================================================
# 1. Calibration data (4-corner robot workspace in mm)
# =========================================================
robot_points = np.array([
    [90, 90], [200, 90], [200, -90], [90, -90]
], dtype=np.float32)
initial_image_points = [[817, 72], [433, 79], [291, 612], [1029, 610]]
CALIB_CENTER_Y = 0.0

# =========================================================
# 2. LLM-based command parser
# =========================================================
class RobotBrain:
    def __init__(self, model_path=r"D:\lora\2"):
        self.model_path = model_path
        print(f"[Brain] Loading model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()
        # IMPORTANT: system_prompt must exactly match training data to avoid output drift.
        # This is Chinese because the LLM was fine-tuned on Chinese robot commands (i18n data).
        self.system_prompt = "你是机械臂JSON转换器。一个指令=一个动作！\n规则：\n1. 单位：厘米×10=毫米\n2. 轴向：上/下=z, 左/右=y, 前/后=x\n3. 正负：上/左/前=正, 下/右/后=负\n4. 目标：所有物体映射为 \"part\"\n5. 只输出JSON数组。\n6. 示例：摇头即输出 [{\"action\": \"shake_head\"}]"

    def think(self, user_text):
        """Convert natural language voice command to a list of JSON action dicts."""
        # Simple direction/release/reset commands go through the rule engine directly
        # to avoid the LLM misclassifying "move down 3cm" as a lift action.
        simple_result = self._try_simple_parse(user_text)
        if simple_result:
            print(f"[Rule] Matched simple command: {simple_result}")
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
            print(f"[LLM raw output] {content}")

            # Extract first complete JSON array or object
            match = re.search(r'\[\s*\{[^[\]]*\}\s*\]', content, re.DOTALL)
            if not match:
                match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', content, re.DOTALL)
            if not match:
                match = re.search(r'\{[^{}]*\}', content, re.DOTALL)

            if match:
                json_str = match.group()

                # Sanitize LLM output: remove trailing punctuation and fix common formatting
                json_str = re.sub(r'[。！？\.\!\?]\s*$', '', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = json_str.replace("'", '"')
                json_str = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
                json_str = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])', r':"\1"\2', json_str)
                json_str = json_str.strip()

                print(f"[LLM sanitized JSON] {json_str}")

                res = json.loads(json_str)
                result = res if isinstance(res, list) else [res]

                # Filter to known actions; enforce single-action-per-command
                valid_actions = ["pick", "lift", "move_inc", "release", "drop", "reset"]
                filtered = [cmd for cmd in result if cmd.get("action") in valid_actions]

                if filtered:
                    # lift takes priority over any co-occurring actions
                    for cmd in filtered:
                        if cmd.get("action") == "lift":
                            print(f"[Filter] Keeping lift action: {cmd}")
                            return [cmd]

                    print(f"[Filter] Keeping first valid action: {filtered[0]}")
                    return [filtered[0]]

        except json.JSONDecodeError as e:
            print(f"[Error] JSON parse failed: {e}")
        except Exception as e:
            print(f"[Error] Brain exception: {e}")

        # Fall back to regex-based parsing
        return self._fallback_parse(user_text)
    
    def _try_simple_parse(self, text):
        """Rule-based fast path for simple commands; returns None to fall through to LLM.

        Chinese voice patterns below are i18n data — they MUST remain in Chinese
        because they match what the user actually says.
        """
        # --- i18n: Chinese voice command patterns ---
        if re.search(r'(松开|放开|释放|松手)', text):
            return [{"action": "release"}]

        if re.search(r'(复位|回到原位|归位|回原点|回原位|回到原点)', text):
            return [{"action": "reset"}]

        # "put down" must be checked before directional "down" to avoid misclassification
        if re.search(r'(放下|放到|放置)', text):
            return [{"action": "drop"}]

        if re.search(r'(点头)', text):
            return [{"action": "nod"}]
        if re.search(r'(摇头)', text):
            return [{"action": "shake_head"}]

        # Commands containing an object name are complex; delegate to LLM for pick/lift
        has_object = re.search(r'(削笔刀|盒子|物块|零件|瓶子|part)', text)
        if has_object:
            return None

        # Helper: extract numeric value and convert to mm
        def extract_value(match, num_group=1, unit_group=2):
            val_str = match.group(num_group)
            # Chinese numeral map (i18n data)
            cn_map = {'一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
                      '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
            if val_str in cn_map:
                value = cn_map[val_str]
            else:
                try:
                    value = int(val_str)
                except ValueError:
                    value = 5  # fallback default
            unit = match.group(unit_group) or '厘米'
            # '米米' is a common Whisper mishearing of '毫米' (mm)
            if '毫米' in unit or 'mm' in unit or '米米' in unit:
                return value
            return value * 10  # cm → mm

        # Pattern: arabic digit OR Chinese numeral
        num_pattern = r'(\d+|[一二两三四五六七八九十])'

        # Up / Down (z-axis)
        m = re.search(f'向?上.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "z", "value": extract_value(m)}]
        m = re.search(f'向?下.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "z", "value": -extract_value(m)}]

        # Left / Right (y-axis)
        m = re.search(f'向?左.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "y", "value": extract_value(m)}]
        m = re.search(f'向?右.{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "y", "value": -extract_value(m)}]

        # Forward / Back (x-axis)
        m = re.search(f'(向?前|往前).{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "x", "value": extract_value(m, 2, 3)}]
        m = re.search(f'(向?后|往后).{{0,4}}?{num_pattern}\\s*(厘米|cm|毫米|mm|米米)?', text)
        if m:
            return [{"action": "move_inc", "axis": "x", "value": -extract_value(m, 2, 3)}]

        # Fuzzy directional commands without explicit distance — use default 5 cm
        DEFAULT_MOVE = 50.0  # mm
        if re.search(r'(向?上|抬起|举起|往上)', text):
            return [{"action": "move_inc", "axis": "z", "value": DEFAULT_MOVE}]
        if re.search(r'(向?下|往下)', text):
            return [{"action": "move_inc", "axis": "z", "value": -DEFAULT_MOVE}]
        if re.search(r'(向?左|往左)', text):
            return [{"action": "move_inc", "axis": "y", "value": DEFAULT_MOVE}]
        if re.search(r'(向?右|往右)', text):
            return [{"action": "move_inc", "axis": "y", "value": -DEFAULT_MOVE}]
        if re.search(r'(向?前|往前)', text):
            return [{"action": "move_inc", "axis": "x", "value": DEFAULT_MOVE}]
        if re.search(r'(向?后|往后)', text):
            return [{"action": "move_inc", "axis": "x", "value": -DEFAULT_MOVE}]

        return None  # not a simple command — fall through to LLM

    def _fallback_parse(self, text):
        """Last-resort regex parser when LLM output cannot be decoded.

        Chinese patterns below are i18n data matching actual spoken commands.
        """
        print("[Fallback] Activating regex fallback parser")
        cmds = []

        # Priority 1: pure directional move (no object name)
        up_match       = re.search(r'向?上.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        down_match     = re.search(r'向?下.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        left_match     = re.search(r'向?左.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        right_match    = re.search(r'向?右.{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        forward_match  = re.search(r'(向?前|往前).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)
        backward_match = re.search(r'(向?后|往后).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)?', text)

        def to_mm(value, unit):
            return value if ('毫米' in unit or 'mm' in unit) else value * 10

        if up_match:
            v = to_mm(int(up_match.group(1)), up_match.group(2) or '厘米')
            cmds.append({"action": "move_inc", "axis": "z", "value": v})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if down_match:
            v = to_mm(int(down_match.group(1)), down_match.group(2) or '厘米')
            cmds.append({"action": "move_inc", "axis": "z", "value": -v})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if left_match:
            v = to_mm(int(left_match.group(1)), left_match.group(2) or '厘米')
            cmds.append({"action": "move_inc", "axis": "y", "value": v})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if right_match:
            v = to_mm(int(right_match.group(1)), right_match.group(2) or '厘米')
            cmds.append({"action": "move_inc", "axis": "y", "value": -v})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if forward_match:
            v = to_mm(int(forward_match.group(2)), forward_match.group(3) or '厘米')
            cmds.append({"action": "move_inc", "axis": "x", "value": v})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if backward_match:
            v = to_mm(int(backward_match.group(2)), backward_match.group(3) or '厘米')
            cmds.append({"action": "move_inc", "axis": "x", "value": -v})
            print(f"[Fallback] result: {cmds}")
            return cmds

        # Priority 2: lift object by height
        lift_pattern1 = re.search(
            r'(把|将)?.{0,6}?(削笔刀|盒子|物块|零件|part).{0,4}?(抬起|拿起|举起).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)', text)
        lift_pattern2 = re.search(
            r'(抬起|拿起|举起).{0,6}?(削笔刀|盒子|物块|零件|part).{0,4}?(\d+)\s*(厘米|cm|毫米|mm)', text)

        if lift_pattern1:
            h = to_mm(int(lift_pattern1.group(4)), lift_pattern1.group(5))
            cmds.append({"action": "lift", "target": "part", "height": h})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if lift_pattern2:
            h = to_mm(int(lift_pattern2.group(3)), lift_pattern2.group(4))
            cmds.append({"action": "lift", "target": "part", "height": h})
            print(f"[Fallback] result: {cmds}")
            return cmds

        # Grab
        if re.search(r'(抓住|抓取|拿住|夹住).{0,6}?(削笔刀|盒子|物块|零件|part)', text):
            cmds.append({"action": "pick", "target": "part"})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if re.search(r'(松开|放开|释放|松手)', text):
            cmds.append({"action": "release"})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if re.search(r'(放下|放到|放置)', text):
            cmds.append({"action": "drop"})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if re.search(r'(复位|回到原位|归位|回原点|回原位)', text):
            cmds.append({"action": "reset"})
            print(f"[Fallback] result: {cmds}")
            return cmds

        if cmds:
            print(f"[Fallback] result: {cmds}")
            return cmds
        return None


# =========================================================
# 3. Vision grasp system (YOLO detection + hand-eye calibration)
# =========================================================
class AutoGraspSystem:
    def __init__(self):
        print("\n[1/3] Initializing robot arm...")
        self.arm = RobotArmUltimate(port='COM3')
        self.arm.set_damping_params(buffer_size=3, max_speed=25.0, damping_factor=0.6)
        self.arm.set_correction(offset_y=-10.0, offset_z=0.0)

        print("[2/3] Loading YOLO model...")
        self.model = YOLO('best.pt')

        print("[3/3] Initializing camera...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Calibration state
        self.image_points = np.array(initial_image_points, dtype=np.float32)
        self.temp_points = []       # temporary click buffer during recalibration
        self.is_calibrating = False
        self.update_homography()

        # Detection results: {class_name: (rx, ry, confidence)}
        self.detected_targets = {}

        self.gripper_closed = False

        print("[OK] System ready!")

    def update_homography(self):
        """Recompute homography matrix from current image calibration points."""
        self.H, _ = cv2.findHomography(self.image_points, robot_points)
        self.pixel_center_u = int(np.mean(self.image_points[:, 0]))
        self.pixel_center_v = int(np.mean(self.image_points[:, 1]))
        print(f"[Calibration] Homography updated, center=({self.pixel_center_u}, {self.pixel_center_v})")

    def mouse_callback(self, event, x, y, flags, param):
        """Record calibration clicks only in calibration mode."""
        if self.is_calibrating and event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append([x, y])
            print(f" -> Recorded P{len(self.temp_points)}: [{x}, {y}]")

            if len(self.temp_points) == 4:
                self.image_points = np.array(self.temp_points, dtype=np.float32)
                self.update_homography()
                self.is_calibrating = False
                print("[OK] Calibration updated, returning to normal mode.")
                print(f"     New points: {self.image_points.tolist()}")

    def start_calibration(self):
        """Enter calibration mode: click 4 corner points in order."""
        print("\n[Calibration] Click 4 points in order:")
        print("  P1(top-left) -> P2(top-right) -> P3(bottom-right) -> P4(bottom-left)")
        print("  Robot coords: (90,90) -> (200,90) -> (200,-90) -> (90,-90)")
        self.temp_points = []
        self.is_calibrating = True

    def pixel_to_robot(self, u, v):
        """Map pixel coordinates (u,v) to robot workspace coordinates (rx, ry) in mm."""
        vec = np.array([u, v, 1], dtype=np.float32).reshape(3, 1)
        res = np.dot(self.H, vec)
        rx = float(res[0, 0] / res[2, 0])
        ry = (CALIB_CENTER_Y * 2) - float(res[1, 0] / res[2, 0])
        return rx, ry

    def update_detections(self, frame):
        """Run YOLO inference on frame; skip during calibration mode."""
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
        """Overlay calibration UI on frame."""
        if self.is_calibrating:
            cv2.putText(frame, f"CALIBRATION MODE: Click Point P{len(self.temp_points)+1}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            for i, pt in enumerate(self.temp_points):
                cv2.circle(frame, tuple(pt), 8, (0, 255, 255), -1)
                cv2.putText(frame, f"P{i+1}", (pt[0]+10, pt[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
            # Normal mode: show calibration reference points
            cv2.drawMarker(frame, (self.pixel_center_u, self.pixel_center_v), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)

            for i, pt in enumerate(self.image_points):
                u, v = int(pt[0]), int(pt[1])
                cv2.circle(frame, (u, v), 8, (0, 0, 255), 2)
                cv2.putText(frame, f"P{i+1}", (u + 10, v - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            pts = self.image_points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

        return frame

    def find_target(self, target_name):
        """Look up detected target by class name; return (rx, ry) or None."""
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
        
        # Open gripper
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.3)

        self.arm.servo_buffer = []

        current_approx = np.array([120.0, 0.0, 60.0])  # default rest position
        self.arm.move_line(current_approx, p_up, p_start=-60, p_end=-60, duration=1.5)

        self.arm.move_line(p_up, p_down, p_start=-60, p_end=-90, duration=1.5)

        # Close gripper
        self.arm.gripper_control(120)
        self.gripper_closed = True
        time.sleep(0.5)

        self.arm.move_line(p_down, p_after, p_start=-90, p_end=-60, duration=1.0)

        return p_after

    def execute_lift(self, rx, ry, height):
        """Grasp target and raise it by `height` mm."""
        print(f"\n[Lift] target=({rx:.1f}, {ry:.1f}), height={height}mm")
        Z_HOVER = 120.0
        Z_GRAB = -15.0

        p_up = np.array([rx, ry, Z_HOVER])
        p_down = np.array([rx, ry, Z_GRAB])
        p_lifted = np.array([rx, ry, Z_GRAB + height])
        p_lifted[2] = np.clip(p_lifted[2], -20, 200)

        # Open gripper
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.3)

        self.arm.servo_buffer = []

        current_approx = np.array([120.0, 0.0, 60.0])
        self.arm.move_line(current_approx, p_up, p_start=-60, p_end=-60, duration=1.5)

        self.arm.move_line(p_up, p_down, p_start=-60, p_end=-90, duration=1.5)

        # Close gripper
        self.arm.gripper_control(120)
        self.gripper_closed = True
        time.sleep(0.5)
        
        self.arm.move_line(p_down, p_lifted, p_start=-90, p_end=-60, duration=1.5)

        print(f"[OK] Lifted to Z={p_lifted[2]:.1f}mm")
        return p_lifted

    def execute_release(self):
        """Open gripper without moving arm."""
        print("\n[Release] Opening gripper")
        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.5)
        print("[OK] Gripper open")

    def execute_drop(self, current_pos):
        """Lower end-effector to table height and release object."""
        Z_TABLE = -15.0  # table surface height (mm)
        drop_height = current_pos[2] - Z_TABLE
        print(f"\n[Drop] from Z={current_pos[2]:.1f}mm -> Z={Z_TABLE}mm (travel={drop_height:.1f}mm)")

        p_drop = np.array([current_pos[0], current_pos[1], Z_TABLE])

        self.arm.servo_buffer.clear()
        self.arm.move_line(current_pos, p_drop, p_start=-60, p_end=-90, duration=2.0)

        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.5)

        print("[OK] Object placed on table")
        return p_drop

    def execute_reset(self):
        """Return arm to rest position."""
        print("\n[Reset] Returning to rest position")
        p_rest = np.array([120, 0, 60])

        self.arm.gripper_control(70)
        self.gripper_closed = False
        time.sleep(0.3)

        self.arm.servo_buffer.clear()

        # Move through a safe intermediate point to avoid workspace boundary issues
        p_safe = np.array([120, 0, 100])
        s_safe = self.arm.inverse_kinematics(p_safe, target_pitch=-60)
        self.arm._send_and_audit(s_safe, p_safe)
        time.sleep(0.5)

        self.arm.move_line(p_safe, p_rest, p_start=-60, p_end=-60, duration=1.5)

        print("[OK] Reset complete")
        return p_rest


# =========================================================
# 4. Main application
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
        """sounddevice InputStream callback: accumulate audio while recording."""
        if self.is_recording:
            self.audio_frames.append(indata.copy())

    def get_audio_text(self):
        """Transcribe recorded audio frames to text."""
        if not self.audio_frames:
            return ""

        audio_data = np.concatenate(self.audio_frames, axis=0)

        # Trim leading/trailing silence to reduce Whisper hallucinations
        audio_flat = audio_data.flatten()
        threshold = 0.01
        nonzero = np.where(np.abs(audio_flat) > threshold)[0]
        if len(nonzero) == 0:
            print("[Audio] No speech detected")
            return ""

        margin = int(16000 * 0.3)  # 0.3 s padding on each side
        start = max(0, nonzero[0] - margin)
        end = min(len(audio_flat), nonzero[-1] + margin)
        audio_trimmed = audio_flat[start:end]

        duration = len(audio_trimmed) / 16000
        if duration < 0.5:
            print(f"[Audio] Too short ({duration:.1f}s), skipping")
            return ""
        if duration > 15.0:
            print(f"[Audio] Too long ({duration:.1f}s), truncating to 15s")
            audio_trimmed = audio_trimmed[:16000 * 15]

        temp_file = "temp_voice.wav"
        wav.write(temp_file, 16000, (audio_trimmed * 32767).astype(np.int16))

        segments, _ = self.ear.model.transcribe(
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

    def execute_command(self, cmd):
        """Dispatch a single parsed action command to the appropriate executor."""
        action = cmd.get("action")
        print(f"\n[Cmd] {cmd}")

        if action == "lift":
            target = cmd.get("target", "part")
            height = float(cmd.get("height", 50))

            pos = self.grasp_sys.find_target(target)
            if pos:
                rx, ry = pos
                self.current_pos = self.grasp_sys.execute_lift(rx, ry, height)
                print(f"[OK] Lifted {target}, pos={self.current_pos}")
            else:
                print(f"[Error] Target not found: {target}")
                print(f"  Visible: {list(self.grasp_sys.detected_targets.keys())}")

        elif action == "pick":
            target = cmd.get("target", "part")

            pos = self.grasp_sys.find_target(target)
            if pos:
                rx, ry = pos
                self.current_pos = self.grasp_sys.execute_pick(rx, ry)
                print(f"[OK] Picked {target}, pos={self.current_pos}")
            else:
                print(f"[Error] Target not found: {target}")
                print(f"  Visible: {list(self.grasp_sys.detected_targets.keys())}")
                
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
            
            print(f"[Move] axis={axis}, delta={value}mm: {self.current_pos} -> {new_pos}")
            self.grasp_sys.arm.move_line(self.current_pos, new_pos, duration=1.5)
            self.current_pos = new_pos
            print(f"[OK] Move complete, pos={self.current_pos}")
            
        elif action == "release":
            self.grasp_sys.execute_release()
            
        elif action == "drop":
            self.current_pos = self.grasp_sys.execute_drop(self.current_pos)
            
        elif action == "reset":
            self.current_pos = self.grasp_sys.execute_reset()
        
        elif action == "nod":
            print("[Nod] Executing nod motion")
            base_pos = self.current_pos.copy()
            dist = 30.0  # 3 cm amplitude

            self.grasp_sys.arm.servo_buffer.clear()

            for _ in range(3):
                up_pos = base_pos.copy(); up_pos[2] += dist
                self.grasp_sys.arm.move_line(base_pos, up_pos, duration=0.5)
                down_pos = base_pos.copy(); down_pos[2] -= dist
                self.grasp_sys.arm.move_line(up_pos, down_pos, duration=0.8)
                self.grasp_sys.arm.move_line(down_pos, base_pos, duration=0.5)

            self.current_pos = base_pos
            print("[OK] Nod complete")

        elif action == "shake_head":
            print("[ShakeHead] Executing shake-head motion")
            base_pos = self.current_pos.copy()
            dist = 30.0  # 3 cm amplitude

            self.grasp_sys.arm.servo_buffer.clear()

            for _ in range(3):
                left_pos = base_pos.copy(); left_pos[1] += dist  # +y = left
                self.grasp_sys.arm.move_line(base_pos, left_pos, duration=0.5)
                right_pos = base_pos.copy(); right_pos[1] -= dist
                self.grasp_sys.arm.move_line(left_pos, right_pos, duration=0.8)
                self.grasp_sys.arm.move_line(right_pos, base_pos, duration=0.5)

            self.current_pos = base_pos
            print("[OK] Shake-head complete")

        else:
            print(f"[Error] Unknown action: {action}")

    def run(self):
        """Main event loop."""
        window_name = "Voice Robot Control"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.grasp_sys.mouse_callback)

        stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=16000
        )
        stream.start()

        print("\n" + "="*50)
        print("  Voice-Controlled Robot Arm")
        print("="*50)
        print(" [SPACE] : Hold to record, release to recognize")
        print(" [C]     : Enter calibration mode (click 4 points)")
        print(" [R]     : Manual reset")
        print(" [O]     : Open gripper")
        print(" [Q]     : Quit")
        print("="*50)

        while True:
            ret, frame = self.grasp_sys.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = self.grasp_sys.update_detections(frame)
            frame = self.grasp_sys.draw_calibration_ui(frame)

            if self.is_recording:
                cv2.putText(frame, "● RECORDING...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            gripper_status = "CLOSED" if self.grasp_sys.gripper_closed else "OPEN"
            gripper_color = (0, 0, 255) if self.grasp_sys.gripper_closed else (0, 255, 0)
            cv2.putText(frame, f"Gripper: {gripper_status}", (1050, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, gripper_color, 2)

            pos_text = f"Pos: X={self.current_pos[0]:.0f} Y={self.current_pos[1]:.0f} Z={self.current_pos[2]:.0f}"
            cv2.putText(frame, pos_text, (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if not self.grasp_sys.is_calibrating:
                targets_text = f"Targets: {list(self.grasp_sys.detected_targets.keys())}"
                cv2.putText(frame, targets_text, (50, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
                
            elif key == ord('c'):
                if not self.grasp_sys.is_calibrating:
                    self.grasp_sys.start_calibration()
                else:
                    print("[Calibration] Cancelled")
                    self.grasp_sys.is_calibrating = False
                    self.grasp_sys.temp_points = []

            elif key == ord(' '):
                # Disable recording during calibration
                if self.grasp_sys.is_calibrating:
                    print("[Hint] Finish or cancel calibration first (press C)")
                    continue

                if not self.is_recording:
                    self.is_recording = True
                    self.audio_frames = []
                    print("\n[REC] Recording... (release SPACE to stop)")
                else:
                    self.is_recording = False
                    print("[ASR] Recognizing...")

                    text = self.get_audio_text()
                    print(f"[ASR] You said: \"{text}\"")

                    if text:
                        print("[Brain] Parsing command...")
                        cmds = self.brain.think(text)

                        if cmds:
                            print(f"[Cmd] Parsed: {cmds}")
                            for cmd in cmds:
                                self.execute_command(cmd)
                            print("\n[Done] Waiting for next command...")
                        else:
                            print("[Error] Could not parse command")

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
        print("\n[Exit] Program terminated")


# =========================================================
# 5. Entry point
# =========================================================
if __name__ == "__main__":
    app = RobotApp()
    app.run()