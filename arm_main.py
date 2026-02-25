"""
arm_main.py
===========
机械臂底层驱动模块。

提供基于 D-H 参数的正/逆运动学求解、多层减震轨迹规划
以及通过 ESP32 串口控制 MG996R 舵机的全套接口。
"""

# ── 标准库 ──
import math
import time

# ── 第三方库 ──
import numpy as np
from scipy.optimize import minimize
import serial

# ── 项目配置 ──
from config import (
    SERIAL_PORT, SERIAL_BAUD,
    ARM_DAMPING_BUFFER_SIZE, ARM_DAMPING_MAX_SPEED, ARM_DAMPING_FACTOR,
    ARM_CORRECTION_OFFSET_Y, ARM_CORRECTION_OFFSET_Z,
    ARM_REST_POSITION,
)


class RobotArmUltimate:
    """
    4-DOF 机械臂控制器。

    集成 D-H 正/逆运动学、S-Curve 轨迹规划、多层减震滤波
    以及 ESP32 串口通信，支持无串口连接时的仿真模式。

    Args:
        port (str): ESP32 串口号，默认读取 config.SERIAL_PORT。
        baud (int): 串口波特率，默认读取 config.SERIAL_BAUD。
    """

    def __init__(self, port: str = SERIAL_PORT, baud: int = SERIAL_BAUD):
        # --- 1. 物理参数 ---
        self.L1, self.L2, self.L3, self.L4 = 70.0, 75.0, 50.0, 130.0
        self.current_servos_logic = np.array([90.0, 90.0, 90.0, 90.0])
        self.last_sent_servos = np.array([0.0, 0.0, 0.0, 0.0]) # 记录上次发送的值
        self.path_history = []
        
        # --- 倾斜修正参数（单位：度）---
        # 若水平移动时末端下沉，可适当增大 OFFSET_Y
        self.OFFSET_Y = ARM_CORRECTION_OFFSET_Y
        self.OFFSET_Z = ARM_CORRECTION_OFFSET_Z

        # --- 2. 减震参数 ---
        self.servo_buffer    = []                       # 指令缓冲（移动平均滤波）
        self.buffer_size     = ARM_DAMPING_BUFFER_SIZE  # 移动平均窗口大小
        self.max_servo_speed = ARM_DAMPING_MAX_SPEED    # 舵机最大速度（度/帧）
        self.damping_factor  = ARM_DAMPING_FACTOR       # 阻尼系数

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(4)
            self.ser.flushInput()
            print(">>> [系统就绪] 已加载 S-Curve 平滑减震算法 + 倾斜修正。")
            self.gripper_control(70)
        except serial.SerialException as e:
            print(f">>> [警告] 串口初始化失败（{e}），进入仿真模式。")
            self.ser = None

    def dh_matrix(self, theta_deg: float, d: float, a: float, alpha_deg: float) -> np.ndarray:
        """计算标准 D-H 变换矩阵。"""
        theta, alpha = np.radians(theta_deg), np.radians(alpha_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),               np.cos(alpha),              d],
            [0,              0,                           0,                          1]
        ])

    def forward_kinematics(self, s1: float, s2: float, s3: float, s4: float):
        """正运动学：由舵机逻辑角度计算末端位置与俯仰角。"""
        t1, t2, t3, t4 = s1-90, 180-s2, 90-s3, s4-90
        T01 = self.dh_matrix(t1, self.L1, 0, 90)
        T12 = T01 @ self.dh_matrix(t2, 0, self.L2, 0)
        T23 = T12 @ self.dh_matrix(t3, 0, self.L3, 0)
        T34 = T23 @ self.dh_matrix(t4, 0, self.L4, 0)
        return T34[0:3, 3], t2 + t3 + t4, [np.array([0,0,0]), T01[0:3,3], T12[0:3,3], T23[0:3,3], T34[0:3,3]]

    def inverse_kinematics(self, target_xyz: np.ndarray, target_pitch: float = -90, init_angles=None) -> np.ndarray:
        """逆运动学：由目标位置与俯仰角求解舵机角度（SLSQP 优化）。"""
        if init_angles is None: init_angles = self.current_servos_logic
        def objective(s):
            pos, pitch, _ = self.forward_kinematics(*s)
            dist_err = np.linalg.norm(pos - target_xyz)
            pitch_err = abs(pitch - target_pitch)
            return dist_err * 2.0 + pitch_err * 0.1
        bounds = [(0, 180)] * 4
        res = minimize(objective, init_angles, bounds=bounds, method='SLSQP', tol=1e-3)
        return res.x

    def _send_and_audit(self, s: np.ndarray, target_xyz: np.ndarray) -> None:
        """带多层减震的舵机指令发送函数。

        层次：倾斜修正 → 移动平均滤波 → 速度限制 → 阻尼因子 → 死区过滤。
        """
        # --- 应用倾斜修正 ---
        s_corrected = s.copy()
        s_corrected[1] += self.OFFSET_Y
        s_corrected[2] += self.OFFSET_Z

        s_logic = np.array([np.clip(x, 0, 180) for x in s_corrected])
        
        # --- 层1: 移动平均滤波器 ---
        self.servo_buffer.append(s_logic)
        if len(self.servo_buffer) > self.buffer_size:
            self.servo_buffer.pop(0)
        s_smoothed = np.mean(self.servo_buffer, axis=0)
        
        # --- 层2: 速度限制 ---
        delta = s_smoothed - self.current_servos_logic
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm > self.max_servo_speed:
            s_smoothed = self.current_servos_logic + delta * (self.max_servo_speed / delta_norm)
        
        # --- 层3: 阻尼因子 ---
        s_damped = self.current_servos_logic * (1 - self.damping_factor) + s_smoothed * self.damping_factor
        
        s_logic_final = np.array([int(round(np.clip(x, 0, 180))) for x in s_damped])
        s_hardware = np.array([s_logic_final[0], s_logic_final[1], 180 - s_logic_final[2], 180 - s_logic_final[3]])
        
        # --- 层4: 死区过滤 ---
        if np.sum(np.abs(s_hardware - self.last_sent_servos)) < 1.0:
            return 

        if self.ser:
            cmd = f"Servo_ArmX{s_hardware[0]}\nServo_ArmY{s_hardware[1]}\n" \
                  f"Servo_ArmZ{s_hardware[2]}\nServo_ArmB{s_hardware[3]}\n"
            self.ser.write(cmd.encode())
            self.last_sent_servos = s_hardware.astype(int)
        
        self.current_servos_logic = s_logic_final # 这里的逻辑值已经是修正后的，保证迭代平滑
        _, _, pts = self.forward_kinematics(*s_logic_final)
        self.path_history.append(pts)

    def move_line(
        self,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        p_start: float = -90,
        p_end: float = -90,
        duration: float = 3.0,
    ) -> None:
        """S-Curve 轨迹规划：在 start_xyz 到 end_xyz 之间内插，动态自适应 FPS。"""
        distance = np.linalg.norm(end_xyz - start_xyz)
        
        if distance < 50:
            fps = 20  
        elif distance < 150:
            fps = 40  
        else:
            fps = 60  
            
        steps = max(int(duration * fps), 10)  
        print(f"规划平滑路径: {start_xyz} -> {end_xyz}, 距离={distance:.1f}mm, FPS={fps}")
        
        for i in range(steps + 1):
            t = i / steps
            smooth_t = 0.5 * (1 - math.cos(math.pi * t)) 
            
            curr_xyz = start_xyz + (end_xyz - start_xyz) * smooth_t
            curr_pitch = p_start + (p_end - p_start) * smooth_t
            
            best_s = self.inverse_kinematics(curr_xyz, target_pitch=curr_pitch)
            self._send_and_audit(best_s, curr_xyz)
            time.sleep(1.0 / fps)

    def gripper_control(self, angle: int) -> None:
        """控制娹爬开关角度。

        Args:
            angle: 目标角度，建议范围 70（张开）~ 120（闭合）。
        """
        if self.ser:
            self.ser.write(f"Servo_Gripper{int(angle)}\n".encode())
            time.sleep(0.8) 

    def set_damping_params(self, buffer_size: int = 3, max_speed: float = 30.0, damping_factor: float = 0.7) -> None:
        """动态调整减震参数。"""
        self.buffer_size = buffer_size
        self.max_servo_speed = max_speed
        self.damping_factor = damping_factor
        print(f"✓ 减震参数已更新: 滤波窗口={buffer_size}, 限速={max_speed}度/帧, 阻尼={damping_factor}")

    def set_correction(self, offset_y: float = 0.0, offset_z: float = 0.0) -> None:
        """设置关节倾斜修正偏移量（单位：度）。"""
        self.OFFSET_Y = offset_y
        self.OFFSET_Z = offset_z
        print(f"✓ 倾斜修正已更新: Y_OFFSET={offset_y}, Z_OFFSET={offset_z}")

    def close(self) -> None:
        """\u5173\u95ed\u4e32\u53e3\u8fde\u63a5。"""
        if self.ser: self.ser.close()

if __name__ == "__main__":
    arm = RobotArmUltimate()

    # 调试用减震参数
    arm.set_damping_params(buffer_size=3, max_speed=30.0, damping_factor=0.7)
    # 调试用倾斜修正（正向移动时若末端下沉，可适当增大 offset_y）
    arm.set_correction(offset_y=ARM_CORRECTION_OFFSET_Y, offset_z=ARM_CORRECTION_OFFSET_Z)

    p_standby = np.array([110,  100,  20])
    p_pick1   = np.array([210,  110,  20])
    p_pick2   = np.array([210, -110,  20])
    p_drop    = np.array([110, -100,  20])

    try:
        print("\n=== 开始平滑动作序列 (带倾斜修正) ===")
        
        # 1. 初始定位
        # s_init = arm.inverse_kinematics(p_standby, target_pitch=-90)
        # arm._send_and_audit(s_init, p_standby)
        # time.sleep(2)

        # 2. 减震下降
        arm.move_line(p_standby, p_pick1, p_start=-60, p_end=-90, duration=1.0)
        arm.gripper_control(120)

        # 3. 减震平移
        arm.move_line(p_pick1, p_pick2, p_start=-90, p_end=-30, duration=1.0)
        time.sleep(1)
        arm.move_line(p_pick2, p_drop, p_start=-30, p_end=-90, duration=1.0)
        arm.gripper_control(70)
        time.sleep(1)

        # 4. 减震返回待命
        arm.move_line(p_drop, p_standby, p_start=-90, p_end=-60, duration=1.0)
        time.sleep(1)

        print("\n>>> 任务结束。")

    except KeyboardInterrupt:
        pass
    finally:
        if arm.ser: arm.ser.close()