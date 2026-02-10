import numpy as np
from scipy.optimize import minimize
import serial
import time
import math
import matplotlib.pyplot as plt

class RobotArmUltimate:
    def __init__(self, port='COM3', baud=115200):
        # --- 1. 物理参数 ---
        self.L1, self.L2, self.L3, self.L4 = 70.0, 75.0, 50.0, 130.0
        self.current_servos_logic = np.array([90.0, 90.0, 90.0, 90.0])
        self.last_sent_servos = np.array([0.0, 0.0, 0.0, 0.0]) # 记录上次发送的值
        self.path_history = []
        
        # --- 【新增】倾斜修正参数 ---
        # 如果水平移动时往下掉，增加 OFFSET_Y (例如 2.0)
        self.OFFSET_Y = 0.0  
        self.OFFSET_Z = 0.0  

        # --- 2. 减震参数 ---
        self.servo_buffer = []  # 指令缓冲用于平滑滤波
        self.buffer_size = 3    # 移动平均窗口大小
        self.max_servo_speed = 30.0  # 舵机最大速度
        self.damping_factor = 0.7  # 阻尼因子

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(4) 
            self.ser.flushInput()
            print(">>> [系统就绪] 已加载 S-Curve 平滑减震算法 + 倾斜修正。")
            self.gripper_control(70)
        except:
            print(">>> [警告] 串口未连接，进入仿真。")
            self.ser = None

    def dh_matrix(self, theta_deg, d, a, alpha_deg):
        theta, alpha = np.radians(theta_deg), np.radians(alpha_deg)
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),               np.cos(alpha),              d],
            [0,              0,                           0,                          1]
        ])

    def forward_kinematics(self, s1, s2, s3, s4):
        t1, t2, t3, t4 = s1-90, 180-s2, 90-s3, s4-90
        T01 = self.dh_matrix(t1, self.L1, 0, 90)
        T12 = T01 @ self.dh_matrix(t2, 0, self.L2, 0)
        T23 = T12 @ self.dh_matrix(t3, 0, self.L3, 0)
        T34 = T23 @ self.dh_matrix(t4, 0, self.L4, 0)
        return T34[0:3, 3], t2 + t3 + t4, [np.array([0,0,0]), T01[0:3,3], T12[0:3,3], T23[0:3,3], T34[0:3,3]]

    def inverse_kinematics(self, target_xyz, target_pitch=-90, init_angles=None):
        if init_angles is None: init_angles = self.current_servos_logic
        def objective(s):
            pos, pitch, _ = self.forward_kinematics(*s)
            dist_err = np.linalg.norm(pos - target_xyz)
            pitch_err = abs(pitch - target_pitch)
            return dist_err * 2.0 + pitch_err * 0.1
        bounds = [(0, 180)] * 4
        res = minimize(objective, init_angles, bounds=bounds, method='SLSQP', tol=1e-3)
        return res.x

    def _send_and_audit(self, s, target_xyz):
        """带有多层减震的发送函数"""
        # --- 【修改点】应用倾斜修正 ---
        # 在进入滤波前，先加上固定的偏移量
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

    def move_line(self, start_xyz, end_xyz, p_start=-90, p_end=-90, duration=3.0):
        """核心改进：多层S-Curve轨迹规划 + 动态自适应FPS"""
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

    def gripper_control(self, angle):
        if self.ser:
            self.ser.write(f"Servo_Gripper{int(angle)}\n".encode())
            time.sleep(0.8) 

    def set_damping_params(self, buffer_size=3, max_speed=30.0, damping_factor=0.7):
        self.buffer_size = buffer_size
        self.max_servo_speed = max_speed
        self.damping_factor = damping_factor
        print(f"✓ 减震参数已更新: 滤波窗口={buffer_size}, 限速={max_speed}度/帧, 阻尼={damping_factor}")

    # --- 【新增】设置修正系数的接口 ---
    def set_correction(self, offset_y=0.0, offset_z=0.0):
        self.OFFSET_Y = offset_y
        self.OFFSET_Z = offset_z
        print(f"✓ 倾斜修正已更新: Y_OFFSET={offset_y}, Z_OFFSET={offset_z}")

    def close(self):
        if self.ser: self.ser.close()

if __name__ == "__main__":
    arm = RobotArmUltimate(port='COM3')
    
    # 1. 设置减震参数
    arm.set_damping_params(buffer_size=3, max_speed=30.0, damping_factor=0.7)

    # 2. 【核心调试】在这里设置修正值
    # 如果正向移动时 Z 往下掉，说明大臂（Y）低了，给它加一点
    arm.set_correction(offset_y=-10.0, offset_z=0.0) 
    
    p_standby = np.array([110, 100, 20]) 
    p_pick1   = np.array([210, 110, 20]) 
    p_pick2   = np.array([210, -110, 20]) 
    p_drop    = np.array([110, -100, 20]) 

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