from collections import deque

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from scipy.optimize import minimize

class RobotArmUltimate:
    def __init__(self, port='COM3', baud=115200):
        # --- 1. Physical parameters (mm) ---
        self.L1, self.L2, self.L3, self.L4 = 70.0, 75.0, 50.0, 130.0
        self.current_servos_logic = np.array([90.0, 90.0, 90.0, 90.0])
        self.last_sent_servos = np.array([0.0, 0.0, 0.0, 0.0])
        self.path_history = []

        # --- 2. Tilt correction offsets ---
        # Increase OFFSET_Y if end-effector droops during horizontal moves
        self.OFFSET_Y = 0.0
        self.OFFSET_Z = 0.0

        # --- 3. Damping parameters ---
        self.buffer_size = 3
        self.servo_buffer = deque(maxlen=self.buffer_size)
        self.max_servo_speed = 30.0  # max servo speed (deg/frame)
        self.damping_factor = 0.7

        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(4)
            self.ser.flushInput()
            print("[System Ready] S-Curve smoothing + tilt correction loaded.")
            self.gripper_control(70)
        except serial.SerialException as e:
            print(f"[Warning] Serial connection failed ({e}), entering simulation mode.")
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
        """Send servo command through multi-layer damping pipeline."""
        # Apply tilt correction before filtering
        s_corrected = s.copy()
        s_corrected[1] += self.OFFSET_Y
        s_corrected[2] += self.OFFSET_Z

        s_logic = np.array([np.clip(x, 0, 180) for x in s_corrected])

        # Layer 1: moving average filter (deque auto-discards oldest)
        self.servo_buffer.append(s_logic)
        s_smoothed = np.mean(self.servo_buffer, axis=0)

        # Layer 2: speed limit
        delta = s_smoothed - self.current_servos_logic
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.max_servo_speed:
            s_smoothed = self.current_servos_logic + delta * (self.max_servo_speed / delta_norm)

        # Layer 3: damping factor
        s_damped = self.current_servos_logic * (1 - self.damping_factor) + s_smoothed * self.damping_factor

        s_logic_final = np.array([int(round(np.clip(x, 0, 180))) for x in s_damped])
        s_hardware = np.array([s_logic_final[0], s_logic_final[1], 180 - s_logic_final[2], 180 - s_logic_final[3]])

        # Layer 4: dead-zone filter - skip if total delta < 1 degree
        if np.sum(np.abs(s_hardware - self.last_sent_servos)) < 1.0:
            return

        if self.ser:
            cmd = (f"Servo_ArmX{s_hardware[0]}\nServo_ArmY{s_hardware[1]}\n"
                   f"Servo_ArmZ{s_hardware[2]}\nServo_ArmB{s_hardware[3]}\n")
            self.ser.write(cmd.encode())
            self.last_sent_servos = s_hardware.astype(int)

        self.current_servos_logic = s_logic_final
        _, _, pts = self.forward_kinematics(*s_logic_final)
        self.path_history.append(pts)

    def move_line(self, start_xyz, end_xyz, p_start=-90, p_end=-90, duration=3.0):
        """S-Curve trajectory planning with adaptive FPS based on distance."""
        distance = np.linalg.norm(end_xyz - start_xyz)

        if distance < 50:
            fps = 20
        elif distance < 150:
            fps = 40
        else:
            fps = 60

        steps = max(int(duration * fps), 10)
        print(f"[Path] {start_xyz} -> {end_xyz}, dist={distance:.1f}mm, fps={fps}")
        
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
        self.servo_buffer = deque(maxlen=buffer_size)
        self.max_servo_speed = max_speed
        self.damping_factor = damping_factor
        print(f"[OK] Damping params updated: window={buffer_size}, speed_limit={max_speed}deg/frame, damping={damping_factor}")

    def set_correction(self, offset_y=0.0, offset_z=0.0):
        self.OFFSET_Y = offset_y
        self.OFFSET_Z = offset_z
        print(f"[OK] Tilt correction updated: Y_OFFSET={offset_y}, Z_OFFSET={offset_z}")

    def close(self):
        if self.ser: self.ser.close()

if __name__ == "__main__":
    arm = RobotArmUltimate(port='COM3')

    arm.set_damping_params(buffer_size=3, max_speed=30.0, damping_factor=0.7)
    # Tune offset_y if end-effector droops during horizontal moves
    arm.set_correction(offset_y=-10.0, offset_z=0.0)

    p_standby = np.array([110, 100, 20])
    p_pick1   = np.array([210, 110, 20])
    p_pick2   = np.array([210, -110, 20])
    p_drop    = np.array([110, -100, 20])

    try:
        print("\n=== Smooth motion sequence (with tilt correction) ===")

        arm.move_line(p_standby, p_pick1, p_start=-60, p_end=-90, duration=1.0)
        arm.gripper_control(120)

        arm.move_line(p_pick1, p_pick2, p_start=-90, p_end=-30, duration=1.0)
        time.sleep(1)
        arm.move_line(p_pick2, p_drop, p_start=-30, p_end=-90, duration=1.0)
        arm.gripper_control(70)
        time.sleep(1)

        arm.move_line(p_drop, p_standby, p_start=-90, p_end=-60, duration=1.0)
        time.sleep(1)

        print("\n>>> Done.")

    except KeyboardInterrupt:
        pass
    finally:
        if arm.ser:
            arm.ser.close()