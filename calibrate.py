"""
calibrate.py — 手眼标定辅助工具（GUI 版）
==========================================
功能：
  1. 键盘步进控制机械臂（WASD / QE）
  2. 输入目标坐标后点击「移动」直接到达
  3. 按「记录当前点」或空格键记录坐标，共 4 个点
  4. 记录完成后在终端打印 CALIB_ROBOT_POINTS

键盘（窗口聚焦时）
  W / S     →  X 轴 前 / 后
  A / D     →  Y 轴 左 / 右
  Q / E     →  Z 轴 上 / 下
  空格      →  记录当前点
  R         →  回待机位
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

os.environ["no_proxy"] = "localhost,127.0.0.1"

from config import SERIAL_PORT, SERIAL_BAUD, ARM_REST_POSITION
from arm_main import RobotArmUltimate


class CalibrateApp:
    STEP_DEFAULT = 5.0
    STEP_MIN = 1.0
    STEP_MAX = 20.0
    MAX_PTS = 4

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("手眼标定工具")
        self.root.resizable(False, False)

        self.arm: RobotArmUltimate | None = None
        self.rest = np.array(ARM_REST_POSITION, dtype=float)
        self.cur  = self.rest.copy()
        self.step = self.STEP_DEFAULT
        self.recorded: list[np.ndarray] = []
        self._busy = False          # 防止并发运动指令

        self._build_ui()
        self._bind_keys()
        self._refresh_ui()

        # 在后台线程连接机械臂，避免阻塞 GUI
        threading.Thread(target=self._connect_arm, daemon=True).start()

    # ─── 构建界面 ────────────────────────────────────────
    def _build_ui(self):
        PAD = dict(padx=8, pady=4)

        # ── 状态栏（顶部） ───────────────────────────────
        frm_top = ttk.LabelFrame(self.root, text="当前机械臂坐标")
        frm_top.grid(row=0, column=0, columnspan=2, sticky="ew", **PAD)

        self.lbl_coord = ttk.Label(frm_top, text="X=---  Y=---  Z=---",
                                   font=("Consolas", 16, "bold"), foreground="#2196F3")
        self.lbl_coord.pack(padx=12, pady=6)

        self.lbl_status = ttk.Label(frm_top, text="正在连接机械臂...",
                                    font=("微软雅黑", 10), foreground="gray")
        self.lbl_status.pack(padx=12, pady=(0, 6))

        # ── 步进控制 ─────────────────────────────────────
        frm_jog = ttk.LabelFrame(self.root, text="键盘步进控制")
        frm_jog.grid(row=1, column=0, sticky="nsew", **PAD)

        # 步进量
        frm_step = ttk.Frame(frm_jog)
        frm_step.pack(fill="x", padx=6, pady=4)
        ttk.Label(frm_step, text="步进量 (mm):").pack(side="left")
        self.spin_step = ttk.Spinbox(frm_step, from_=self.STEP_MIN, to=self.STEP_MAX,
                                     width=6, increment=1,
                                     command=self._on_step_change)
        self.spin_step.set(int(self.step))
        self.spin_step.pack(side="left", padx=4)

        # 方向按钮（3×3 网格模拟方向盘）
        frm_dir = ttk.Frame(frm_jog)
        frm_dir.pack(padx=6, pady=6)

        btn_cfg = dict(width=6)
        ttk.Label(frm_dir, text="X轴").grid(row=0, column=0, columnspan=3)
        ttk.Button(frm_dir, text="▲ 前(W)", **btn_cfg,
                   command=lambda: self._jog(0, +1)).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(frm_dir, text="◀ 左(A)", **btn_cfg,
                   command=lambda: self._jog(1, +1)).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(frm_dir, text="▶ 右(D)", **btn_cfg,
                   command=lambda: self._jog(1, -1)).grid(row=2, column=2, padx=2, pady=2)
        ttk.Button(frm_dir, text="▼ 后(S)", **btn_cfg,
                   command=lambda: self._jog(0, -1)).grid(row=3, column=1, padx=2, pady=2)

        ttk.Label(frm_dir, text="Z轴").grid(row=0, column=4, columnspan=2)
        ttk.Button(frm_dir, text="↑ 上(Q)", **btn_cfg,
                   command=lambda: self._jog(2, +1)).grid(row=1, column=4, columnspan=2, padx=(12,2), pady=2)
        ttk.Button(frm_dir, text="↓ 下(E)", **btn_cfg,
                   command=lambda: self._jog(2, -1)).grid(row=2, column=4, columnspan=2, padx=(12,2), pady=2)

        ttk.Button(frm_jog, text="R  回待机位",
                   command=self._goto_rest).pack(padx=6, pady=(0, 6), fill="x")

        # ── 指定坐标移动 ─────────────────────────────────
        frm_goto = ttk.LabelFrame(self.root, text="移动到指定坐标")
        frm_goto.grid(row=2, column=0, sticky="ew", **PAD)

        entries = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            ttk.Label(frm_goto, text=f"{axis} (mm):").grid(row=i, column=0, sticky="e", padx=6, pady=3)
            var = tk.StringVar()
            ent = ttk.Entry(frm_goto, textvariable=var, width=10)
            ent.grid(row=i, column=1, padx=6, pady=3)
            entries[axis] = var
        self.goto_vars = entries

        ttk.Button(frm_goto, text="移动到该坐标",
                   command=self._goto_target).grid(row=3, column=0, columnspan=2,
                                                    padx=6, pady=6, sticky="ew")

        # ── 记录点 ───────────────────────────────────────
        frm_rec = ttk.LabelFrame(self.root, text="标定点记录（共 4 个点）")
        frm_rec.grid(row=1, column=1, rowspan=2, sticky="nsew", **PAD)

        self.listbox = tk.Listbox(frm_rec, width=32, height=6,
                                  font=("Consolas", 11))
        self.listbox.pack(padx=6, pady=6, fill="both", expand=True)

        btn_frm = ttk.Frame(frm_rec)
        btn_frm.pack(fill="x", padx=6, pady=(0, 6))
        self.btn_record = ttk.Button(btn_frm, text="记录当前点  [空格]",
                                     command=self._record)
        self.btn_record.pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(btn_frm, text="撤销", command=self._undo).pack(side="left")

        ttk.Button(frm_rec, text="完成 — 打印坐标到终端",
                   command=self._finish).pack(padx=6, pady=(0, 6), fill="x")

    # ─── 键盘绑定 ────────────────────────────────────────
    def _bind_keys(self):
        self.root.bind("<KeyPress-w>", lambda e: self._jog(0, +1))
        self.root.bind("<KeyPress-s>", lambda e: self._jog(0, -1))
        self.root.bind("<KeyPress-a>", lambda e: self._jog(1, +1))
        self.root.bind("<KeyPress-d>", lambda e: self._jog(1, -1))
        self.root.bind("<KeyPress-q>", lambda e: self._jog(2, +1))
        self.root.bind("<KeyPress-e>", lambda e: self._jog(2, -1))
        self.root.bind("<KeyPress-r>", lambda e: self._goto_rest())
        self.root.bind("<space>",      lambda e: self._record())

    # ─── 后台连接机械臂 ──────────────────────────────────
    def _connect_arm(self):
        try:
            self.arm = RobotArmUltimate(port=SERIAL_PORT, baud=SERIAL_BAUD)
            self.root.after(0, lambda: self.lbl_status.config(
                text="机械臂已连接 ✓", foreground="green"))
        except Exception as ex:
            self.root.after(0, lambda: self.lbl_status.config(
                text=f"连接失败（仿真模式）: {ex}", foreground="orange"))

    # ─── UI 刷新 ─────────────────────────────────────────
    def _refresh_ui(self):
        self.lbl_coord.config(
            text=f"X={self.cur[0]:7.1f}   Y={self.cur[1]:7.1f}   Z={self.cur[2]:7.1f}  mm")
        self.listbox.delete(0, tk.END)
        for i, pt in enumerate(self.recorded):
            self.listbox.insert(tk.END,
                f"P{i+1}:  X={pt[0]:.1f}  Y={pt[1]:.1f}  Z={pt[2]:.1f}")
        can_record = len(self.recorded) < self.MAX_PTS
        self.btn_record.config(state="normal" if can_record else "disabled")

    # ─── 步进 ────────────────────────────────────────────
    def _on_step_change(self):
        try:
            self.step = float(self.spin_step.get())
        except ValueError:
            pass

    def _jog(self, axis: int, sign: int):
        if self._busy or self.arm is None:
            return
        self.cur[axis] += sign * self.step
        threading.Thread(target=self._send_cur, daemon=True).start()

    def _send_cur(self):
        self._busy = True
        try:
            angles = self.arm.inverse_kinematics(self.cur, target_pitch=-90)
            self.arm._send_and_audit(angles, self.cur)
        finally:
            self._busy = False
            self.root.after(0, self._refresh_ui)

    # ─── 回待机 ──────────────────────────────────────────
    def _goto_rest(self):
        if self._busy or self.arm is None:
            return
        def _move():
            self._busy = True
            try:
                self.arm.move_line(self.cur.copy(), self.rest, duration=2.0)
                self.cur[:] = self.rest
            finally:
                self._busy = False
                self.root.after(0, self._refresh_ui)
        threading.Thread(target=_move, daemon=True).start()

    # ─── 移动到指定坐标 ──────────────────────────────────
    def _goto_target(self):
        if self._busy or self.arm is None:
            return
        try:
            x = float(self.goto_vars["X"].get())
            y = float(self.goto_vars["Y"].get())
            z = float(self.goto_vars["Z"].get())
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字坐标")
            return
        target = np.array([x, y, z])
        def _move():
            self._busy = True
            try:
                self.arm.move_line(self.cur.copy(), target, duration=2.0)
                self.cur[:] = target
            finally:
                self._busy = False
                self.root.after(0, self._refresh_ui)
        threading.Thread(target=_move, daemon=True).start()

    # ─── 记录当前点 ──────────────────────────────────────
    def _record(self):
        if len(self.recorded) >= self.MAX_PTS:
            return
        self.recorded.append(self.cur.copy())
        self._refresh_ui()

    def _undo(self):
        if self.recorded:
            self.recorded.pop()
            self._refresh_ui()

    # ─── 完成 ────────────────────────────────────────────
    def _finish(self):
        if len(self.recorded) < self.MAX_PTS:
            messagebox.showwarning("未完成", f"还需记录 {self.MAX_PTS - len(self.recorded)} 个点")
            return
        print("\n=== 4 个点已全部记录 ===")
        for i, pt in enumerate(self.recorded):
            print(f"  P{i+1}: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
        print("\nCALIB_ROBOT_POINTS（取 X/Y，复制到 config.py）：")
        print("CALIB_ROBOT_POINTS = [")
        for pt in self.recorded:
            print(f"    [{pt[0]:.1f}, {pt[1]:.1f}],")
        print("]")
        messagebox.showinfo("完成", "坐标已打印到终端，请复制到 config.py 的 CALIB_ROBOT_POINTS。")

    # ─── 关闭窗口 ────────────────────────────────────────
    def on_close(self):
        if self.arm:
            try:
                self.arm.move_line(self.cur.copy(), self.rest, duration=2.0)
            except Exception:
                pass
            self.arm.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = CalibrateApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
