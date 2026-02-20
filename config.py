# config.py — hardware constants and motion parameters
# All values can be overridden via environment variables.
import os

# ---------------------------------------------------------------------------
# Serial
# ---------------------------------------------------------------------------
SERIAL_PORT = os.environ.get("ROBOT_PORT", "COM3")
SERIAL_BAUD = int(os.environ.get("ROBOT_BAUD", "115200"))

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
LLM_MODEL_PATH  = os.environ.get("LLM_MODEL_PATH",  r"D:\lora\2")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "best.pt")

# ---------------------------------------------------------------------------
# Z-axis key heights (mm), relative to robot base
# ---------------------------------------------------------------------------
Z_HOVER      = float(os.environ.get("Z_HOVER",      "120.0"))  # safe transit height
Z_GRAB       = float(os.environ.get("Z_GRAB",       "-15.0"))  # table / grasp surface
Z_AFTER_PICK = float(os.environ.get("Z_AFTER_PICK", "50.0"))   # clearance after grasp

# ---------------------------------------------------------------------------
# Rest (home) position [x, y, z] in mm
# Exposed as three scalars so config.py has no numpy dependency.
# Callers: np.array([config.REST_X, config.REST_Y, config.REST_Z])
# ---------------------------------------------------------------------------
REST_X = float(os.environ.get("REST_X", "120.0"))
REST_Y = float(os.environ.get("REST_Y", "0.0"))
REST_Z = float(os.environ.get("REST_Z", "60.0"))

# ---------------------------------------------------------------------------
# Workspace hard limits (mm) — movements are clipped to these bounds
# ---------------------------------------------------------------------------
WS_X = (float(os.environ.get("WS_X_MIN", "80.0")),   float(os.environ.get("WS_X_MAX", "250.0")))
WS_Y = (float(os.environ.get("WS_Y_MIN", "-120.0")), float(os.environ.get("WS_Y_MAX", "120.0")))
WS_Z = (float(os.environ.get("WS_Z_MIN", "-20.0")),  float(os.environ.get("WS_Z_MAX", "200.0")))

# ---------------------------------------------------------------------------
# Damping / smoothing
# ---------------------------------------------------------------------------
DAMPING_BUFFER_SIZE = int(os.environ.get("DAMPING_BUFFER",    "3"))
DAMPING_MAX_SPEED   = float(os.environ.get("DAMPING_MAX_SPEED", "25.0"))  # deg/frame
DAMPING_FACTOR      = float(os.environ.get("DAMPING_FACTOR",    "0.6"))

# ---------------------------------------------------------------------------
# Tilt correction offsets (servo degrees)
# Tune OFFSET_Y > 0 if the end-effector droops during horizontal moves.
# ---------------------------------------------------------------------------
OFFSET_Y = float(os.environ.get("OFFSET_Y", "-10.0"))
OFFSET_Z = float(os.environ.get("OFFSET_Z",  "0.0"))
