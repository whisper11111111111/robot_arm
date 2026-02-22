/*
 * ESP32 native hardware PWM servo controller (ESP32 Arduino Core 3.x)
 * Uses the updated ledcAttach/ledcWrite API (replaces deprecated ledcSetup).
 */

// --- 1. Pin assignments ---
const int PIN_X = 14;  // X-axis servo
const int PIN_Y = 4;   // Y-axis servo
const int PIN_Z = 5;   // Z-axis servo
const int PIN_B = 18;  // Base rotation servo
const int PIN_G = 23;  // Gripper servo

// --- 2. PWM parameters (standard servo) ---
const int freq = 50;        // 50 Hz (20 ms period)
const int resolution = 12;  // 12-bit resolution (0–4095)

// Duty-cycle values for servo angles (12-bit, 50 Hz):
//   0.5 ms (  0°) → ~102
//   1.5 ms ( 90°) → ~307
//   2.5 ms (180°) → ~512

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n--- ESP32 Core 3.x LEDC driver initializing ---");

  // Core 3.x API: ledcAttach(pin, frequency, resolution)
  ledcAttach(PIN_X, freq, resolution);
  ledcAttach(PIN_Y, freq, resolution);
  ledcAttach(PIN_Z, freq, resolution);
  ledcAttach(PIN_B, freq, resolution);
  ledcAttach(PIN_G, freq, resolution);

  // Center all servos at startup
  writeAngle(PIN_X, 90); delay(500);
  writeAngle(PIN_Y, 90); delay(500);
  writeAngle(PIN_Z, 90); delay(500);
  writeAngle(PIN_B, 90); delay(500);
  writeAngle(PIN_G, 90); delay(500);

  Serial.println("--- 5-axis hardware PWM ready ---");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if      (cmd.startsWith("Servo_ArmX"))    writeAngle(PIN_X, extractAngle(cmd, "Servo_ArmX"));
    else if (cmd.startsWith("Servo_ArmY"))    writeAngle(PIN_Y, extractAngle(cmd, "Servo_ArmY"));
    else if (cmd.startsWith("Servo_ArmZ"))    writeAngle(PIN_Z, extractAngle(cmd, "Servo_ArmZ"));
    else if (cmd.startsWith("Servo_ArmB"))    writeAngle(PIN_B, extractAngle(cmd, "Servo_ArmB"));
    else if (cmd.startsWith("Servo_Gripper")) {
      int a = extractAngle(cmd, "Servo_Gripper");
      if (a != -1) {
        writeAngle(PIN_G, a);
        Serial.print("Gripper moved to: "); Serial.println(a);
      }
    }
  }
}

/**
 * Write a target angle to a servo pin.
 * pin  : GPIO pin number
 * angle: target angle in degrees (0–180, clamped)
 */
void writeAngle(int pin, int angle) {
  if (angle < 0)   angle = 0;
  if (angle > 180) angle = 180;

  // Linear map: 0° → duty 102, 180° → duty 512
  int duty = (angle * (512 - 102) / 180) + 102;

  // Core 3.x API: ledcWrite takes pin number directly
  ledcWrite(pin, duty);
}

/** Extract the numeric argument after a known command prefix. Returns -1 on failure. */
int extractAngle(String cmd, String prefix) {
  String valPart = cmd.substring(prefix.length());
  if (valPart.length() > 0) return valPart.toInt();
  return -1;
}
