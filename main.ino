/*
 * ESP32 原生硬件 PWM 控制程序 (适配 Core 3.x 版本)
 * 解决 'ledcSetup' was not declared 报错问题
 */

// --- 1. 引脚定义 ---
const int PIN_X = 14; 
const int PIN_Y = 4;  
const int PIN_Z = 5;  
const int PIN_B = 18; 
const int PIN_G = 23; 

// --- 2. PWM 参数 (舵机标准) ---
const int freq = 50;           // 频率 50Hz (周期 20ms)
const int resolution = 12;     // 分辨率 12位 (数值范围 0-4095)

// 舵机角度对应的占空比数值 (12位分辨率)
// 0.5ms (0度)   -> 约 102
// 1.5ms (90度)  -> 约 307
// 2.5ms (180度) -> 约 512

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n--- 使用 ESP32 Core 3.x LEDC 驱动初始化 ---");

  // 在新版 Core 中，直接使用 ledcAttach(引脚, 频率, 分辨率)
  ledcAttach(PIN_X, freq, resolution);
  ledcAttach(PIN_Y, freq, resolution);
  ledcAttach(PIN_Z, freq, resolution);
  ledcAttach(PIN_B, freq, resolution);
  ledcAttach(PIN_G, freq, resolution);

  // 初始归中
  writeAngle(PIN_X, 90); delay(500);
  writeAngle(PIN_Y, 90); delay(500);
  writeAngle(PIN_Z, 90); delay(500);
  writeAngle(PIN_B, 90); delay(500);
  writeAngle(PIN_G, 90); delay(500);

  Serial.println("--- 5轴硬件 PWM 初始化完成 ---");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // 根据指令解析并控制
    if (cmd.startsWith("Servo_ArmX")) writeAngle(PIN_X, extractAngle(cmd, "Servo_ArmX"));
    else if (cmd.startsWith("Servo_ArmY")) writeAngle(PIN_Y, extractAngle(cmd, "Servo_ArmY"));
    else if (cmd.startsWith("Servo_ArmZ")) writeAngle(PIN_Z, extractAngle(cmd, "Servo_ArmZ"));
    else if (cmd.startsWith("Servo_ArmB")) writeAngle(PIN_B, extractAngle(cmd, "Servo_ArmB"));
    else if (cmd.startsWith("Servo_Gripper")) {
      int a = extractAngle(cmd, "Servo_Gripper");
      if (a != -1) {
        writeAngle(PIN_G, a);
        Serial.print("夹爪已转动至: "); Serial.println(a);
      }
    }
  }
}

/**
 * 核心控制函数
 * pin: 控制的引脚号
 * angle: 目标角度 (0-180)
 */
void writeAngle(int pin, int angle) {
  if (angle < 0) angle = 0;
  if (angle > 180) angle = 180;
  
  // 线性映射计算占空比
  // 0度 = 102, 180度 = 512
  int duty = (angle * (512 - 102) / 180) + 102;
  
  // 在新版 Core 中，ledcWrite 直接接收引脚号和数值
  ledcWrite(pin, duty);
}

// 提取命令中的数字
int extractAngle(String cmd, String prefix) {
  int prefixLen = prefix.length();
  String valPart = cmd.substring(prefixLen);
  if (valPart.length() > 0) return valPart.toInt();
  return -1;
}
