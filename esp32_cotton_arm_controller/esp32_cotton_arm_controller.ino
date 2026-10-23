#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>

// =====================================================
// WIFI ACCESS POINT
// =====================================================

const char* AP_SSID = "ESP32-RobotArm";
const char* AP_PASSWORD = "";

IPAddress local_IP(192, 168, 4, 1);
IPAddress gateway(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

WebServer server(80);

// =====================================================
// SERVO OBJECTS
// =====================================================

Servo motorB;  // Base - GPIO12
Servo motorS;  // Shoulder - GPIO13
Servo motorE;  // Elbow - GPIO27
Servo motorA;  // Arm - GPIO14

Servo gripper1;
Servo gripper2;
Servo gripper3;

// =====================================================
// SERVO PINS
// =====================================================

const int PIN_B = 12;
const int PIN_S = 13;
const int PIN_E = 27;
const int PIN_A = 14;

const int PIN_GRIPPER1 = 26;
const int PIN_GRIPPER2 = 25;
const int PIN_GRIPPER3 = 33;

// =====================================================
// TB6612FNG DC MOTOR PINS
// =====================================================

const int DC_PWM  = 18;
const int DC_AIN2 = 19;
const int DC_AIN1 = 21;
const int DC_STBY = 23;

// =====================================================
// SETTINGS
// =====================================================

const int MIN_US = 500;
const int MAX_US = 2500;

const int DC_SPEED = 150;

// Speed = milliseconds per degree (larger = slower).
// Slower speed (85 ms/deg = ~11.7 deg/sec) prevents camera shaking and eliminates vibration.
const int DEFAULT_SPEED_DELAY = 85;
const int MIN_SPEED_DELAY = 30;
const int MAX_SPEED_DELAY = 150;

// Smooth motion profile
const float ACCEL_DEG_S2 = 35.0f;     // gentle ramp up / ramp down rate, eliminates shaking
const unsigned long TICK_MS = 20;     // one update per servo PWM frame (50 Hz)

// =====================================================
// HOME POSITIONS
// =====================================================

const int HOME_B = 90;
const int HOME_S = 130;
const int HOME_E = 130;
const int HOME_A = 130;

// =====================================================
// WEB SERVO ANGLE LIMITS
// =====================================================

const int BASE_MIN = 20;
const int BASE_MAX = 130;

const int SHOULDER_MIN = 50;
const int SHOULDER_MAX = 130;

const int ELBOW_MIN = 50;
const int ELBOW_MAX = 130;

const int ARM_MIN = 50;
const int ARM_MAX = 130;

const int GRIPPER_OPEN = 60;
const int GRIPPER_CLOSE = 10;

// =====================================================
// AXES
// =====================================================

enum Axis : uint8_t {
  
  AX_B = 0,
  AX_S,
  AX_E,
  AX_A,
  AX_G,
  AXIS_COUNT
};

const char AXIS_CHAR[AXIS_COUNT] = { 'B', 'S', 'E', 'A', 'G' };

const int AXIS_MIN[AXIS_COUNT] = {
  BASE_MIN, SHOULDER_MIN, ELBOW_MIN, ARM_MIN, GRIPPER_CLOSE
};

const int AXIS_MAX[AXIS_COUNT] = {
  BASE_MAX, SHOULDER_MAX, ELBOW_MAX, ARM_MAX, GRIPPER_OPEN
};

const int AXIS_HOME[AXIS_COUNT] = {
  HOME_B, HOME_S, HOME_E, HOME_A, GRIPPER_OPEN
};

// Sequence step types. These must be declared up here, before any
// function, because the Arduino IDE auto-generates function prototypes
// at the top of the file and they need to know about Step.
enum StepType : uint8_t {
  ST_MOVE,      // move axis to angle, finish when it arrives
  ST_WAIT,      // wait value ms
  ST_DC_CW,
  ST_DC_CCW,
  ST_DC_STOP,
  ST_MOVE_PITCH_HOME, // Retract & Lift all pitch joints (E, A, S) simultaneously to Home
  ST_MOVE_HOME_ALL    // Move Base to 90 and all axes simultaneously to Home
};

struct Step {
  StepType type;
  uint8_t axis;
  int value;
};

float axisPos[AXIS_COUNT];      // current position (deg, fractional)
float axisTarget[AXIS_COUNT];   // where it is heading
float axisVel[AXIS_COUNT];      // current velocity (deg/s)
int   speedDelay[AXIS_COUNT];   // ms per degree, user controlled
int   lastPulseUs[AXIS_COUNT];

// The ESP32 does not output any servo signal until the first
// interaction from the web page (or Serial Monitor).
bool armed = false;
bool dcReady = false;
unsigned long lastTick = 0;

// =====================================================
// EMBEDDED WEB INTERFACE
// =====================================================

const char INDEX_HTML[] PROGMEM = R"ROBOTPAGE(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32 Robot Arm Controller</title>

<style>
:root {
  --bg:#0f172a;
  --card:#1e293b;
  --accent:#38bdf8;
  --amber:#f59e0b;
  --text:#f8fafc;
}

body {
  font-family:system-ui,sans-serif;
  background:var(--bg);
  color:var(--text);
  margin:0;
  padding:12px;
}

h2 {
  font-size:1.1rem;
  color:var(--accent);
  text-align:center;
}

.header-info {
  text-align:center;
  font-size:.8rem;
  color:#94a3b8;
  margin-bottom:12px;
}

.btn-row {
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  justify-content:center;
  margin-bottom:12px;
}

button {
  background:#334155;
  border:1px solid var(--accent);
  color:white;
  padding:8px 12px;
  border-radius:6px;
  font-weight:600;
  cursor:pointer;
  font-size:.8rem;
}

button:active {
  transform:scale(.96);
}

.btn-primary { background:#0284c7; }

.btn-danger {
  background:#e11d48;
  border-color:#f43f5e;
}

.btn-success {
  background:#059669;
  border-color:#10b981;
}

.btn-amber {
  background:#b45309;
  border-color:var(--amber);
}

.grid {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
  gap:10px;
  max-width:900px;
  margin:auto;
}

.card,.save-card {
  background:var(--card);
  border:1px solid #334155;
  border-radius:8px;
  padding:12px;
}

.servo-title {
  display:flex;
  justify-content:space-between;
  gap:8px;
  font-size:.85rem;
  font-weight:700;
  margin-bottom:4px;
}

.val-badge {
  color:var(--accent);
  font-family:monospace;
  font-size:.95rem;
}

.spd-badge {
  color:var(--amber);
  font-family:monospace;
  font-size:.8rem;
}

input[type=range] {
  width:100%;
  margin:8px 0;
  accent-color:var(--accent);
}

.spd-range {
  accent-color:var(--amber)!important;
}

.spd-row {
  display:flex;
  justify-content:space-between;
  font-size:.75rem;
  color:#94a3b8;
  margin-top:6px;
}

.save-card {
  max-width:900px;
  margin:12px auto;
}

.save-title {
  font-weight:700;
  color:var(--accent);
  margin-bottom:8px;
}

.save-form,.run-form {
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-bottom:8px;
}

.save-form input,.run-form input {
  flex:1;
  min-width:170px;
  background:#0f172a;
  border:1px solid #334155;
  color:white;
  padding:9px;
  border-radius:6px;
}

.saved-list-title {
  font-size:.8rem;
  color:#94a3b8;
  margin-top:12px;
  margin-bottom:6px;
}

.pose-tags {
  display:flex;
  flex-wrap:wrap;
  gap:6px;
}

.pose-chip {
  background:#334155;
  border:1px solid #475569;
  padding:8px;
  border-radius:6px;
  font-size:.8rem;
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:8px;
}

.pose-chip-info {
  color:#cbd5e1;
  font-family:monospace;
  font-size:.75rem;
}

#message {
  text-align:center;
  color:#94a3b8;
  font-size:.8rem;
  margin:8px;
}
</style>
</head>

<body>

<h2>ESP32 Robot Arm Controller</h2>

<div class="header-info">
  Wi-Fi: <strong>ESP32-RobotArm</strong> |
  IP: <strong>192.168.4.1</strong>
  <br>
  <span id="connection">Connecting to ESP32...</span>
</div>

<div class="btn-row">

  <button class="btn-primary" onclick="runSeq('2','Running: Complete Pick/Release')">
    1 - Complete Pick/Release
  </button>

  <button class="btn-danger" onclick="runSeq('S','Stopping...')">
    S - Stop
  </button>

  <button class="btn-primary" onclick="runSeq('3','Running: Base Search')">
    2 - Base Search
  </button>

  <button class="btn-success" onclick="runSeq('y','Running: Pick &amp; CW')">
    y - Pick &amp; CW
  </button>

  <button class="btn-primary" onclick="runSeq('s','Running: Drop &amp; CCW')">
    s - Drop &amp; CCW
  </button>

  <button onclick="runSeq('H','Going home: Elbow, Arm, Shoulder, Base')">
    H - Home
  </button>

  <button class="btn-amber" onclick="homeSpeed()">
    Home Speed (45)
  </button>

</div>

<div id="message">Ready</div>

<div class="grid">

  <!-- BASE: B -->

  <div class="card">
    <div class="servo-title">
      <span>B - Base 60 kg</span>
      <span class="val-badge" id="vB">90&deg;</span>
    </div>

    <input type="range" id="rB"
      min="20" max="130" value="90"
      oninput="setAngle('B',this.value)">

    <div class="spd-row">
      <span>Speed delay (45 fastest)</span>
      <span class="spd-badge" id="sB">45 ms/deg</span>
    </div>

    <input type="range" class="spd-range" id="rsB"
      min="45" max="55" value="45"
      oninput="setSpeed('B',this.value)">
  </div>

  <!-- SHOULDER: S -->

  <div class="card">
    <div class="servo-title">
      <span>S - Shoulder 30 kg</span>
      <span class="val-badge" id="vS">130&deg;</span>
    </div>

    <input type="range" id="rS"
      min="50" max="130" value="130"
      oninput="setAngle('S',this.value)">

    <div class="spd-row">
      <span>Speed delay (45 fastest)</span>
      <span class="spd-badge" id="sS">45 ms/deg</span>
    </div>

    <input type="range" class="spd-range" id="rsS"
      min="45" max="55" value="45"
      oninput="setSpeed('S',this.value)">
  </div>

  <!-- ELBOW: E -->

  <div class="card">
    <div class="servo-title">
      <span>E - Elbow 25 kg</span>
      <span class="val-badge" id="vE">130&deg;</span>
    </div>

    <input type="range" id="rE"
      min="50" max="130" value="130"
      oninput="setAngle('E',this.value)">

    <div class="spd-row">
      <span>Speed delay (45 fastest)</span>
      <span class="spd-badge" id="sE">45 ms/deg</span>
    </div>

    <input type="range" class="spd-range" id="rsE"
      min="45" max="55" value="45"
      oninput="setSpeed('E',this.value)">
  </div>

  <!-- ARM: A -->

  <div class="card">
    <div class="servo-title">
      <span>A - Arm 15 kg</span>
      <span class="val-badge" id="vA">130&deg;</span>
    </div>

    <input type="range" id="rA"
      min="50" max="130" value="130"
      oninput="setAngle('A',this.value)">

    <div class="spd-row">
      <span>Speed delay (45 fastest)</span>
      <span class="spd-badge" id="sA">45 ms/deg</span>
    </div>

    <input type="range" class="spd-range" id="rsA"
      min="45" max="55" value="45"
      oninput="setSpeed('A',this.value)">
  </div>

  <!-- GRIPPER: G -->

  <div class="card">
    <div class="servo-title">
      <span>G - Gripper (3 x MG90S)</span>
      <span class="val-badge" id="vG">60&deg;</span>
    </div>

    <input type="range" id="rG"
      min="10" max="60" value="60"
      oninput="setAngle('G',this.value)">

    <div class="spd-row">
      <span>Speed delay (45 fastest)</span>
      <span class="spd-badge" id="sG">45 ms/deg</span>
    </div>

    <input type="range" class="spd-range" id="rsG"
      min="45" max="55" value="45"
      oninput="setSpeed('G',this.value)">
  </div>

</div>

<!-- SAVED POSITIONS -->

<div class="save-card">

  <div class="save-title">Saved Arm Positions</div>

  <div class="save-form">
    <input type="text" id="poseNameInput"
      placeholder="Position name, e.g. Pickup1">

    <button class="btn-success" onclick="saveCurrentPose()">
      Save Current Angles
    </button>
  </div>

  <div class="run-form">
    <input type="text" id="runNameInput"
      placeholder="Enter saved position name">

    <button class="btn-primary" onclick="runPoseByName()">
      Run Position
    </button>
  </div>

  <div class="saved-list-title">Saved Positions:</div>
  <div id="savedPoseList" class="pose-tags"></div>

</div>

<script>
let sendTimers = {};
let speedTimers = {};

// motor -> {v: target angle the user asked for, t: time}
// While a motor is heading to what the user asked for,
// the status poll must not drag the slider back.
const pendingAngle = {};
let lastSpeedInput = 0;

const motors = ['B', 'S', 'E', 'A', 'G'];
const DEFAULT_SPEED = 45;

function showMessage(text) {
  document.getElementById('message').innerText = text;
}

// The GUI is always updated BEFORE this is called.
// The request is sent on the next frame, after the page has painted.
function sendCmd(command) {
  requestAnimationFrame(() => {
    setTimeout(() => {
      fetch('/cmd?cmd=' + encodeURIComponent(command))
        .then(response => {
          if (!response.ok) throw new Error('HTTP error');
          return response.text();
        })
        .then(text => {
          showMessage(text || ('Sent: ' + command));
        })
        .catch(() => {
          showMessage('ESP32 request failed. Check Wi-Fi connection.');
        });
    }, 0);
  });
}

// Buttons that start a sequence / stop
function runSeq(command, text) {
  motors.forEach(m => delete pendingAngle[m]);
  showMessage(text);
  sendCmd(command);
}

function setAngle(motor, value) {
  // 1) GUI first
  document.getElementById('v' + motor).innerText = value + '\u00b0';
  pendingAngle[motor] = { v: Number(value), t: Date.now() };

  // 2) then ESP32
  clearTimeout(sendTimers[motor]);

  sendTimers[motor] = setTimeout(() => {
    sendCmd('ANGLE ' + motor + ' ' + value);
  }, 60);
}

function setSpeed(motor, value) {
  document.getElementById('s' + motor).innerText = value + ' ms/deg';
  lastSpeedInput = Date.now();

  clearTimeout(speedTimers[motor]);

  speedTimers[motor] = setTimeout(() => {
    sendCmd('SPD ' + motor + ' ' + value);
  }, 100);
}

// Reset every motor to the default 45 ms/deg
function homeSpeed() {
  motors.forEach(m => {
    document.getElementById('rs' + m).value = DEFAULT_SPEED;
    document.getElementById('s' + m).innerText = DEFAULT_SPEED + ' ms/deg';
  });

  lastSpeedInput = Date.now();
  showMessage('Speed reset to 45 ms/deg');
  sendCmd('HSPEED');
}

function syncStatus() {
  fetch('/status')
    .then(response => response.json())
    .then(data => {
      const now = Date.now();

      motors.forEach(m => {
        // Angles
        if (data[m] !== undefined) {
          const p = pendingAngle[m];
          let hold = false;

          if (p) {
            if (Math.abs(data[m] - p.v) <= 1 || now - p.t > 20000) {
              delete pendingAngle[m];
            } else {
              hold = true;
            }
          }

          if (!hold) {
            document.getElementById('r' + m).value = data[m];
            document.getElementById('v' + m).innerText = data[m] + '\u00b0';
          }
        }

        // Speeds
        if (data['s' + m] !== undefined && now - lastSpeedInput > 1500) {
          document.getElementById('rs' + m).value = data['s' + m];
          document.getElementById('s' + m).innerText =
            data['s' + m] + ' ms/deg';
        }
      });

      document.getElementById('connection').innerText =
        data.busy ? 'ESP32 connected - sequence running'
                  : 'ESP32 connected';
    })
    .catch(() => {
      document.getElementById('connection').innerText =
        'ESP32 not responding';
    });
}

function getSavedPoses() {
  try {
    return JSON.parse(
      localStorage.getItem('robotPoses') || '{}'
    );
  } catch (error) {
    return {};
  }
}

function renderSavedPoses() {
  const list = document.getElementById('savedPoseList');
  const poses = getSavedPoses();
  const names = Object.keys(poses);

  if (names.length === 0) {
    list.innerHTML =
      '<span style="color:#94a3b8;font-size:.8rem;">' +
      'No saved positions yet.</span>';
    return;
  }

  list.innerHTML = '';

  names.forEach(name => {
    const pose = poses[name];

    const chip = document.createElement('div');
    chip.className = 'pose-chip';

    const title = document.createElement('strong');
    title.textContent = name;

    const info = document.createElement('span');
    info.className = 'pose-chip-info';

    info.textContent =
      'B:' + pose.B + '\u00b0 ' +
      'S:' + pose.S + '\u00b0 ' +
      'E:' + pose.E + '\u00b0 ' +
      'A:' + pose.A + '\u00b0 ' +
      'G:' + pose.G + '\u00b0';

    const runButton = document.createElement('button');
    runButton.className = 'btn-primary';
    runButton.textContent = 'Run';
    runButton.onclick = () => applyPose(name);

    const deleteButton = document.createElement('button');
    deleteButton.className = 'btn-danger';
    deleteButton.textContent = 'Delete';
    deleteButton.onclick = () => deletePose(name);

    chip.append(title, info, runButton, deleteButton);
    list.appendChild(chip);
  });
}

function readCurrentPose() {
  return {
    B: Number(document.getElementById('rB').value),
    S: Number(document.getElementById('rS').value),
    E: Number(document.getElementById('rE').value),
    A: Number(document.getElementById('rA').value),
    G: Number(document.getElementById('rG').value)
  };
}

function saveCurrentPose() {
  const input = document.getElementById('poseNameInput');
  const name = input.value.trim();

  if (!name) {
    alert('Please enter a position name.');
    return;
  }

  const poses = getSavedPoses();
  poses[name] = readCurrentPose();

  localStorage.setItem('robotPoses', JSON.stringify(poses));

  input.value = '';
  renderSavedPoses();
  showMessage('Saved position: ' + name);
}

function applyPose(name) {
  const poses = getSavedPoses();
  const pose = poses[name];

  if (!pose) {
    alert('Saved position not found: ' + name);
    return;
  }

  // GUI first
  motors.forEach(motor => {
    document.getElementById('r' + motor).value = pose[motor];
    document.getElementById('v' + motor).innerText =
      pose[motor] + '\u00b0';
    pendingAngle[motor] = { v: Number(pose[motor]), t: Date.now() };
  });

  showMessage('Running saved position: ' + name);

  // then one single command moves all axes together, smoothly
  sendCmd(
    'POSE ' +
    pose.B + ' ' + pose.S + ' ' + pose.E + ' ' +
    pose.A + ' ' + pose.G
  );
}

function runPoseByName() {
  const input = document.getElementById('runNameInput');
  const name = input.value.trim();

  if (!name) {
    alert('Please enter a position name.');
    return;
  }

  applyPose(name);
  input.value = '';
}

function deletePose(name) {
  const poses = getSavedPoses();

  delete poses[name];
  localStorage.setItem('robotPoses', JSON.stringify(poses));

  renderSavedPoses();
  showMessage('Deleted position: ' + name);
}

window.addEventListener('DOMContentLoaded', () => {
  renderSavedPoses();
  syncStatus();
  setInterval(syncStatus, 500);
});
</script>

</body>
</html>
)ROBOTPAGE";

// =====================================================
// LOW LEVEL SERVO OUTPUT
// =====================================================

void writeAxis(int i, bool force = false) {
  int us = MIN_US +
           (int)lroundf(axisPos[i] * (MAX_US - MIN_US) / 180.0f);

  if (!force && us == lastPulseUs[i]) return;
  lastPulseUs[i] = us;

  switch (i) {
    case AX_B: motorB.writeMicroseconds(us); break;
    case AX_S: motorS.writeMicroseconds(us); break;
    case AX_E: motorE.writeMicroseconds(us); break;
    case AX_A: motorA.writeMicroseconds(us); break;
    case AX_G:
      gripper1.writeMicroseconds(us);
      gripper2.writeMicroseconds(us);
      gripper3.writeMicroseconds(us);
      break;
  }
}

// =====================================================
// DC MOTOR CONTROL
// =====================================================

void dcMotorClockwise() {
  digitalWrite(DC_STBY, HIGH);
  digitalWrite(DC_AIN1, HIGH);
  digitalWrite(DC_AIN2, LOW);

  if (dcReady) ledcWrite(DC_PWM, DC_SPEED);
}

void dcMotorAnticlockwise() {
  digitalWrite(DC_STBY, HIGH);
  digitalWrite(DC_AIN1, LOW);
  digitalWrite(DC_AIN2, HIGH);

  if (dcReady) ledcWrite(DC_PWM, DC_SPEED);
}

void dcMotorStop() {
  if (dcReady) ledcWrite(DC_PWM, 0);

  digitalWrite(DC_AIN1, LOW);
  digitalWrite(DC_AIN2, LOW);
  digitalWrite(DC_STBY, LOW);
}

// =====================================================
// ARMING - the first interaction attaches the servos
// and starts sending signals. Nothing is sent before.
// =====================================================

void ensureArmed() {
  if (armed) return;

  // CRITICAL: Allocate all 4 LEDC timers to support up to 16 servos on ESP32!
  // Without this, ESP32Servo only allows Timer 0 (max 4 servos), causing
  // gripper2 (GPIO 25) and gripper3 (GPIO 33) to fail to attach or clash with DC PWM!
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  motorB.setPeriodHertz(50);
  motorS.setPeriodHertz(50);
  motorE.setPeriodHertz(50);
  motorA.setPeriodHertz(50);

  gripper1.setPeriodHertz(50);
  gripper2.setPeriodHertz(50);
  gripper3.setPeriodHertz(50);

  motorB.attach(PIN_B, MIN_US, MAX_US);
  motorS.attach(PIN_S, MIN_US, MAX_US);
  motorE.attach(PIN_E, MIN_US, MAX_US);
  motorA.attach(PIN_A, MIN_US, MAX_US);

  gripper1.attach(PIN_GRIPPER1, MIN_US, MAX_US);
  gripper2.attach(PIN_GRIPPER2, MIN_US, MAX_US);
  gripper3.attach(PIN_GRIPPER3, MIN_US, MAX_US);

  // Hold the tracked positions (home at power-up)
  for (int i = 0; i < AXIS_COUNT; i++) {
    writeAxis(i, true);
  }

  // DC motor PWM is attached AFTER the servos so the LEDC
  // channels are never shared with the servos.
  dcReady = ledcAttach(DC_PWM, 1000, 8);

  if (!dcReady) {
    Serial.println("ERROR: DC PWM setup failed");
  }

  dcMotorStop();

  lastTick = millis();
  armed = true;

  Serial.println("First interaction: servo signals started");
}

// =====================================================
// SMOOTH MOTION ENGINE (non-blocking)
// Each axis accelerates, cruises at its set speed
// (ms per degree) and decelerates into its target.
// =====================================================

void setTarget(int axis, float degrees) {
  axisTarget[axis] = constrain(degrees, 0.0f, 180.0f);
}

bool axisDone(int axis) {
  return fabsf(axisTarget[axis] - axisPos[axis]) < 0.001f;
}

// Bring every axis to an immediate halt with ZERO servo movement
void brakeAll() {
  for (int i = 0; i < AXIS_COUNT; i++) {
    axisTarget[i] = axisPos[i];
    axisVel[i] = 0.0f;
    writeAxis(i);
  }
}

void updateMotion() {
  if (!armed) return;

  unsigned long now = millis();
  unsigned long elapsed = now - lastTick;

  if (elapsed < TICK_MS) return;
  lastTick = now;

  float dt = elapsed / 1000.0f;

  for (int i = 0; i < AXIS_COUNT; i++) {
    float dist = axisTarget[i] - axisPos[i];

    if (fabsf(dist) < 0.0005f && fabsf(axisVel[i]) < 0.0005f) {
      axisPos[i] = axisTarget[i];
      axisVel[i] = 0;
      continue;
    }

    float dir = dist > 0 ? 1.0f : -1.0f;
    float absDist = fabsf(dist);
    float vMax = 1000.0f / speedDelay[i];   // deg/s

    // Smooth proportional deceleration taper within 2.0 degrees of target (ELIMINATES ARRIVAL JERK):
    float vWanted;
    if (absDist < 2.0f) {
      vWanted = absDist * (vMax / 2.0f);
    } else {
      vWanted = sqrtf(2.0f * ACCEL_DEG_S2 * absDist);
    }
    if (vWanted > vMax) vWanted = vMax;
    vWanted *= dir;

    float dv = ACCEL_DEG_S2 * dt;

    if (vWanted > axisVel[i] + dv) axisVel[i] += dv;
    else if (vWanted < axisVel[i] - dv) axisVel[i] -= dv;
    else axisVel[i] = vWanted;

    axisPos[i] += axisVel[i] * dt;

    float newDist = axisTarget[i] - axisPos[i];

    if (fabsf(newDist) < 0.08f || (newDist * dist <= 0.0f)) {
      axisPos[i] = axisTarget[i];
      axisVel[i] = 0.0f;
    }

    writeAxis(i);
  }
}

// =====================================================
// SEQUENCE ENGINE (non-blocking)
// =====================================================

// (StepType and Step are declared near the top of the file)

// Y: pick & clockwise
const Step SEQ_PICK[] = {
  { ST_MOVE, AX_G, GRIPPER_CLOSE },
  { ST_WAIT, 0, 1000 },
  { ST_DC_CW, 0, 0 },
  { ST_WAIT, 0, 1000 },
  { ST_DC_STOP, 0, 0 }
};

// S: drop & anticlockwise
const Step SEQ_DROP[] = {
  { ST_MOVE, AX_G, GRIPPER_OPEN },
  { ST_WAIT, 0, 1000 },
  { ST_DC_CCW, 0, 0 },
  { ST_WAIT, 0, 1000 },
  { ST_DC_STOP, 0, 0 }
};

// Command 2: complete pick / release
const Step SEQ_COMPLETE[] = {
  // Y sequence
  { ST_MOVE, AX_G, GRIPPER_CLOSE },
  { ST_WAIT, 0, 1000 },
  { ST_DC_CW, 0, 0 },
  { ST_WAIT, 0, 1000 },
  { ST_DC_STOP, 0, 0 },

  { ST_WAIT, 0, 1000 },
  { ST_WAIT, 0, 1000 },

  { ST_MOVE, AX_S, 140 },
  { ST_MOVE, AX_B, 20 },
  { ST_MOVE, AX_S, 70 },
  { ST_MOVE, AX_E, 140 },

  // S sequence (arm A intentionally not moved)
  { ST_MOVE, AX_G, GRIPPER_OPEN },
  { ST_WAIT, 0, 1000 },
  { ST_DC_CCW, 0, 0 },
  { ST_WAIT, 0, 1000 },
  { ST_DC_STOP, 0, 0 }
};

// Proper Homing Concept (Zero Jerk, Base 90 Immediate):
// Base centers to 90 deg along with all pitch joints (S140, A130, E140)
const Step SEQ_HOME[] = {
  { ST_MOVE_HOME_ALL, 0, 0 },
  { ST_MOVE, AX_G, GRIPPER_OPEN },
  { ST_DC_STOP, 0, 0 }
};

// Command 1: elbow 130 FIRST, then shoulder 100, then base sweep 50 <-> 100
// with the shoulder moving 80 <-> 110 at each end (loops from step 2)
const Step SEQ_SEARCH1[] = {
  { ST_MOVE, AX_E, 130 },  // Elbow moves FIRST!
  { ST_MOVE, AX_S, 100 },

  { ST_MOVE, AX_B, 100 },
  { ST_MOVE, AX_S, 80 },
  { ST_MOVE, AX_S, 110 },

  { ST_MOVE, AX_B, 50 },
  { ST_MOVE, AX_S, 80 },
  { ST_MOVE, AX_S, 110 }
};

// Command 3: elbow 130 FIRST, then shoulder 100, then base oscillation 50 <-> 100
const Step SEQ_SEARCH3[] = {
  { ST_MOVE, AX_E, 130 },  // Elbow moves FIRST!
  { ST_MOVE, AX_S, 100 },

  { ST_MOVE, AX_B, 100 },
  { ST_MOVE, AX_B, 50 }
};

#define SEQ_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))

const Step* seqSteps = nullptr;
int seqLen = 0;
int seqIdx = 0;
int seqLoopFrom = -1;
bool seqActive = false;
bool stepRunning = false;
unsigned long stepStart = 0;

void cancelSequence() {
  if (!seqActive) return;

  seqActive = false;
  stepRunning = false;

  dcMotorStop();
  brakeAll();

  Serial.println("Sequence stopped");
}

void stopAll() {
  seqActive = false;
  stepRunning = false;

  dcMotorStop();

  if (armed) brakeAll();

  Serial.println("Stopped");
}

void startSequence(const Step* steps, int len, int loopFrom,
                   const char* name) {
  ensureArmed();
  cancelSequence();

  Serial.println();
  Serial.print("===== ");
  Serial.print(name);
  Serial.println(" STARTED =====");

  seqSteps = steps;
  seqLen = len;
  seqLoopFrom = loopFrom;
  seqIdx = 0;
  stepRunning = false;
  seqActive = true;
}

void beginStep(const Step &s) {
  switch (s.type) {
    case ST_MOVE:
      setTarget(s.axis, s.value);
      break;

    case ST_MOVE_PITCH_HOME:
      // Retract and Lift all 3 pitch axes simultaneously into Home pose (140, 130, 140)
      setTarget(AX_E, HOME_E);
      setTarget(AX_A, HOME_A);
      setTarget(AX_S, HOME_S);
      break;

    case ST_MOVE_HOME_ALL:
      // Immediately move Base to 90 and all pitch axes to Home
      setTarget(AX_B, HOME_B);
      setTarget(AX_E, HOME_E);
      setTarget(AX_A, HOME_A);
      setTarget(AX_S, HOME_S);
      break;

    case ST_WAIT:
      stepStart = millis();
      break;

    case ST_DC_CW:
      dcMotorClockwise();
      break;

    case ST_DC_CCW:
      dcMotorAnticlockwise();
      break;

    case ST_DC_STOP:
      dcMotorStop();
      break;
  }
}

bool stepFinished(const Step &s) {
  switch (s.type) {
    case ST_MOVE:
      return axisDone(s.axis);

    case ST_MOVE_PITCH_HOME:
      return axisDone(AX_E) && axisDone(AX_A) && axisDone(AX_S);

    case ST_MOVE_HOME_ALL:
      return axisDone(AX_B) && axisDone(AX_E) && axisDone(AX_A) && axisDone(AX_S);

    case ST_WAIT:
      return millis() - stepStart >= (unsigned long)s.value;

    default:
      return true;
  }
}

void updateSequence() {
  if (!seqActive) return;

  if (!stepRunning) {
    if (seqIdx >= seqLen) {
      if (seqLoopFrom >= 0 && seqLoopFrom < seqLen) {
        seqIdx = seqLoopFrom;
      } else {
        seqActive = false;
        Serial.println("===== SEQUENCE COMPLETE =====");
        return;
      }
    }

    beginStep(seqSteps[seqIdx]);
    stepRunning = true;
  }

  if (stepFinished(seqSteps[seqIdx])) {
    stepRunning = false;
    seqIdx++;
  }
}

// =====================================================
// SEQUENCE SHORTCUTS
// =====================================================

void startSearchSequence() {
  startSequence(SEQ_SEARCH1, SEQ_LEN(SEQ_SEARCH1), 2, "COMMAND 1");
}

void startCommand3Sequence() {
  startSequence(SEQ_SEARCH3, SEQ_LEN(SEQ_SEARCH3), 2, "COMMAND 3");
}

void runCommand2() {
  startSequence(SEQ_COMPLETE, SEQ_LEN(SEQ_COMPLETE), -1, "COMMAND 2");
}

void clockwiseAndClose() {
  startSequence(SEQ_PICK, SEQ_LEN(SEQ_PICK), -1, "PICK & CW");
}

void anticlockwiseAndOpen() {
  startSequence(SEQ_DROP, SEQ_LEN(SEQ_DROP), -1, "DROP & CCW");
}

void goHome() {
  cancelSequence();
  ensureArmed();
  setTarget(AX_B, HOME_B);
  startSequence(SEQ_HOME, SEQ_LEN(SEQ_HOME), -1,
                "HOME (Base 90 & All Joints Home)");
}

// =====================================================
// DIRECT AXIS CONTROL
// =====================================================

int axisIndex(char motor) {
  motor = toupper(motor);

  for (int i = 0; i < AXIS_COUNT; i++) {
    if (AXIS_CHAR[i] == motor) return i;
  }

  return -1;
}

// Web: limited to the safe range of each axis
bool setServoAngle(char motor, int angle) {
  int i = axisIndex(motor);

  if (i < 0) return false;
  if (angle < AXIS_MIN[i] || angle > AXIS_MAX[i]) return false;

  ensureArmed();
  cancelSequence();
  setTarget(i, angle);

  return true;
}

bool setServoSpeed(char motor, int delayMs) {
  if (delayMs < MIN_SPEED_DELAY || delayMs > MAX_SPEED_DELAY) {
    return false;
  }

  int i = axisIndex(motor);
  if (i < 0) return false;

  speedDelay[i] = delayMs;

  Serial.print("Speed for ");
  Serial.print(AXIS_CHAR[i]);
  Serial.print(" = ");
  Serial.print(delayMs);
  Serial.println(" ms/degree");

  return true;
}

void resetAllSpeeds() {
  for (int i = 0; i < AXIS_COUNT; i++) {
    speedDelay[i] = DEFAULT_SPEED_DELAY;
  }

  Serial.println("All speeds reset to default");
}

// =====================================================
// WEB COMMAND HANDLER
// =====================================================

void handleWebCommand() {
  if (!server.hasArg("cmd")) {
    server.send(400, "text/plain", "Missing cmd");
    return;
  }

  String command = server.arg("cmd");
  command.trim();

  Serial.print("Web command: ");
  Serial.println(command);

  if (command == "1") {
    startSearchSequence();
    server.send(200, "text/plain", "Command 1 started");
    return;
  }

  if (command == "2") {
    runCommand2();
    server.send(
      200,
      "text/plain",
      "Command 2 started: Complete Pick/Release"
    );
    return;
  }

  if (command == "S") {
    stopAll();
    server.send(200, "text/plain", "Stopped");
    return;
  }

  if (command == "3") {
    startCommand3Sequence();
    server.send(200, "text/plain", "Base search started");
    return;
  }

  if (command == "y") {
    clockwiseAndClose();
    server.send(200, "text/plain", "Pick & CW started");
    return;
  }

  if (command == "s") {
    anticlockwiseAndOpen();
    server.send(200, "text/plain", "Drop & CCW started");
    return;
  }

  if (command == "H") {
    goHome();
    server.send(200, "text/plain",
                "Going home: Elbow, Arm, Shoulder, Base");
    return;
  }

  if (command == "HSPEED") {
    resetAllSpeeds();
    server.send(200, "text/plain", "Speed reset to 45 ms/deg");
    return;
  }

  // POSE b s e a g  - all axes at once
  if (command.startsWith("POSE ")) {
    int v[AXIS_COUNT];

    int n = sscanf(
      command.c_str() + 5, "%d %d %d %d %d",
      &v[0], &v[1], &v[2], &v[3], &v[4]
    );

    if (n != AXIS_COUNT) {
      server.send(400, "text/plain", "Use POSE B S E A G");
      return;
    }

    for (int i = 0; i < AXIS_COUNT; i++) {
      if (v[i] < AXIS_MIN[i] || v[i] > AXIS_MAX[i]) {
        server.send(400, "text/plain", "Pose angle out of range");
        return;
      }
    }

    ensureArmed();
    cancelSequence();

    for (int i = 0; i < AXIS_COUNT; i++) {
      setTarget(i, v[i]);
    }

    server.send(200, "text/plain", "Pose started");
    return;
  }

  // ANGLE B 90
  if (command.startsWith("ANGLE ")) {
    int firstSpace = command.indexOf(' ');
    int secondSpace = command.indexOf(' ', firstSpace + 1);

    if (secondSpace < 0) {
      server.send(400, "text/plain", "Use ANGLE B 90");
      return;
    }

    String motorText =
      command.substring(firstSpace + 1, secondSpace);

    String angleText =
      command.substring(secondSpace + 1);

    motorText.trim();
    angleText.trim();

    if (motorText.length() != 1 || angleText.length() == 0) {
      server.send(400, "text/plain", "Invalid angle command");
      return;
    }

    for (unsigned int i = 0; i < angleText.length(); i++) {
      if (!isDigit(angleText.charAt(i))) {
        server.send(400, "text/plain", "Invalid angle");
        return;
      }
    }

    char motor = toupper(motorText.charAt(0));
    int angle = angleText.toInt();

    if (!setServoAngle(motor, angle)) {
      server.send(400, "text/plain", "Invalid motor or angle");
      return;
    }

    server.send(200, "text/plain", "Angle updated");
    return;
  }

  // SPD B 45
  if (command.startsWith("SPD ")) {
    int firstSpace = command.indexOf(' ');
    int secondSpace = command.indexOf(' ', firstSpace + 1);

    if (secondSpace < 0) {
      server.send(400, "text/plain", "Use SPD B 45");
      return;
    }

    String motorText =
      command.substring(firstSpace + 1, secondSpace);

    String speedText =
      command.substring(secondSpace + 1);

    motorText.trim();
    speedText.trim();

    if (motorText.length() != 1 || speedText.length() == 0) {
      server.send(400, "text/plain", "Invalid speed command");
      return;
    }

    for (unsigned int i = 0; i < speedText.length(); i++) {
      if (!isDigit(speedText.charAt(i))) {
        server.send(400, "text/plain", "Invalid speed");
        return;
      }
    }

    char motor = toupper(motorText.charAt(0));
    int delayMs = speedText.toInt();

    if (!setServoSpeed(motor, delayMs)) {
      server.send(400, "text/plain", "Speed must be 45-55");
      return;
    }

    server.send(200, "text/plain", "Speed updated");
    return;
  }

  server.send(400, "text/plain", "Unknown command");
}

// =====================================================
// STATUS ENDPOINT
// =====================================================

void handleStatus() {
  String json = "{";

  for (int i = 0; i < AXIS_COUNT; i++) {
    json += "\"";
    json += AXIS_CHAR[i];
    json += "\":";
    json += String((int)lroundf(axisPos[i]));
    json += ",";

    json += "\"s";
    json += AXIS_CHAR[i];
    json += "\":";
    json += String(speedDelay[i]);
    json += ",";
  }

  json += "\"busy\":";
  json += seqActive ? "1" : "0";
  json += ",\"armed\":";
  json += armed ? "1" : "0";

  json += "}";

  server.send(200, "application/json", json);
}

// =====================================================
// SETUP
// =====================================================

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5); // Sub-millisecond non-blocking serial timeout for instant STOP braking

  // Tracked state only - NO servo signal is sent here.
  for (int i = 0; i < AXIS_COUNT; i++) {
    axisPos[i] = AXIS_HOME[i];
    axisTarget[i] = AXIS_HOME[i];
    axisVel[i] = 0;
    speedDelay[i] = DEFAULT_SPEED_DELAY;
    lastPulseUs[i] = -1;
  }

  // DC motor driver held in standby until first interaction
  pinMode(DC_PWM, OUTPUT);
  pinMode(DC_AIN1, OUTPUT);
  pinMode(DC_AIN2, OUTPUT);
  pinMode(DC_STBY, OUTPUT);

  digitalWrite(DC_PWM, LOW);
  digitalWrite(DC_AIN1, LOW);
  digitalWrite(DC_AIN2, LOW);
  digitalWrite(DC_STBY, LOW);

  // Start standalone access point
  WiFi.mode(WIFI_AP);
  WiFi.softAPConfig(local_IP, gateway, subnet);

  // Empty password creates an open access point.
  bool apStarted;

  if (strlen(AP_PASSWORD) == 0) {
    apStarted = WiFi.softAP(AP_SSID);
  } else {
    apStarted = WiFi.softAP(AP_SSID, AP_PASSWORD);
  }

  Serial.println();
  Serial.println("===== ESP32 ROBOT ARM READY =====");

  if (apStarted) {
    Serial.println("Wi-Fi Access Point started");

    Serial.print("Wi-Fi Name: ");
    Serial.println(AP_SSID);

    if (strlen(AP_PASSWORD) == 0) {
      Serial.println("Wi-Fi Password: None (open network)");
    } else {
      Serial.println("Wi-Fi password configured");
    }

    Serial.print("IP Address: ");
    Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("ERROR: Failed to start Wi-Fi AP");
  }

  // Web server routes
  server.on("/", HTTP_GET, []() {
    server.send_P(200, "text/html", INDEX_HTML);
  });

  server.on("/cmd", HTTP_GET, handleWebCommand);
  server.on("/status", HTTP_GET, handleStatus);

  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });

  server.begin();

  Serial.println("Web server started");
  Serial.println("Open http://192.168.4.1");
  Serial.println("Servos stay idle until the first command.");
}

// =====================================================
// MAIN LOOP
// =====================================================

void loop() {
  // PRIORITY 1: Read Serial STOP & control commands immediately (sub-millisecond brake response)
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.length() > 0) {
      // STOP: Instantly halts and brakes all arm motion in place smoothly
      if (command.equalsIgnoreCase("STOP")) {
        cancelSequence();
        brakeAll();
        dcMotorStop();
        Serial.printf("STOP: All arm servos halted and braked in place | POS B:%d S:%d E:%d A:%d\n",
                      (int)round(axisPos[AX_B]), (int)round(axisPos[AX_S]),
                      (int)round(axisPos[AX_E]), (int)round(axisPos[AX_A]));
        return;
      }

      // Handle other serial commands immediately
      handleSerialCommand(command);
    }
  }

  server.handleClient();

  updateSequence();
  updateMotion();
}

void handleSerialCommand(String command) {

  // SPEED <ms>: Slower, smooth speed configuration from Python
  if (command.startsWith("SPEED")) {
    int spd = command.substring(5).toInt();
    if (spd < 20) spd = DEFAULT_SPEED_DELAY;
    for (int i = 0; i < AXIS_COUNT; i++) speedDelay[i] = spd;
    Serial.print("SPEED: Set all servos delay to ");
    Serial.print(spd);
    Serial.println(" ms/deg");
    return;
  }

  if (command.equalsIgnoreCase("H")) {
    goHome();
    return;
  }

  if (command == "1") {
    startSearchSequence();
    return;
  }

  if (command == "2") {
    runCommand2();
    return;
  }

  if (command == "3") {
    startCommand3Sequence();
    return;
  }

  // Serial Y = Pick & CW (Close gripper & spin roller)
  if (command.equalsIgnoreCase("Y")) {
    clockwiseAndClose();
    return;
  }

  // Serial CW = Spin DC Roller Clockwise (Pluck)
  if (command.equalsIgnoreCase("CW")) {
    dcMotorClockwise();
    return;
  }

  // Serial CCW = Spin DC Roller Counter-Clockwise (Eject)
  if (command.equalsIgnoreCase("CCW")) {
    dcMotorAnticlockwise();
    return;
  }

  // Serial STOP_DC = Stop DC Plucker roller
  if (command.equalsIgnoreCase("STOP_DC")) {
    dcMotorStop();
    return;
  }

  // Serial S = Drop & CCW
  if (command.equalsIgnoreCase("S")) {
    anticlockwiseAndOpen();
    return;
  }

  if (command.equalsIgnoreCase("O")) {
    ensureArmed();
    cancelSequence();
    setTarget(AX_G, GRIPPER_OPEN);
    return;
  }

  if (command.equalsIgnoreCase("C")) {
    ensureArmed();
    cancelSequence();
    setTarget(AX_G, GRIPPER_CLOSE);
    return;
  }

  // Individual Serial servo commands:
  // B90 = Base, S90 = Shoulder,
  // E90 = Elbow, A90 = Arm

  if (command.length() < 2) {
    Serial.println("Invalid command");
    return;
  }

  char motor = toupper(command.charAt(0));
  String angleText = command.substring(1);

  for (unsigned int i = 0; i < angleText.length(); i++) {
    if (!isDigit(angleText.charAt(i))) {
      Serial.println("Invalid angle");
      return;
    }
  }

  int angle = angleText.toInt();
  int axis = axisIndex(motor);

  if (axis < 0 || axis == AX_G) {
    Serial.println("Unknown motor");
    return;
  }

  ensureArmed();
  cancelSequence();
  setTarget(axis, constrain(angle, 0, 180));
}