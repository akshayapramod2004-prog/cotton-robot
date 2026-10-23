import time
import serial
import serial.tools.list_ports

class SerialArmCommunicator:
    """
    Serial Communication Bridge for Robotic Arm / Manipulator:
    - Auto-detects COM ports (e.g. COM3, COM4, COM5)
    - Sends command 'S' when cotton is detected
    - Provides Simulation Mode if no hardware is connected
    """
    def __init__(self, port=None, baudrate=115200, cooldown_sec=2.0):
        self.baudrate = baudrate
        self.cooldown_sec = cooldown_sec
        self.last_sent_time = 0.0
        self.ser = None
        self.is_connected = False
        self.port = port

        self.connect(port)

    def find_available_ports(self):
        ports = list(serial.tools.list_ports.comports())
        return [p.device for p in ports]

    def connect(self, port=None):
        available_ports = self.find_available_ports()
        target_ports_to_try = []

        if port:
            target_ports_to_try.append(port)
        for p in available_ports:
            if p not in target_ports_to_try:
                target_ports_to_try.append(p)

        connected = False
        for p in target_ports_to_try:
            try:
                self.ser = serial.Serial(p, self.baudrate, timeout=0.1)
                try:
                    self.ser.dtr = False
                    self.ser.rts = False
                except Exception:
                    pass
                self.port = p
                self.is_connected = True
                connected = True
                # Allow ESP32 board to settle after USB port open
                time.sleep(1.2)
                # Flush startup bytes
                try:
                    if self.ser.in_waiting > 0:
                        self.ser.read(self.ser.in_waiting)
                except Exception:
                    pass
                print(f"[SERIAL OK] Connected to ESP32 Robotic Arm on Port: {self.port} at {self.baudrate} Baud (Ready)")
                break
            except Exception as e:
                err_msg = str(e)
                if "PermissionError" in err_msg or "Access is denied" in err_msg:
                    print(f"\n" + "!" * 75)
                    print(f" [ACTION REQUIRED] Port {p} is currently LOCKED by Arduino IDE!")
                    print(f" --> Please CLOSE the Serial Monitor tab in Arduino IDE (click 'X' tab).")
                    print(f" --> Once closed, Python will connect directly to {p}.")
                    print("!" * 75 + "\n")
                else:
                    print(f"[SERIAL WARN] Could not connect to port {p}: {e}")

        if not connected:
            print("[SERIAL MODE] Running in SIMULATED SERIAL MODE (Waiting for COM7 hardware to become free)")
            self.is_connected = False
        return self.is_connected

    def try_reconnect(self):
        """Actively attempts to claim real hardware serial port if previously locked."""
        if self.is_connected and self.ser and self.ser.is_open:
            return True
        return self.connect(self.port)

    def send_cotton_detected(self, target_id=1, cx=0, cy=0, z_mm=0.0, x_mm=0.0, y_mm=0.0):
        current_time = time.time()
        # Cooldown check to prevent flooding the microcontroller
        if (current_time - self.last_sent_time) < self.cooldown_sec:
            return False

        # In ESP32 firmware: 'Y' is Pick & CW (Closes gripper + runs DC roller to harvest cotton)
        simple_cmd = "Y\n"

        if z_mm > 0:
            z_cm = z_mm / 10.0
            dist_str = f"Distance: {z_cm:.1f} cm ({z_mm:.0f} mm) [3D: X:{x_mm}mm, Y:{y_mm}mm]"
        else:
            dist_str = f"Pixel Position: (X:{cx}, Y:{cy})"

        if self.is_connected and self.ser and self.ser.is_open:
            try:
                # Send 'Y' command over Serial to Arduino / Arm Controller
                self.ser.write(simple_cmd.encode('utf-8'))
                self.ser.flush()
                time.sleep(0.1)
                self.ser.write(b"CW\n")
                self.ser.flush()
                self.last_sent_time = current_time
                print(f"[SERIAL TRANSMIT] Sent 'Y' and 'CW' commands: Cotton Detected! Actuating Gripper & Plucker! {dist_str} | Target #{target_id}")
                return True
            except Exception as e:
                print(f"[SERIAL ERROR] Transmission failed: {e}")
                self.is_connected = False
                return False
        else:
            # Simulated Serial Transmission Output
            self.last_sent_time = current_time
            print(f"[SIMULATED SERIAL TRANSMIT] Sent 'Y' and 'CW' commands: Cotton Detected! Actuating Gripper & Plucker! {dist_str} | Lock Target #{target_id}")
            return True

    def send_servo(self, axis_char, angle):
        """Sends single servo angle command, e.g. B90, S140, E140, A140"""
        cmd_str = f"{axis_char.upper()}{int(angle)}\n"
        if self.is_connected and self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd_str.encode('utf-8'))
                self.ser.flush()
                return True
            except Exception as e:
                print(f"[SERIAL ERROR] Failed to send angle {cmd_str.strip()}: {e}")
                return False
        return True

    def send_multi_servo(self, b=None, s=None, a=None, e=None):
        """Sends servo angle commands sequentially to all specified axes."""
        if b is not None:
            self.send_servo('B', b)
            time.sleep(0.015)
        if s is not None:
            self.send_servo('S', s)
            time.sleep(0.015)
        if a is not None:
            self.send_servo('A', a)
            time.sleep(0.015)
        if e is not None:
            self.send_servo('E', e)
            time.sleep(0.015)
        return True

    def send_cmd(self, cmd_char):
        """Sends arbitrary command character (e.g. 'Y' for gripper, 'H' for home, '2' for cycle) over serial"""
        cmd_str = f"{cmd_char.strip()}\n"
        if self.is_connected and self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd_str.encode('utf-8'))
                self.ser.flush()
                print(f"[SERIAL TRANSMIT] Sent command '{cmd_char.upper()}' to ESP32 Arm Controller!")
                return True
            except Exception as e:
                print(f"[SERIAL ERROR] Transmission failed: {e}")
                return False
        else:
            print(f"[SIMULATED SERIAL TRANSMIT] Sent command '{cmd_char.upper()}' to ESP32 Arm Controller!")
            return True

    def read_response(self):
        """Read feedback messages sent back from Arduino / Microcontroller IDE"""
        if self.is_connected and self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if response:
                        # Suppress spurious "Invalid angle" noise from older firmware
                        if "Invalid angle" in response:
                            return None
                        print(f"[SERIAL MONITOR READ] {response}")
                        return response
            except Exception:
                pass
        return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SERIAL] Port closed cleanly.")

