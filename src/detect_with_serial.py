"""
detect_with_serial.py
Full integrated system: webcam detection + ESP32 traffic light.

Pipeline:
  Camera detects ambulance -> Python sends 'A' to ESP32
  ESP32 receives -> Green LED ON
  ESP32 independently detects siren via W104 -> Yellow LED
  Nothing -> Red LED (default)
"""

import cv2
import time
import serial
from ultralytics import YOLO
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parent.parent
weights_path = project_root / "models" / "ambulance_v5" / "weights" / "best.pt"

# ===== Detection config =====
CONF_THRESHOLD = 0.35
IMGSZ = 640
DETECTION_HOLD_FRAMES = 15

# ===== Serial config =====
SERIAL_PORT = 'COM7'              # Change if needed
SERIAL_BAUD = 115200
SEND_INTERVAL = 0.4               # Min seconds between 'A' sends
# =============================

CLASS_COLORS = {
    0: (50, 50, 200),     # ambulance -> red box
    1: (50, 200, 50),     # non-ambulance -> green box
}


def open_serial(port, baud):
    """Open ESP32 serial. Returns Serial object or None on failure."""
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)        # ESP32 resets on connect - wait for boot
        ser.reset_input_buffer()
        print(f"[OK] Serial connected: {port} @ {baud}")
        return ser
    except serial.SerialException as e:
        print(f"[WARN] Cannot open {port}: {e}")
        print("[WARN] Running in camera-only mode (no ESP32 commands)")
        return None


def main():
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        return

    esp = open_serial(SERIAL_PORT, SERIAL_BAUD)

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: webcam not opened")
        if esp:
            esp.close()
        return

    print(f"\nSettings: imgsz={IMGSZ}, conf={CONF_THRESHOLD}, hold={DETECTION_HOLD_FRAMES}")
    print("Press 'q' in webcam window to quit.\n")

    # State
    ambulance_hold = 0
    last_send = 0
    frame_count = 0
    fps_start = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            fps_frame_count += 1

            # Inference
            results = model.predict(
                frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False
            )

            ambulance_this_frame = False
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_name = r.names[cls_id]
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))

                    if cls_id == 0:
                        ambulance_this_frame = True

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8),
                                (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

            # Smoothing
            if ambulance_this_frame:
                ambulance_hold = DETECTION_HOLD_FRAMES
            elif ambulance_hold > 0:
                ambulance_hold -= 1

            ambulance_active = ambulance_hold > 0

            # Send to ESP32 (rate-limited)
            now = time.time()
            if esp and ambulance_active and (now - last_send) >= SEND_INTERVAL:
                try:
                    esp.write(b'A')
                    last_send = now
                    print(f"[Frame {frame_count}] -> ESP32: A")
                except Exception as e:
                    print(f"[ERROR] Serial write failed: {e}")

            # Read ESP32 output
            if esp and esp.in_waiting:
                try:
                    line = esp.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        print(f"[ESP32] {line}")
                except Exception:
                    pass

            # Status banner
            if ambulance_active:
                if ambulance_this_frame:
                    status = "AMBULANCE -> ESP32"
                    status_color = (50, 50, 220)
                else:
                    status = f"AMBULANCE (holding {ambulance_hold})"
                    status_color = (50, 100, 220)
            else:
                status = "MONITORING"
                status_color = (180, 180, 180)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (30, 30, 30), -1)
            cv2.putText(frame, status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # FPS + ESP32 indicator
            if time.time() - fps_start > 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start)
                fps_start = time.time()
                fps_frame_count = 0

            esp_label = "ESP32: ON" if esp else "ESP32: OFF"
            esp_color = (50, 200, 50) if esp else (50, 50, 200)
            cv2.putText(frame, esp_label,
                        (frame.shape[1] - 200, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, esp_color, 1)
            cv2.putText(frame, f"FPS: {current_fps:.1f}",
                        (frame.shape[1] - 90, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Ambulance Detection + ESP32", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if esp:
            esp.close()
            print("\n[OK] Serial closed")
        print(f"Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()