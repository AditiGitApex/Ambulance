"""
detect_webcam.py
Live ambulance detection with stability improvements.

Changes from v1:
- Larger imgsz (640) for better small-object detection
- Lower confidence threshold (0.35)
- Detection smoothing - holds state for N frames to prevent flicker
- FPS display
"""

import cv2
import time
from ultralytics import YOLO
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parent.parent
weights_path = project_root / "models" / "ambulance_v5" / "weights" / "best.pt"

# ===== Tunable config =====
CONF_THRESHOLD = 0.35           # Lower = more sensitive (was 0.5)
IMGSZ = 640                     # Larger = better for distant objects (was 416)
DETECTION_HOLD_FRAMES = 15      # Hold detection state for N frames after last hit
# ==========================

CLASS_COLORS = {
    0: (50, 50, 200),     # ambulance -> red
    1: (50, 200, 50),     # non-ambulance -> green
}


def main():
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        return

    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam open nahi hua")
        return

    print(f"Settings: imgsz={IMGSZ}, conf={CONF_THRESHOLD}, hold={DETECTION_HOLD_FRAMES} frames")
    print("Press 'q' to quit.\n")

    # Smoothing state
    ambulance_hold = 0          # frames remaining to hold ambulance state
    frame_count = 0
    fps_start = time.time()
    fps_frame_count = 0
    current_fps = 0.0

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

        # Detection smoothing
        if ambulance_this_frame:
            ambulance_hold = DETECTION_HOLD_FRAMES
        elif ambulance_hold > 0:
            ambulance_hold -= 1

        ambulance_active = ambulance_hold > 0

        # Status banner
        if ambulance_active:
            if ambulance_this_frame:
                status = "AMBULANCE DETECTED"
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

        # FPS counter (update every 1 sec)
        if time.time() - fps_start > 1.0:
            current_fps = fps_frame_count / (time.time() - fps_start)
            fps_start = time.time()
            fps_frame_count = 0
        cv2.putText(frame, f"FPS: {current_fps:.1f}",
                    (frame.shape[1] - 110, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Ambulance Detection - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()