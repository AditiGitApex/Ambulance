# 🚑 AI-Powered Ambulance Detection System with IoT Integration

Real-time ambulance detection system combining **computer vision** (YOLOv8), **audio frequency analysis** (FFT on ESP32), and **IoT** (traffic light LEDs). Detects ambulances from a camera feed or sirens from a microphone, then controls a 3-LED traffic light via serial communication.

![Status](https://img.shields.io/badge/status-working-brightgreen) ![Python](https://img.shields.io/badge/python-3.12-blue) ![YOLOv8](https://img.shields.io/badge/model-YOLOv8n-orange)

---

## ✨ Features

- 🎥 Real-time ambulance detection via YOLOv8n (80% mAP50)
- 🔊 Frequency-selective siren detection using FFT (500–1800 Hz band)
- 💡 Priority LED logic: 🟢 Green (ambulance) > 🟡 Yellow (siren) > 🔴 Red (idle)
- 🔌 Python ↔ ESP32 serial communication at 115200 baud
- ⚡ Runs on CPU at 4–10 FPS with <200 ms end-to-end latency

---

## 🛠️ Tech Stack

**Software:** Python 3.12, OpenCV, Ultralytics YOLOv8, PySerial, Arduino C++, arduinoFFT
**Hardware:** ESP32 Dev Module, W104 sound sensor, 3× LEDs, 220 Ω resistors
**Dataset:** Roboflow (custom 2-class dataset: ambulance / non-ambulance)

---

## 📈 Model Performance

| Class | Precision | Recall | mAP50 |
|-------|-----------|--------|-------|
| ambulance | 0.728 | 0.769 | 0.797 |
| non-ambulance | 0.824 | 0.739 | 0.802 |
| **Overall** | **0.776** | **0.754** | **0.800** |

Trained for 40 epochs on ~1,800 annotations with balanced class distribution.

---

## 🔌 Wiring

| Component | ESP32 Pin |
|-----------|-----------|
| W104 VCC / GND / AO | 3.3V / GND / GPIO 34 |
| Red LED (via 220 Ω) | GPIO 18 |
| Yellow LED (via 220 Ω) | GPIO 19 |
| Green LED (via 220 Ω) | GPIO 21 |

> Use the W104's **AO pin**, not DO — FFT needs the raw analog signal.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Upload firmware to ESP32
# Open hardware/ambulance_detection.ino in Arduino IDE and upload

# 3. Run the system
python src/detect_with_serial.py
```

> ⚠️ Close Arduino Serial Monitor before running the Python script (only one program can hold the COM port).

---

## 📚 Training Journey

Five iterations to find the right balance:

- **v1** — Single class, 97.9% mAP but false positives on every vehicle
- **v2/v3** — Failed due to mislabeled datasets
- **v4** — Clean single class, 81.2% mAP
- **v5** (final) — Two-class balanced dataset, 80.0% mAP with real-world usability

**Key lesson:** Data quality and class balance > raw accuracy numbers.

---

## 🚀 Future Improvements

- Real dashcam-footage dataset for better generalization
- CNN-based siren classifier (mel-spectrogram) to distinguish siren vs. horn
- GPS integration for intersection-level deployment
- INT8 quantization for 2–3× CPU speedup

---

⭐ If you found this useful, please star the repo!

