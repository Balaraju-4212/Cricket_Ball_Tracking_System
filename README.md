# 🏏 Cricket Ball AI Tracking System

A simple, fast, and highly effective **Computer Vision** and **Motion Tracking** system for detecting and tracking cricket balls in video footage, estimating ball velocity, and predicting trajectory bounce points.

---

## 📌 Key Features

1. **Dual Detection Engine (`tracker.py`)**:
   - OpenCV HSV Color & Contour Masking (optimized for Red & White cricket balls with circularity and morphological filters).
   - Ultralytics YOLO model fallback for deep learning object detection.

2. **Kalman Filter Motion Tracking**:
   - 2D state vector Kalman Filter ($x, y, v_x, v_y$) to smooth trajectory paths and predict ball positions during fast motion blur or temporary occlusion.

3. **Real-time Telemetry & HUD Overlay**:
   - Calculates delivery velocity in **km/h** using pitch pixel length calibration.
   - Detects pitch bounce location ($x, y$ coordinates).
   - Renders interactive trajectory line trails and motion HUD on video frames.

4. **Dual Execution Options**:
   - **Command Line CLI (`main.py`)**: Process videos directly from terminal with full analysis reports.
   - **Streamlit Web UI (`app.py`)**: Clean web dashboard to upload videos, adjust threshold parameters, and view tracking outputs.

---

## 📂 Simplified Project Architecture

```
Cricket_Ball_Tracking_AI/
├── tracker.py        # Core Ball Tracker (HSV + YOLO, Kalman Filter, Trajectory, Speed & HUD)
├── app.py            # Streamlit Web Dashboard UI
├── main.py           # CLI Command Line Interface
├── test_tracker.py   # Unit Test Suite
├── requirements.txt  # Project Dependencies
├── README.md         # Documentation
├── data/
│   └── raw/          # Input cricket videos (contains sample clip)
└── outputs/          # Output tracked videos
```

---

## 🚀 Quick Start Instructions

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run via Command Line Interface (CLI)
Track ball movement on any video file:
```bash
python main.py --input data/raw/cricket_video.mp4 --output outputs/tracked_output.mp4
```

### 3. Launch Web Application Dashboard
Run the Streamlit web dashboard:
```bash
streamlit run app.py
```

### 4. Run Automated Unit Tests
```bash
python -m unittest test_tracker.py
```

---

## 📜 License
MIT License - Open Source for Research and AI Applications.
# Cricket_Ball_Tracking_System
