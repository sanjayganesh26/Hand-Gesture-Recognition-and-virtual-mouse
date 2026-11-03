# Real-Time Hand Gesture Recognition & AI Virtual Mouse ✋🖱️

An end-to-end Computer Vision and Machine Learning application that performs real-time hand landmark tracking, custom gesture classification using Scikit-Learn, system volume/media automation, and virtual mouse pointer control using MediaPipe Tasks API and OpenCV.

---

## 🌟 Key Features

- **Real-Time Hand Landmark Tracking:** Tracks 21 3D hand joint coordinates at high FPS using the modern **MediaPipe Tasks API**.
- **Custom Gesture Classifier (Machine Learning):** Built-in dataset collection and Random Forest model training pipeline for custom hand gestures.
- **AI Virtual Mouse:** Control your system's cursor using index finger coordinates with exponential moving average smoothing and pinch-to-click detection.
- **System Automation:** Control system volume and media playback via recognized hand gestures using `PyAutoGUI`.
- **Dynamic Mode Switching:** Seamlessly toggle between **Gesture Automation Mode** and **Virtual Mouse Mode** during execution with a single keypress.

---

## 📁 Project Structure

```text
Hand-Gesture-Recognition/
├── hand_landmarker.task      # Downloaded MediaPipe vision task asset model
├── gesture_data.csv          # Recorded 63-coordinate feature dataset
├── gesture_model.pkl         # Trained Random Forest classifier weights
│
├── collect_data.py           # Dataset collection tool for hand coordinates
├── train_model.py            # Model training & accuracy evaluation script
├── realtime_recognition.py   # Real-time gesture classification testing
├── gesture_actions.py        # Gesture-to-system automation mapping script
├── virtual_mouse.py          # Standalone AI virtual mouse script
└── main.py                   # Unified main application with live mode switching