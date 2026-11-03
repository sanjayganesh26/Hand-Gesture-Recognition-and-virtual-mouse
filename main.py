import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import pyautogui
import time

# Disable PyAutoGUI delay for fast mouse response
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# Load trained Random Forest model
try:
    model = joblib.load('gesture_model.pkl')
    has_model = True
except FileNotFoundError:
    has_model = False
    print("Warning: gesture_model.pkl not found! ML Gesture Mode will be disabled until trained.")

# Initialize MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# State Management
mode = "GESTURE"  # Options: "GESTURE" or "MOUSE"
prev_x, prev_y = 0, 0
smoothing = 5
last_action_time = 0
COOLDOWN = 0.8

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Unified Control Running!")
print("Press 'm' to toggle between [GESTURE CONTROL] and [VIRTUAL MOUSE] mode.")
print("Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        wrist = hand_landmarks[0]

        # Draw hand keypoints
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        current_time = time.time()

        # MODE 1: GESTURE CONTROL (Media / Volume Automation)
        if mode == "GESTURE" and has_model:
            features = []
            for lm in hand_landmarks:
                features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)[0]
            confidence = np.max(model.predict_proba(features_array)[0]) * 100

            if confidence > 80.0 and (current_time - last_action_time) > COOLDOWN:
                if prediction == "thumbs_up":
                    pyautogui.press("volumeup")
                    last_action_time = current_time
                elif prediction == "thumbs_down":
                    pyautogui.press("volumedown")
                    last_action_time = current_time
                elif prediction == "peace":
                    pyautogui.press("playpause")
                    last_action_time = current_time

            cv2.putText(frame, f"Gesture: {prediction} ({confidence:.0f}%)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # MODE 2: VIRTUAL MOUSE (Pointer Movement & Pinch Click)
        elif mode == "MOUSE":
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]

            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Map coordinates to screen size
            target_x = np.interp(index_tip.x, [0.1, 0.9], [0, screen_w])
            target_y = np.interp(index_tip.y, [0.1, 0.9], [0, screen_h])

            # Apply smoothing
            curr_x = prev_x + (target_x - prev_x) / smoothing
            curr_y = prev_y + (target_y - prev_y) / smoothing
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Pinch detection (click distance threshold)
            dist = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (255, 0, 0), 2)

            if dist < 35 and (current_time - last_action_time) > 0.4:
                pyautogui.click()
                cv2.putText(frame, "CLICK!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                last_action_time = current_time

    # Display Current Active Mode Status
    cv2.putText(frame, f"MODE: {mode} (Press 'M' to toggle)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition & Control Center", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "MOUSE" if mode == "GESTURE" else "GESTURE"
        print(f"Switched mode to: {mode}")
    elif key == ord('q') or key == ord('Q') or key == 27:  # 27 is the ESC key
        break

cap.release()
cv2.destroyAllWindows()