import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import pyautogui
import time

# Load trained Random Forest model
model = joblib.load('gesture_model.pkl')

# Initialize MediaPipe HandLandmarker Tasks API
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Action Debounce Setup (prevents rapid-fire trigger loops)
last_action_time = 0
COOLDOWN_SECONDS = 1.0  # Wait 1 second between executing actions

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        hand_landmarks = detection_result.hand_landmarks[0]
        
        # Draw keypoints
        h, w, _ = frame.shape
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Extract features (normalized relative to wrist)
        wrist = hand_landmarks[0]
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        # Predict gesture
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probs = model.predict_proba(features_array)[0]
        confidence = np.max(probs) * 100

        # Execute system action if confidence is high (> 85%) and cooldown has passed
        current_time = time.time()
        if confidence > 85.0 and (current_time - last_action_time) > COOLDOWN_SECONDS:
            if prediction == "thumbs_up":
                pyautogui.press("volumeup")
                print("Action: Volume Up")
                last_action_time = current_time
            elif prediction == "thumbs_down":
                pyautogui.press("volumedown")
                print("Action: Volume Down")
                last_action_time = current_time
            elif prediction == "peace":
                pyautogui.press("playpause")
                print("Action: Play/Pause")
                last_action_time = current_time

        # Display gesture label and confidence
        text = f"{prediction} ({confidence:.1f}%)"
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()