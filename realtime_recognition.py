import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib

# Load trained Scikit-Learn Model
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
        
        # Draw landmarks
        h, w, _ = frame.shape
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Extract features (normalized relative to Landmark 0)
        wrist = hand_landmarks[0]
        features = []
        for lm in hand_landmarks:
            features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        # Model Prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probs = model.predict_proba(features_array)[0]
        confidence = np.max(probs) * 100

        # Display output text on screen
        text = f"{prediction} ({confidence:.1f}%)"
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Custom Gesture Classifier", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:  # 27 is the ESC key
        break

cap.release()
cv2.destroyAllWindows()