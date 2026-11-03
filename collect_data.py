import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import numpy as np

# 1. Initialize MediaPipe Tasks HandLandmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Gesture recording parameters
GESTURE_NAME = "4"  # Change this label for each gesture you record
TARGET_COUNT = 300
count = 0
recording = False

# Open camera using Windows DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam on index 0. Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print(f"Recording Target: '{GESTURE_NAME}' | Open the window and press 's' to start, 'q' to quit.")

with open('gesture_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        # Draw and record landmarks
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Draw key points for feedback
            h, w, _ = frame.shape
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Record features when recording is toggled ON
            if recording and count < TARGET_COUNT:
                wrist = hand_landmarks[0]
                row = []
                for lm in hand_landmarks:
                    row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                row.append(GESTURE_NAME)
                writer.writerow(row)
                count += 1

        # Display on-screen instructions
        status_text = f"Gesture: {GESTURE_NAME} | Recorded: {count}/{TARGET_COUNT}" if recording else "Press 'S' to Start Recording"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Render the OpenCV Window
        cv2.imshow("Data Collector - Hand Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            recording = True
        elif key == ord('q') or count >= TARGET_COUNT:
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Recorded {count} samples for '{GESTURE_NAME}'.")