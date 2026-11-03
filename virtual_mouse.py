import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import time

# Disable PyAutoGUI fail-safe pause for smooth real-time movement
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# Screen resolution
screen_w, screen_h = pyautogui.size()

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

# Smoothing parameters
prev_x, prev_y = 0, 0
smoothing = 5  # Increase for smoother, less jittery mouse movement

# Click cooldown parameters
last_click_time = 0
CLICK_COOLDOWN = 0.4

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Starting Virtual Mouse... Move your Index Finger. Pinch Index + Thumb to Click. Press 'q' to exit.")

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
        
        # Landmark 4: Thumb Tip | Landmark 8: Index Finger Tip
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]

        # Convert index tip landmark to pixel coordinates
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Draw fingertips
        cv2.circle(frame, (index_x, index_y), 8, (255, 0, 0), -1)
        cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 0, 255), -1)

        # Map webcam region to full screen coordinates
        target_x = np.interp(index_tip.x, [0.1, 0.9], [0, screen_w])
        target_y = np.interp(index_tip.y, [0.1, 0.9], [0, screen_h])

        # Apply exponential moving average for smooth cursor motion
        curr_x = prev_x + (target_x - prev_x) / smoothing
        curr_y = prev_y + (target_y - prev_y) / smoothing

        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Calculate Euclidean distance between index tip and thumb tip for Pinch Detection
        distance = np.sqrt((index_x - thumb_x)**2 + (index_y - thumb_y)**2)

        # Visual indicator when pinching
        cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 255, 0), 2)

        # Pinch threshold (in pixels) to trigger a left-click
        current_time = time.time()
        if distance < 35 and (current_time - last_click_time) > CLICK_COOLDOWN:
            pyautogui.click()
            cv2.circle(frame, (index_x, index_y), 12, (0, 255, 0), -1)
            cv2.putText(frame, "CLICK!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            last_click_time = current_time

    cv2.imshow("AI Virtual Mouse Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()