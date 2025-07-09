import cv2
import mediapipe as mp
from collections import deque
import winsound  # For beep sound
from datetime import datetime

# Voting parameters
VOTE_WIN = 10
VOTE_NEED = 3
gesture_history = deque(maxlen=VOTE_WIN)

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

middle_confirmed = False
beep_played = False  # To prevent repeated beeps

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as hands, \
    mp_face.FaceDetection(min_detection_confidence=0.5) as face_detection:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape

        # Hand detection
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark

            wrist_y = landmarks[0].y
            tips_y = [landmarks[i].y for i in [8, 12, 16, 20]]
            top_tip_y = min(tips_y)
            hand_height = wrist_y - top_tip_y

            finger_flags = []
            for tip_i, pip_i in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
                tip_rel = wrist_y - landmarks[tip_i].y
                norm_height = tip_rel / hand_height if hand_height else 0
                is_up = 1 if norm_height > 0.6 else 0
                finger_flags.append(is_up)

            is_middle = (finger_flags == [0, 0, 1, 0, 0])
            gesture_history.append(is_middle)
            middle_confirmed = sum(gesture_history) >= VOTE_NEED
        else:
            gesture_history.clear()
            middle_confirmed = False
            beep_played = False  # Reset beep when hand disappears

        # Face detection and blurring
        face_results = face_detection.process(frame_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * img_w)
                y1 = int(bbox.ymin * img_h)
                x2 = int((bbox.xmin + bbox.width) * img_w)
                y2 = int((bbox.ymin + bbox.height) * img_h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_w, x2), min(img_h, y2)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    face_blur = cv2.GaussianBlur(face_roi, (99, 99), 30)
                    frame[y1:y2, x1:x2] = face_blur

        # Middle finger detection: blur + sound
        if middle_confirmed:
            try:
                xs = [int(landmarks[i].x * img_w) for i in [9, 10, 11, 12]]
                ys = [int(landmarks[i].y * img_h) for i in [9, 10, 11, 12]]

                x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, img_w)
                y1, y2 = max(min(ys) - 20, 0), min(max(ys) + 20, img_h)

                pad = 5
                x1_p = max(x1 - pad, 0)
                x2_p = min(x2 + pad, img_w)
                y1_p = max(y1 - pad, 0)
                y2_p = min(y2 + pad, img_h)

                roi = frame[y1_p:y2_p, x1_p:x2_p]
                if roi.size > 0:
                    blur = cv2.GaussianBlur(roi, (99, 99), 0)
                    frame[y1_p:y2_p, x1_p:x2_p] = blur
                    cv2.putText(frame, "Middle Finger Detected", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    if not beep_played:
                        winsound.Beep(1000, 300)
                        beep_played = True

            except Exception as e:
                print(f"Skipping blur due to error: {e}")
        else:
            cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('webcam', frame)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
