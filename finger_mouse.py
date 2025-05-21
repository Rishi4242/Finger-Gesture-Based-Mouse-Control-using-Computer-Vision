import cv2
import numpy as np
import mediapipe as mp
import pyautogui

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen resolution
screen_width, screen_height = pyautogui.size()

# Webcam capture and resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

# State variables
clicking = False
prev_x, prev_y = 0, 0
smoothening = 7  # Increase this for smoother movement (but with more delay)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)  # mirror view
    h, w, _ = frame.shape

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # If hand is detected
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if lm_list:
                # Index finger tip
                x1, y1 = lm_list[8][1], lm_list[8][2]

                # Convert coordinates to screen size
                target_x = np.interp(x1, (0, w), (0, screen_width))
                target_y = np.interp(y1, (0, h), (0, screen_height))

                # Smooth the movement
                curr_x = prev_x + (target_x - prev_x) / smoothening
                curr_y = prev_y + (target_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Thumb tip
                x2, y2 = lm_list[4][1], lm_list[4][2]

                # Distance between index and thumb
                distance = np.hypot(x2 - x1, y2 - y1)

                # Click if fingers are close together
                if distance < 40:
                    if not clicking:
                        clicking = True
                        pyautogui.click()
                        # print("Click!")
                        cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                else:
                    clicking = False

    cv2.imshow("Finger Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
