import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

# Cursor smoothing
prev_x, prev_y = 0, 0
smooth_factor = 0.3

# Gesture state flags
single_click_triggered = False
double_click_triggered = False
click_threshold = 40

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark

            # Index finger tip
            x_index = int(lm_list[8].x * img.shape[1])
            y_index = int(lm_list[8].y * img.shape[0])

            # Middle finger tip
            x_middle = int(lm_list[12].x * img.shape[1])
            y_middle = int(lm_list[12].y * img.shape[0])

            # Thumb tip
            x_thumb = int(lm_list[4].x * img.shape[1])
            y_thumb = int(lm_list[4].y * img.shape[0])

            # Convert index finger to screen coordinates
            screen_x = np.interp(x_index, [0, img.shape[1]], [0, screen_w])
            screen_y = np.interp(y_index, [0, img.shape[0]], [0, screen_h])

            # Smooth cursor movement
            curr_x = prev_x + (screen_x - prev_x) * smooth_factor
            curr_y = prev_y + (screen_y - prev_y) * smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate distances
            dist_single = np.hypot(x_index - x_thumb, y_index - y_thumb)
            dist_double = np.hypot(x_middle - x_thumb, y_middle - y_thumb)

            # Show distances for debugging
            cv2.putText(img, f'SingleDist: {int(dist_single)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, f'DoubleDist: {int(dist_double)}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Single Click Gesture
            if dist_single < click_threshold and not single_click_triggered:
                pyautogui.click()
                single_click_triggered = True
                cv2.circle(img, (x_index, y_index), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, 'Single Click', (x_index - 40, y_index - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif dist_single > click_threshold + 10:
                single_click_triggered = False

            # Double Click Gesture
            if dist_double < click_threshold and not double_click_triggered:
                pyautogui.doubleClick()
                double_click_triggered = True
                cv2.circle(img, (x_middle, y_middle), 15, (0, 0, 255), cv2.FILLED)
                cv2.putText(img, 'Double Click', (x_middle - 40, y_middle - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif dist_double > click_threshold + 10:
                double_click_triggered = False

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

 