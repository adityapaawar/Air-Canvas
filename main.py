import cv2
import numpy as np
import mediapipe as mp
import time

# ----------------- Setup -----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0
brush_size = 6
color = (255, 0, 0)  # Default Blue

# Define color buttons
colors = {
    "Blue": ((255,0,0), (10,10,110,60)),
    "Green": ((0,255,0), (120,10,220,60)),
    "Red": ((0,0,255), (230,10,330,60)),
    "Eraser": ((0,0,0), (340,10,480,60))
}

# ----------------- Functions -----------------
def draw_ui(frame):
    for name, (col, (x1,y1,x2,y2)) in colors.items():
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, -1)
        cv2.putText(frame, name, (x1+10,y1+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# ----------------- Main Loop -----------------
with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        draw_ui(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            x, y = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)

            # Check if finger is on top buttons
            if y < 60:
                for name, (col, (x1,y1,x2,y2)) in colors.items():
                    if x1 < x < x2:
                        color = col
            else:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), color, brush_size)
                prev_x, prev_y = x, y
        else:
            prev_x, prev_y = 0, 0

        # Combine canvas & frame
        frame = cv2.addWeighted(frame, 0.7, canvas, 1, 0)
        cv2.imshow("Air Canvas", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Clear canvas
            canvas = np.zeros_like(frame)
        elif key == ord('s'):  # Save drawing
            filename = f"AirCanvas_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Saved {filename}")
        elif key == ord('q'):  # Quit
            break

cap.release()
cv2.destroyAllWindows()
