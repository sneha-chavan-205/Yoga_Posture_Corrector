import cv2
import mediapipe as mp
import pickle
import pandas as pd
import time
import pyttsx3

# Load trained model
model = pickle.load(open("pose_model.pkl", "rb"))

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize voice engine
engine = pyttsx3.init()

# Open webcam
cap = cv2.VideoCapture(0)

# Pose tracking variables
pose_start_time = None
current_pose = ""

while True:

    ret, frame = cap.read()

    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)

    pose_name = "Detecting..."
    confidence_text = ""
    hold_time = 0
    feedback = ""

    if results and results.pose_landmarks:

        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        landmarks = results.pose_landmarks.landmark

        row = []

        for lm in landmarks:
            row.extend([lm.x, lm.y, lm.z])

        X = pd.DataFrame([row], columns=model.feature_names_in_)

        pose_name = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities) * 100
        confidence_text = f"Confidence: {confidence:.2f}%"

        # Timer logic
        if pose_name == current_pose:
            if pose_start_time is None:
                pose_start_time = time.time()
        else:
            current_pose = pose_name
            pose_start_time = time.time()

        hold_time = int(time.time() - pose_start_time)

        # Feedback
        if hold_time < 5:
            feedback = "Hold the pose..."
        else:
            feedback = "Good posture!"

        # Voice feedback
        engine.say(feedback)
        engine.runAndWait()

        # Progress bar calculation
        progress = min(hold_time / 10, 1)

        bar_x = 50
        bar_y = 250
        bar_width = 300
        bar_height = 20

        # Draw empty bar
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (255,255,255),
                      2)

        # Fill progress bar
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + int(bar_width * progress), bar_y + bar_height),
                      (0,255,0),
                      -1)

    # Display text
    cv2.putText(frame,
                pose_name,
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.putText(frame,
                confidence_text,
                (50,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2)

    cv2.putText(frame,
                f"Hold Time: {hold_time}s",
                (50,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,200,255),
                2)

    cv2.putText(frame,
                feedback,
                (50,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,0,255),
                2)

    cv2.imshow("AI Yoga Trainer", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()