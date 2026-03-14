import cv2
import mediapipe as mp
import os
import csv

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

dataset_path = "dataset"

data = []

for pose_name in os.listdir(dataset_path):

    pose_folder = os.path.join(dataset_path, pose_name)

    for image_file in os.listdir(pose_folder):

        image_path = os.path.join(pose_folder, image_file)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image_rgb)

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            row = []

            for lm in landmarks:
                row.append(lm.x)
                row.append(lm.y)
                row.append(lm.z)

            row.append(pose_name)

            data.append(row)

with open("pose_dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

print("Dataset created successfully!")