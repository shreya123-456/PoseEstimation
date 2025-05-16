import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import csv
import math
import os
from datetime import datetime

# Load MoveNet Thunder model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

# Define connections between keypoints for drawing skeleton
KEYPOINT_EDGE_CONNECTIONS = {
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
}

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Calculate angle between three points
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360 - angle

# Process a single frame/image
def process_image(img, save_csv=False, output_name="output.jpg"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = tf.image.resize_with_pad(tf.expand_dims(tf.convert_to_tensor(img_rgb), axis=0), 256, 256)
    input_tensor = tf.cast(input_tensor, dtype=tf.int32)

    results = movenet(input_tensor)
    keypoints = results['output_0'].numpy()[0, 0, :, :]

    h, w, _ = img.shape
    points = []

    for idx, (y, x, conf) in enumerate(keypoints):
        if conf > 0.3:
            cx, cy = int(x * w), int(y * h)
            points.append((cx, cy))
            cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(img, str(idx), (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            points.append(None)

    # Draw skeleton
    for edge in KEYPOINT_EDGE_CONNECTIONS:
        pt1, pt2 = edge
        if points[pt1] and points[pt2]:
            cv2.line(img, points[pt1], points[pt2], (255, 0, 0), 2)

    # Calculate angles (example: left elbow, right elbow)
    angle_text = ""
    if points[5] and points[7] and points[9]:
        left_elbow_angle = calculate_angle(points[5], points[7], points[9])
        angle_text += f"Left Elbow: {int(left_elbow_angle)}°  "
    if points[6] and points[8] and points[10]:
        right_elbow_angle = calculate_angle(points[6], points[8], points[10])
        angle_text += f"Right Elbow: {int(right_elbow_angle)}°"

    cv2.putText(img, angle_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Pose classification (basic)
    if points[9] and points[10] and points[0]:
        if points[9][1] < points[0][1] and points[10][1] < points[0][1]:
            cv2.putText(img, "Pose: Hands Raised", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif points[13] and points[15] and abs(points[13][1] - points[15][1]) < 40:
            cv2.putText(img, "Pose: Squat", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save keypoints to CSV
    if save_csv:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"keypoints_{now}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "name", "x", "y", "confidence"])
            for i, kp in enumerate(keypoints):
                writer.writerow([i, KEYPOINT_NAMES[i], kp[1], kp[0], kp[2]])

    cv2.imwrite(output_name, img)
    return img

# -------- Main Execution --------
def main():
    use_webcam = False  # Change to True for webcam mode
    image_path = "pose_sample.jpg"
    output_image = "pose_output.jpg"

    if use_webcam:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_frame = process_image(frame.copy(), save_csv=False)
            cv2.imshow("Real-time Pose", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(image_path)
        result_img = process_image(image, save_csv=True, output_name=output_image)
        cv2.imshow("Pose Estimation", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Pose Estimation completed. Output saved to {output_image}")

if __name__ == "__main__":
    main()

