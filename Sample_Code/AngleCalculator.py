import cv2
import numpy as np
import PoseModule as pm
import math

# Initialize video capture and pose detector
cap = cv2.VideoCapture("Videos\RoundKickLeft.mp4")
detector = pm.PoseDetector()

# Function to calculate angle between three points
def calculate_angle(x1, y1, x2, y2, x3, y3):
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

# Function to calculate centroid of three points
def calculate_centroid(point1, point2, point3):
    x = (point1[0] + point2[0] + point3[0]) / 3
    y = (point1[1] + point2[1] + point3[1]) / 3
    return int(x), int(y)

# Process pose function
def process_pose(lmList, *landmark_sets):
    angles = []
    angle_text_list = []
    for landmark_set in landmark_sets:
        points = [(lmList[idx][1], lmList[idx][2]) for idx in landmark_set if 0 <= idx < len(lmList)]
        if len(points) == 3:
            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]
            angle = calculate_angle(x1, y1, x2, y2, x3, y3)
            angles.append(angle)
            angle_text_list.append(f"{round(angle)}")
            centroid = calculate_centroid(points[1], points[2], points[0])
            # Draw points and lines for each set of landmarks
            for idx in landmark_set:
                if 0 <= idx < len(lmList):
                    x, y = lmList[idx][1:3]
                    cv2.circle(img, (x, y), 8, (255, 0, 255), cv2.FILLED)
            for i in range(len(landmark_set) - 1):
                idx1, idx2 = landmark_set[i], landmark_set[i + 1]
                if 0 <= idx1 < len(lmList) and 0 <= idx2 < len(lmList):
                    x1, y1 = lmList[idx1][1:3]
                    x2, y2 = lmList[idx2][1:3]
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Draw angle text below centroid
            text_pos = (centroid[0], centroid[1] + 30)
            cv2.putText(img, angle_text_list[-1], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)  # Increased thickness to 2
    return angles, angle_text_list

# Initialize lists to store calculated angles
all_angles = []

# Main loop
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))
    #img = cv2.flip(img, 1)
    img = detector.findPose(img, True)
    lmList = detector.findPosition(img, True)

    # Calculate angles for specified landmark sets
    angles, angle_text_list = process_pose(lmList, [12, 14, 16], [12, 24, 26], [26, 28, 32], [12, 11, 23], [11, 23, 25], [23, 25, 27], [25, 27, 31])

    # Add angles to the list
    all_angles.append(angles)

    # Print all the calculated angles
    print("All calculated angles:")
    for idx, angles in enumerate(all_angles):
        print(f"Set {idx + 1}: {angles}")

    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
