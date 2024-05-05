import math
import time
import numpy as np
import cv2
from cvzone.PoseModule import PoseDetector

direction = 0

# Define cooldown duration in seconds
cooldown_duration = 5  # Adjust this value as needed

def calculate_angle(x1, y1, x2, y2, x3, y3):
    global direction

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle


def draw_points_and_lines(img, lmlist, indices):
    points = [(lmlist[idx][1], lmlist[idx][2]) for idx in indices if 0 <= idx < len(lmlist)]

    if len(points) < 3:
        return None

    # Draw circles for points
    for x, y in points:
        cv2.circle(img, (x, y), 8, (255, 255, 0), 5)
        cv2.circle(img, (x, y), 6, (0, 255, 0), 5)

    # Draw lines between points
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (0, 10, 255), 4)

    # Calculate angle if there are enough points
    if len(points) >= 3:
        angle = calculate_angle(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1])

        # Display angle below the line
        cv2.putText(img, f"{angle:.2f}", (int((points[0][0] + points[1][0] + points[2][0]) / 3),
                                          int((points[0][1] + points[1][1] + points[2][1]) / 3) + 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return angle
    else:
        return None


def process_pose(lmlist, *sets_of_indices):
    angles = []
    for indices in sets_of_indices:
        angle = draw_points_and_lines(img, lmlist, indices)
        if angle is not None:
            angles.append(angle)
    return angles


def angle_threshold(lmlist, *sets_of_indices, thresholds):
    angles = []
    for indices in sets_of_indices:
        angle = draw_points_and_lines(img, lmlist, indices)
        if angle is not None:
            angles.append(angle)

    return all(angle > threshold for angle, threshold in zip(angles, thresholds))


# Define a new function to detect the initial stance
# Define a new function to detect the initial stance
def initial_stance(lmlist):
    if len(lmlist) >= 16:  # Check if lmlist contains at least 16 landmarks
        wrist = lmlist[20][2]  # Y-coordinate of the wrist landmark
        elbow = lmlist[14][2]  # Y-coordinate of the elbow landmark
        return wrist < elbow
    else:
        return False

#cap = cv2.VideoCapture('Sample.mp4')
cap = cv2.VideoCapture('Videos\ChopRight.mp4')
pTime = 0
pd = PoseDetector(trackCon=0.70, detectionCon=0.70)

# Define thresholds for each set of landmarks
thresholds = [50, 50, 40, 10]

# Initialize flags
initial_stance_detected = False
final_pose_detected = False
cooldown_active = False
lost_stance_timer = time.time()  # Initialize a timer to track how long the initial stance is lost

while True:
    ret, img = cap.read()

    # Flip the frame horizontally
    img = cv2.flip(img, 1)

    img = cv2.resize(img, (1280, 720))

    # Detect pose
    pd.findPose(img, draw=True)
    lmlist, bbox = pd.findPosition(img, draw=False, bboxWithHands=False)

    # Ensure each landmark list has the landmark number from 0 to 31
    for i, landmarks in enumerate(lmlist):
        if landmarks:
            lmlist[i] = [i] + landmarks

    # Check if the user is in the initial stance
    if not initial_stance_detected:
        initial_stance_detected = initial_stance(lmlist)
        if not initial_stance_detected:
            # If the initial stance is not detected, check if it's been lost for too long
            if time.time() - lost_stance_timer > 5:  # Adjust the time threshold as needed
                # Reset initial_stance_detected and final_pose_detected flags
                initial_stance_detected = False
                final_pose_detected = False
                # Update the timer
                lost_stance_timer = time.time()
    else:
        # Once the initial stance is detected, proceed to detect the final pose
        if not final_pose_detected:
            if initial_stance(lmlist):  # Require the user to follow the initial stance before detecting the final pose
                if angle_threshold(lmlist, [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], thresholds=thresholds):
                    final_pose_detected = True
                    cooldown_active = True
                    cooldown_timer = time.time() + cooldown_duration
        else:
            # Check if cooldown period is over
            if time.time() > cooldown_timer:
                cooldown_active = False

            # Process pose only when cooldown is not active
            if not cooldown_active:
                # Check if the user's arms are lowered for an extended period
                if not initial_stance(lmlist):
                    # If the initial stance is lost, reset the pose detection
                    initial_stance_detected = False
                    final_pose_detected = False
                    lost_stance_timer = time.time()  # Reset the timer

                else:
                    # Example usage of process_pose
                    angles = process_pose(lmlist, [23, 25, 27], [11, 23, 25], [11, 13, 15], [12, 24, 26], [25, 27, 31], [12, 14, 16], [26, 28, 32])

                    # Interpolate and assign to a single variable
                    interpolated_angles = [int(np.interp(angle, src, dst)) for angle, src, dst in zip(angles,
                                                                                                          [[180, 140], [173, 174],
                                                                                                           [350, 280], [215, 216], [217, 230], [345, 280], [160, 220]],
                                                                                                          [[0, 100], [0, 100],
                                                                                                           [0, 100], [0, 100], [0,100], [0, 100], [0, 100]])]

                    angle_variable = interpolated_angles

                    # Calculate the average angle
                    avg_angle = sum(angle_variable) / len(angle_variable) if len(angle_variable) > 0 else 0

                    # Calculate the height of the white rectangle
                    max_bar_height = 100  # Maximum possible height of the bar
                    white_rect_height = max_bar_height * 2

                    # Calculate the top position of the white rectangle
                    white_rect_top = 400 - white_rect_height

                    # Draw a static white rectangle with increased thickness
                    border_thickness = 6  # Increased thickness
                    cv2.rectangle(img, (10 - border_thickness, white_rect_top - border_thickness),
                                  (90 + border_thickness, 400 + border_thickness), (255, 255, 255), -1)

                    # Calculate the height of the green bar
                    bar_height = int(avg_angle * 2)

                    # Calculate the top position of the bar
                    bar_top = 400 - bar_height

                    # Change color to red if bar height is more than 70%
                    if bar_height > 1.7 * max_bar_height:
                        color = (0, 0, 255)  # Red
                    else:
                        color = (0, 255, 0)  # Green

                    # Draw a single rectangle representing the average angle, making it longer
                    cv2.rectangle(img, (10, bar_top), (90, 400), color, -1)

                    # Add text if bar height surpasses 170%
                    if bar_height > 1.7 * max_bar_height:
                        text = "Correct Pose"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                        text_x = (img.shape[1] - text_size[0]) // 2
                        text_y = img.shape[0] - 50  # Bottom center
                        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                    print(avg_angle)

                    if avg_angle <= 10:
                        print("Angles when avg_angle <= 10:")
                        print(angles)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('frame', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

