import math
import time
import numpy as np
import cv2
from cvzone.PoseModule import PoseDetector
from flask import Flask, Response, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

direction = 0
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

    for x, y in points:
        cv2.circle(img, (x, y), 8, (255, 255, 0), 5)
        cv2.circle(img, (x, y), 6, (0, 255, 0), 5)

    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (0, 10, 255), 4)

    if len(points) >= 3:
        angle = calculate_angle(points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1])
        cv2.putText(img, f"{angle:.2f}", (int((points[0][0] + points[1][0] + points[2][0]) / 3),
                                          int((points[0][1] + points[1][1] + points[2][1]) / 3) + 30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        return angle
    else:
        return None

def process_pose(img, lmlist, *sets_of_indices):
    angles = []
    for indices in sets_of_indices:
        angle = draw_points_and_lines(img, lmlist, indices)
        if angle is not None:
            angles.append(angle)
    return angles

def angle_threshold(img, lmlist, *sets_of_indices, thresholds):
    angles = []
    for indices in sets_of_indices:
        angle = draw_points_and_lines(img, lmlist, indices)
        if angle is not None:
            angles.append(angle)

    return all(angle > threshold for angle, threshold in zip(angles, thresholds))

def initial_stance(lmlist):
    if len(lmlist) >= 16:
        wrist = lmlist[16][2]
        elbow = lmlist[14][2]
        return wrist < elbow
    else:
        return False

@app.route('/')
def index():
    return render_template('GroinKickLeft.html')

def gen_frames():
    cap = cv2.VideoCapture(0)
    pd = PoseDetector(trackCon=0.70, detectionCon=0.70)

    initial_stance_detected = False
    final_pose_detected = False
    cooldown_active = False
    cooldown_timer = 0
    lost_stance_timer = 0

    thresholds = [50, 50, 40, 10]

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, (1280, 720))
        pd.findPose(img, draw=True)
        lmlist, bbox = pd.findPosition(img, draw=False, bboxWithHands=False)

        if not initial_stance_detected:
            initial_stance_detected = initial_stance(lmlist)
            if not initial_stance_detected:
                if time.time() - lost_stance_timer > 5:
                    initial_stance_detected = False
                    final_pose_detected = False
                    lost_stance_timer = time.time()
        else:
            if not final_pose_detected:
                if initial_stance(lmlist):
                    if angle_threshold(img, lmlist, [11, 13, 15], [12, 14, 16], [13, 11, 23], [14, 12, 24], thresholds=thresholds):
                        final_pose_detected = True
                        cooldown_active = True
                        cooldown_timer = time.time() + cooldown_duration
            else:
                if time.time() > cooldown_timer:
                    cooldown_active = False

                if not cooldown_active:
                    if not initial_stance(lmlist):
                        initial_stance_detected = False
                        final_pose_detected = False
                        lost_stance_timer = time.time()
                    else:
                        angles = process_pose(img, lmlist, [12, 14, 16], [12, 24, 26], [26, 28, 32], [12, 11, 23], [11, 23, 25], [23, 25, 27], [25, 27, 31])
                        interpolated_angles = [int(np.interp(angle, src, dst)) for angle, src, dst in zip(angles,
                                                                                                              [[50, 90], [200, 160, 190],
                                                                                                               [115, 105], [260, 230], [145, 90], [200, 190], [93, 120]],
                                                                                                              [[0, 100], [0, 100, 0],
                                                                                                               [0, 100], [0, 100], [0,100], [0, 100], [0, 100]])]

                        angle_variable = interpolated_angles

                        avg_angle = sum(angle_variable) / len(angle_variable) if len(angle_variable) > 0 else 0

                        max_bar_height = 100
                        white_rect_height = max_bar_height * 2
                        white_rect_top = 400 - white_rect_height

                        border_thickness = 6
                        cv2.rectangle(img, (10 - border_thickness, white_rect_top - border_thickness),
                                      (90 + border_thickness, 400 + border_thickness), (255, 255, 255), -1)

                        bar_height = int(avg_angle * 2)
                        bar_top = 400 - bar_height

                        if bar_height > 1.7 * max_bar_height:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)

                        cv2.rectangle(img, (10, bar_top), (90, 400), color, -1)

                        if bar_height > 1.7 * max_bar_height:
                            text = "Correct Pose"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                            text_x = (img.shape[1] - text_size[0]) // 2
                            text_y = img.shape[0] - 50
                            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

                        print(avg_angle)

                        if avg_angle <= 10:
                            print("Angles when avg_angle <= 10:")
                            print(angles)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
