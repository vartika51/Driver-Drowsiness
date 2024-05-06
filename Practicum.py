import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame  # Import pygame for playing alarm sound

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for heart rate calculation
prev_mean_value = 0
prev_sec = cv2.getTickCount()  # Initialize to current time to avoid initial false heart rate spike
heart_rates = []

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/vartika/Desktop/shape_predictor_68_face_landmarks.dat")

# Status marking for current state
status_counts = {'sleep': 0, 'drowsy': 0, 'active': 0}
status = ""
color = (0, 0, 0)

# Initialize pygame for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("/Users/vartika/Downloads/alarm.wav")  # Path to alarm sound file
alarm_playing = False

def compute_distance(ptA, ptB):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(ptA - ptB)

def blinked(landmarks, points):
    """Determine blink status based on eye landmark ratios."""
    vertical = compute_distance(landmarks[points[1]], landmarks[points[3]]) + compute_distance(landmarks[points[2]], landmarks[points[4]])
    horizontal = compute_distance(landmarks[points[0]], landmarks[points[5]])
    ratio = vertical / (2.0 * horizontal)
    if ratio > 0.25:
        return 2  # Fully open
    elif ratio > 0.19:
        return 1  # Partially open
    else:
        return 0  # Closed


def classify_heart_rate(heart_rate):
    """Classify heart rate status."""
    if heart_rate < 60:
        return "Drowsy"
    elif heart_rate > 100:
        return "Stress"
    else:
        return "Active :)"


def process_frame(frame, detector, predictor, status_counts, prev_mean_value, prev_sec, heart_rates):
    """Process each frame for face detection, blink detection, and heart rate calculation."""
    global status, color, alarm_playing  # Declare status, color, and alarm_playing as global variables
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = face_utils.shape_to_np(predictor(gray, face))

        left_blink = blinked(landmarks, [36, 37, 38, 41, 40, 39])
        right_blink = blinked(landmarks, [42, 43, 44, 47, 46, 45])

        # Update status counts based on blink detection
        if left_blink == 0 and right_blink == 0:
            status_counts['sleep'] += 1
            status_counts['drowsy'] = status_counts['active'] = 0
            if status_counts['sleep'] > 6:
                status, color = "SLEEPING !!!", (255, 0, 0)
                if not alarm_playing:
                    alarm_sound.play(-1)  # Play alarm sound in loop
                    alarm_playing = True
        elif left_blink == 1 and right_blink == 1:
            status_counts['drowsy'] += 1
            status_counts['sleep'] = status_counts['active'] = 0
            if status_counts['drowsy'] > 6:
                status, color = "Drowsy !", (0, 0, 255)
                if not alarm_playing:
                    alarm_sound.play(-1)  # Play alarm sound in loop
                    alarm_playing = True
        else:
            status_counts['active'] += 1
            status_counts['sleep'] = status_counts['drowsy'] = 0
            if status_counts['active'] > 6:
                status, color = "Active :)", (0, 255, 0)
                if alarm_playing:
                    alarm_sound.stop()  # Stop alarm sound
                    alarm_playing = False

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Heart rate calculation using face ROI mean intensity
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        mean_value = np.mean(roi_gray)

        if abs(mean_value - prev_mean_value) > 2 and (cv2.getTickCount() - prev_sec) / cv2.getTickFrequency() > 0.5:
            if prev_sec != 0:
                time_elapsed = (cv2.getTickCount() - prev_sec) / cv2.getTickFrequency()
                heart_rate = 60 / time_elapsed
                heart_rates.append(heart_rate)
                status = classify_heart_rate(heart_rate)

            prev_sec = cv2.getTickCount()

        prev_mean_value = mean_value

    if heart_rates:
        avg_heart_rate = np.mean(heart_rates[-10:])  # Calculate average of last 10 to smooth out the data
        cv2.putText(frame, f'Heart Rate: {avg_heart_rate:.1f} bpm ({status})',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    return prev_mean_value, prev_sec, status_counts


while True:
    ret, frame = cap.read()
    if not ret:
        break  # If frame not read correctly, exit loop

    prev_mean_value, prev_sec, status_counts = process_frame(frame, detector, predictor, status_counts, prev_mean_value,
                                                             prev_sec, heart_rates)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if alarm_playing:
            alarm_sound.stop()  # Stop alarm sound if 'q' is pressed
            alarm_playing = False
        break

cap.release()
cv2.destroyAllWindows()

