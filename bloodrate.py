import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for heart rate calculation
prev_mean_value = 0
prev_sec = 0
heart_rates = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract region of interest (ROI) containing the face
        roi_gray = gray[y:y+h, x:x+w]

        # Calculate mean pixel intensity in the ROI
        mean_value = np.mean(roi_gray)

        # Check for a significant change in pixel intensity (possible due to blood flow)
        if abs(mean_value - prev_mean_value) > 2 and (cv2.getTickCount() - prev_sec) / cv2.getTickFrequency() > 0.5:
            # Calculate heart rate based on time between peaks
            if prev_sec != 0:  # Ensuring we have a previous time to calculate the time elapsed
                time_elapsed = (cv2.getTickCount() - prev_sec) / cv2.getTickFrequency()
                heart_rate = 60 / time_elapsed
                heart_rates.append(heart_rate)

            # Reset timer
            prev_sec = cv2.getTickCount()

        # Update previous mean value
        prev_mean_value = mean_value

    # Display heart rate on the frame
    if heart_rates:
        avg_heart_rate = np.mean(heart_rates)
        cv2.putText(frame, f'Heart Rate: {avg_heart_rate:.1f} bpm', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

