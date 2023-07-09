import cv2
import numpy as np
import time
from yolov5.detect import YoloDetector

# Define the path to the trained.pt file
model_weights = "trained.pt"

# Select specific classes that you want to detect and assign a color to each detection
selected = {
    "cat": (0, 255, 255),
    "dog": (0, 0, 0),
    "person": (255, 0, 0),
    "bus": (0, 255, 0),
    "car": (0, 0, 255)
}

# Initialize the detector with the path to the trained.pt file
detector = YoloDetector(model_weights)

# Initialize video stream
cap = cv2.VideoCapture("input.mp4")

# Read the first frame
ret, frame = cap.read()

# Loop to read frames and update the window
while ret:
    start = time.time()

    # Perform object detection using the detector
    detections = detector.detect(frame)

    # Loop over the selected items and check if they exist in the detected items
    for cls, color in selected.items():
        if cls in detections:
            for box in detections[cls]:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=1)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    end = time.time()
    cv2.putText(frame, "fps: %.2f" % (1 / (end - start)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 0, 0))

    # Display the detections
    cv2.imshow("Detections", frame)

    # Wait for key press
    key_press = cv2.waitKey(1) & 0xFF

    # Exit loop if 'q' is pressed or on reaching EOF
    if key_press == ord('q'):
        break

    # Read the next frame
    ret, frame = cap.read()

# Release resources
cap.release()

# Destroy the window
cv2.destroyAllWindows()
