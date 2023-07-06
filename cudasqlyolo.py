import torch
import numpy as np
import cv2
import mysql.connector
import time
import os
from datetime import datetime

# Load the YOLOv5 model
weights = "yolov5s.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)

# Load the dataset configuration
data = "data/coco128.yaml"
model.yaml = data

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# OpenCV setup for video capture
cap = cv2.VideoCapture(0)  # Use webcam (change the index if you have multiple cameras)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# SQL setup for saving detection results
mydb = mysql.connector.connect(
    host="localhost",
    user="your_username",
    password="your_password",
    database="object_detection"
)
mycursor = mydb.cursor()

# Create a table for storing detection results
mycursor.execute("CREATE TABLE IF NOT EXISTS detections (id INT AUTO_INCREMENT PRIMARY KEY, timestamp DATETIME, class VARCHAR(255), confidence FLOAT, x INT, y INT, width INT, height INT)")

# Folder setup for saving cropped detections
output_folder = 'detection_crops'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load class labels
class_labels = model.names

# Load classes and their numbers from SQL table
class_mapping = {}
mycursor.execute("SELECT class_name, class_num FROM class_mapping")
rows = mycursor.fetchall()
for row in rows:
    class_mapping[row[0]] = row[1]

# Load thresholds from SQL table
thresholds = {}
mycursor.execute("SELECT class_name, start_time, end_time FROM thresholds")
rows = mycursor.fetchall()
for row in rows:
    class_name = row[0]
    start_time = row[1]
    end_time = row[2]
    thresholds[class_name] = (start_time, end_time)

# Initialize variables for counting and time tracking
counters = {class_name: 0 for class_name in class_mapping.keys()}
start_time = time.time()
previous_time = start_time

# Object detection loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time < 0.1:
        continue

    # Convert the frame to the required format
    img = torch.from_numpy(frame).to(device)
    img = img.float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    # Perform object detection
    results = model(img)

    # Get detection information
    detections = results.pandas().xyxy[0]

    # Filter detections for specified classes
    filtered_detections = detections[detections['name'].isin(class_mapping.keys())]

    # Check if any specified classes are detected
    if len(filtered_detections) > 0:
        # Apply non-maximum suppression (NMS)
        filtered_detections = filtered_detections.sort_values(by='confidence', ascending=False)
        boxes = filtered_detections[['xmin', 'ymin', 'xmax', 'ymax']].values
        scores = filtered_detections['confidence'].values
        labels = filtered_detections['name'].map(class_mapping).values
        keep_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.5, 0.4).flatten()
        filtered_detections = filtered_detections.iloc[keep_indices]

        for _, detection in filtered_detections.iterrows():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            class_name = detection['name']
            class_num = class_mapping[class_name]

            # Crop the detection
            crop = frame[int(detection['ymin']):int(detection['ymax']),
                         int(detection['xmin']):int(detection['xmax'])]

            # Check if the counter exceeds the threshold for specified classes
            if counters[class_name] >= thresholds[class_name][0] and counters[class_name] <= thresholds[class_name][1]:
                # Save the cropped image with timestamp and location
                location = "bus_station_village_district"  # Replace with the actual location
                image_name = f"{class_name}_{timestamp}_{location}.jpg"
                cv2.imwrite(os.path.join(output_folder, image_name), crop)

            # Increment the counter
            counters[class_name] += 1

            # Log the detection in the SQL database
            sql = "INSERT INTO detections (timestamp, class, confidence, x, y, width, height) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (timestamp, class_num, detection['confidence'], int(detection['xmin']), int(detection['ymin']), int(detection['xmax'] - detection['xmin']), int(detection['ymax'] - detection['ymin']))
            mycursor.execute(sql, val)
            mydb.commit()

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(detection['xmin']), int(detection['ymin'])),
                          (int(detection['xmax']), int(detection['ymax'])), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name} {detection["confidence"]:.2f}',
                        (int(detection['xmin']), int(detection['ymin']) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with bounding boxes and labels
    cv2.imshow('Object Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:  # Press Esc to exit
        break

    previous_time = current_time

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

