import cv2
from ultralytics import YOLO
import supervision as sv
from screeninfo import get_monitors  # Import screeninfo to get screen resolution
import time
import datetime
import csv

# Load the YOLOv8 model
model = YOLO("yolo11n.pt")  # Replace with your custom model if needed
model.export(format="ncnn")  # Creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("yolo11n_ncnn_model")

# Initialize the tracker and annotators
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open the default webcam (0 for the first webcam)
cap = cv2.VideoCapture(0)

# Automatically get screen width and height
monitor = get_monitors()[0]  # Get the first monitor (in case of multi-monitor setup)
screen_width = monitor.width
screen_height = monitor.height

# Set camera resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Open the CSV file to write the log
csv_file = open("/home/rapberryiitj/temp/persons_detected_records/person_detection_log_1.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Write the header to the CSV file
csv_writer.writerow(["Date", "Time", "Number of People Detected"])

# Set the window to fullscreen
cv2.namedWindow("Person Detection Only", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Person Detection Only", cv2.WND_PROP_FULLSCREEN, 1)

# Variables for FPS calculation and logging
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = ncnn_model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # Filter for person (class_id == 0)
    detections = tracker.update_with_detections(detections)

    # Count number of people detected in the current frame
    num_people_detected = len(detections)

    # Get the current date and time
    now = datetime.datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Log the number of people detected with date and time to the CSV file
    if num_people_detected>0:
        csv_writer.writerow([current_date, current_time, num_people_detected])

    # Annotate the frame (optional)
    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

    # Show the live video (optional)
    cv2.imshow("Person Detection Only", annotated_frame)

    # Calculate FPS every second
    frame_count += 1
    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = end_time

        # Optionally, display FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the CSV file after the loop ends
csv_file.close()

# Release resources
cap.release()
cv2.destroyAllWindows()
