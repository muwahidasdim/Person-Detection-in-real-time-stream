
import cv2
import numpy as np
import random
import time
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# Function to generate random color avoiding shades of red
def generate_color():
    while True:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if color[2] < 100:  # Avoid shades of red (i.e., low values in the red channel)
            return color

# Start capturing video from the RTSP stream
cap = cv2.VideoCapture("rtsp://192.168.1.100:8080/h264.sdp")

# Set up the video window
cv2.namedWindow("RTSP Stream")
if not cap.isOpened():
    print("Cannot open RTSP stream")
    exit()

# Function to handle mouse click events
def select_person(event, x, y, flags, param):
    global selected_person, timers, selected_tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        for person_id, (box, color) in box_colors.items():
            if box[0] < x < box[2] and box[1] < y < box[3]:
                if selected_person != person_id:
                    if selected_person:
                        box_colors[selected_person] = (box_colors[selected_person][0], generate_color())
                    selected_person = person_id
                    timers[selected_person] = time.time()
                    box_colors[selected_person] = (box_colors[selected_person][0], (0, 0, 255))

                    # Initialize a new tracker for the selected person
                    selected_tracker = cv2.TrackerKCF_create()
                    bbox = (box[0], box[1], box[2] - box[0], box[3] - box[1])
                    selected_tracker.init(frame, bbox)
# Function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Calculate areas of both bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

# Set the mouse callback function
cv2.setMouseCallback("RTSP Stream", select_person)

# Dictionary to store bounding box colors and timers
box_colors = {}
timers = {}
selected_person = None
selected_tracker = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 object detection
    results = model(frame)

         # Extract bounding boxes for people
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    new_box_colors = {}
    for box, cls, conf in zip(boxes, classes, confidences):
        xmin, ymin, xmax, ymax = box
        confidence = conf
        class_id = cls
        class_name = names[int(cls)]
        if class_name == "person":
            new_person_id = f"{xmin}-{ymin}-{xmax}-{ymax}"
            found_existing_person = False
            for person_id, (prev_box, color) in box_colors.items():
                iou = calculate_iou(prev_box, box)
                if iou > 0.5:
                    # Update the existing person's bounding box
                    box_colors[person_id] = ((xmin, ymin, xmax, ymax), color)
                    found_existing_person = True
                    new_person_id = person_id
                    break
            if not found_existing_person:
                # Create a new person entry
                box_colors[new_person_id] = ((xmin, ymin, xmax, ymax), generate_color())
            new_box_colors[new_person_id] = box_colors[new_person_id]

    box_colors = new_box_colors
        
    # Update the selected person's tracker
    if selected_person and selected_tracker:
        success, bbox = selected_tracker.update(frame)
        if success:
            xmin, ymin, w, h = [int(v) for v in bbox]
            xmax, ymax = xmin + w, ymin + h
            box_colors[selected_person] = ((xmin, ymin, xmax, ymax), (0, 0, 255))

    # Draw bounding boxes
    for person_id, (box, color) in box_colors.items():
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)

    # Display timer for selected person
    if selected_person in box_colors:
        timer_start = timers.get(selected_person, time.time())
        elapsed_time = int(time.time() - timer_start)
        cv2.putText(frame, f"Timer: {elapsed_time}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
