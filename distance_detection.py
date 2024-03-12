import torch
import cv2
import numpy as np

def calculate_distance(bbox, focal_length, baseline, disparity_map):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max, _, _ = bbox

    # Calculate the center of the bounding box
    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    # Extract the disparity value at the center of the bounding box
    disparity = disparity_map[y_center, x_center]

    # Check if the disparity value is valid
    if disparity == 0:
        return None

    # Calculate the distance using triangulation formula
    distance = baseline * focal_length / disparity

    return distance


focal_length = 2063  
baseline = 0.5446

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load left and right images
left_image_path = './Images/image_L/2018-07-11-14-48-52_2018-07-11-15-09-59-868.png'
right_image_path = './Images/image_R/2018-07-11-14-48-52_2018-07-11-15-09-59-868.png'

left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)

# YOLOv5 Inference on left and right images
left_results = model([left_image_path])
right_results = model([right_image_path])

# Extract bounding boxes and other information
left_boxes = left_results.xyxy[0].cpu().numpy()
right_boxes = right_results.xyxy[0].cpu().numpy()

# Load the disparity image
disparity_image = cv2.imread('./Images/disparity/2018-07-11-14-48-52_2018-07-11-15-09-59-868.png', cv2.IMREAD_GRAYSCALE)

# Draw bounding boxes, centers, and distances on left image
for box in left_boxes:
    x_min, y_min, x_max, y_max, _, _ = map(int, box)
    cv2.rectangle(left_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)
    cv2.circle(left_image, (x_center, y_center), 5, (255, 0, 0), -1)
    
    # Calculate distance
    dist = calculate_distance(box, focal_length, baseline, disparity_image)
    if dist is not None:
        cv2.putText(left_image, f"{dist:.2f}m", (x_center - 50, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('Left Image with Bounding Boxes, Centers, and Distances', left_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
