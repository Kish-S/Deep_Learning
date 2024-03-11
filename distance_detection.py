import torch
import cv2
import numpy as np

def calculate_overlap(box1, box2):
    x1, y1, w1, h1, _, _ = box1
    x2, y2, w2, h2, _, _ = box2

    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

    intersection = x_overlap * y_overlap
    area1 = w1 * h1
    area2 = w2 * h2

    iou = intersection / (area1 + area2 - intersection)
    return iou


focal_length = 2063.400  # Focal length for camera 103 in pixels
baseline = 0.5446076  # Baseline for camera 103 in meters
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

print(f"Left boxes: {left_boxes}")
print(f"Right boxes: {right_boxes}")

# Load the disparity image
disparity_image = cv2.imread('./Images/disparity/2018-07-11-14-48-52_2018-07-11-15-09-59-868.png', cv2.IMREAD_GRAYSCALE)

# Match bounding boxes (simple overlap-based matching for demonstration)
matched_boxes = []
for left_box in left_boxes:
    for right_box in right_boxes:
        overlap = calculate_overlap(left_box, right_box)
        if overlap > 0.5:  # Adjust the threshold as needed
            matched_boxes.append((left_box, right_box))
            break

# Draw bounding boxes and annotations on the images
for left_box, right_box in matched_boxes:
    left_x, left_y, left_w, left_h, _, obj_type = map(int, left_box)
    disparity_roi = disparity_image[left_y:left_y+left_h, left_x:left_x+left_w]

    # Calculate average disparity within the bounding box
    average_disparity = np.mean(disparity_roi)

    # Calculate depth using stereo calibration parameters

    depth = baseline * focal_length / (average_disparity + 1e-6)

    # Draw bounding box and annotation on the left image
    cv2.rectangle(left_image, (left_x, left_y), (left_x + left_w, left_y + left_h), (0, 255, 0), 2)
    cv2.putText(left_image, f"{obj_type}: {depth:.2f} meters", (left_x, left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display or save the modified left image
cv2.imshow('Result Image', left_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
