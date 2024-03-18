import torch
import cv2
import numpy as np

def check_proximity_warning(distance, left_image):
    if distance is not None and distance < 6:

        cv2.putText(left_image, f"WARNING, CAR TOO CLOSE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5, cv2.LINE_AA)
        print("Warning: Car is too close!")

def calculate_distance(bbox, focal_length, baseline, disparity_map, left_boxes):
    # Extract bounding box coordinates
    x_min, y_min, x_max, y_max, _, _ = map(int, bbox)

    # Calculate the center of the bounding box
    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    # Create a mask to exclude overlapping regions for the current bounding box
    for box in left_boxes:
         if not np.array_equal(box, bbox):
            x1, y1, x2, y2, _, _ = map(int, box)
            if (x1 <= x_max and y1 <= y_max and x1 >= x_min and y1 >= y_min):
                disparity_map[y1:y_max, x1:x_max] = 0
            if (x2 >= x_min and x2 <= x_max and y2 >= y_min and y2 <= y_max):
                disparity_map[y_min:y2, x_min:x2] = 0

    # Extract the disparity region of interest (ROI)
    disparity_roi = disparity_map[y_min:y_max, x_min:x_max]

    # Check if the ROI contains valid data
    if np.count_nonzero(disparity_roi) == 0 or np.isnan(np.mean(disparity_roi)):
        return None

    # Calculate the mean disparity value within the ROI
    mean_disparity = np.mean(disparity_roi[disparity_roi != 0])

    # Check if the mean disparity value is valid
    if np.isnan(mean_disparity) or mean_disparity == 0:
        return None

    # Calculate the distance using triangulation formula
    distance = baseline * focal_length / mean_disparity

    return distance




def calculate_disparity_map(img_left, img_right):
    # Convert images to grayscale
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Initialize stereo block matching object
    stereo = cv2.StereoBM_create(numDisparities=80, blockSize=15)

    # Compute disparity map
    disparity = stereo.compute(img_left_gray, img_right_gray)

    # Normalize the disparity map for display
    disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return disparity_normalized

focal_length = 2063
baseline = 0.5446

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load left and right images
left_image_path = './Images/image_L/2018-10-11-16-03-19_2018-10-11-16-05-28-455.png'
right_image_path = './Images/image_R/2018-10-11-16-03-19_2018-10-11-16-05-28-455.png'

left_image = cv2.imread(left_image_path)
right_image = cv2.imread(right_image_path)

# YOLOv5 Inference on left and right images
left_results = model([left_image_path])
right_results = model([right_image_path])

# Extract bounding boxes and other information
left_boxes = left_results.xyxy[0].cpu().numpy()
right_boxes = right_results.xyxy[0].cpu().numpy()

disparity_map = calculate_disparity_map(left_image, right_image)
# Draw bounding boxes, centers, and distances on left image
for box in left_boxes:
    x_min, y_min, x_max, y_max, _, category = map(int, box)
    probability = box[4]
    if (category == 2 or category == 5 or category == 7) and probability >= 0.5:

        cv2.rectangle(left_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        cv2.circle(left_image, (x_center, y_center), 5, (255, 0, 0), -1)

        # Calculate distance
        dist = calculate_distance(box, focal_length, baseline, disparity_map, left_boxes)
        if dist is not None:
            cv2.putText(left_image, f"{dist:.2f}m", (x_center - 50, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        # After calculating distance, add this line to check proximity and display warning if necessary
        check_proximity_warning(dist,left_image)

# Display the disparity map
cv2.imshow('Disparity Map', left_image)
cv2.imshow('Right', right_image)
cv2.imshow('Disparity', disparity_map)
cv2.waitKey(0)
cv2.destroyAllWindows()