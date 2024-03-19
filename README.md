# Deep Learning in Autonomous Vehicle Systems and Vehicle Networks

## Running the code

source Scripts/activate

### To run distance detection on all images
python3 distance_detection.py

### To run distance detection on single image
python3 single_image.py
NOTE: change image path in code

## Project Description

The goal of this project was to take images from stereo camera and detect distance to other vehicles and notify if the vehicle is too close.

Dataset used can be found on: https://drivingstereo-dataset.github.io/

### Object Detection

For the purpose of object detection, a pretrained YOLOv5 model was used. Output from object detection is a list of 6 values:

- x coordinate of top-left corner of object boundary
- y coordinate of top-left corner of object boundary
- x coordinate of bottom-right corner of object boundary
- y coordinate of bottom-right corner of object boundary
- probability of object being a certain class
- class of the object detected

### Calculating distance to detected objects

In order to calculate distance, a formula can be used:  Z = f * b/d

- Z is depth(distance)
- f is focal point of the camera
- b is baseline, representing distance between the center of the cameras
- d is disparity of certain point between inputs of both cameras

Focal length and baseline are dependent on specifics of cameras and their calibration. These values can be found in 2018-07-11-14-48-52.txt file provided.

#### Disparity calculation

In order to calculate distance, first step was to calculate disparity. That was done in function  **calculate_disparity_map**.

Disparity map represents the distance between two sets of coordinates for the same scene point. In order to calculate the disparity map, images were loaded as grayscale. Afterwards the StereoBM_create function from cv2 library was used, and using trial and error, the optimal parameters were found to be numDisparities=80 and blockSize=15. Under the output of StereoBM_create, compute function was called, and as a result disparity map was created. Finally, map was normalized to reduce errors and emphasise the disparities. 

#### Distance calucation

The calulation was done in function calculate_distance. Firstly, coordinates of bounding box of detected objects were extracted. Afterwars the mask was created in order to eliminate overlapping parts of bounding boxes if there are any (i.e. if vehicles are too close to eachother). Next step was to define region of interest(ROI) of the disparity map. Bounding box of object was used as ROI. Furthermore, mean value of disparity ROI was calulated, excluding pixels whose value is 0. Finally, distance is calculated with the formula mentioned above.

#### Main loop

As a result, we have done distance detection using abovementioned method for each picture from the dataset. As output the bounding boxes, whose class is 2, 5 or 7(which represent vehicles) and whose probability is greater than 0.6,  were marked on each entry, their center point and the distance to the object was written.

[image0]: ./Images/image_L/2018-08-07-13-46-08_2018-08-07-14-14-18-815.png "Original Picture"
[image1]: ./Disparity_815.png "Disparity"
[image2]: ./output/result_181.png "Processed Picture"

![alt text][image0]
![alt text][image1]


In the disparity image, black patches can be found, indicating overlapping regions that are not included in calculation of mean disparity of ROI.


![alt text][image2]


All inputs can be found in Images/ folder
All outputs can be found in output/ folder.


