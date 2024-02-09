import cv2 # pip install opencv-python
import numpy as np
from ultralytics import YOLO # pip install ultralytics after installing pytorch
# pip install torch torchvision torchaudio


# create a videocapture object

# Video by Anna Bondarenko: https://www.pexels.com/video/person-playing-with-dogs-9252757/
# load the dogs_playing.mp4 video
cap = cv2.VideoCapture("datasets/dogs_playing.mp4")

# instantiate the YOLOv8 model
model = YOLO("yolov8m.pt")

while True:
# capture a frame from the video
    ret, frame = cap.read()
    # if there are no frames to capture, break out of the loop
    if not ret:
        break
    results = model(frame,device="mps") # prediction from the model for each frame of the video. in windows "Cuda"
    print(results) # printing just to show the contents of the results object. later we can disable this to avoid unnecessary prints
    result = results[0] # access the first element of the the results object for the first frame
    bboxes = np.array(result.boxes.xyxy.cpu(),dtype="int") # getting the coordinates of the bounding boxes
    classes = np.array(result.boxes.cls.cpu(),dtype="int") # getting the classes of the objects detected in the bounding boxes


    for cls,bbox in zip(classes,bboxes):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # in the frame draw a rectange with center at (x1,y1) and width and height of x2 and y2 respectively in red color with line thickness of 2
        class_names_dict = model.names # get the class names from the model
        # extract the corresponding class name of the class id from the class_names list
        class_name = class_names_dict[cls]
        cv2.putText(frame, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# display the captured frame

    cv2.imshow('frame', frame)
    key = cv2.waitKey(0)
    # stop the loop on key press 'esc'
    if key == 27:
        break

# release the videocapture object
cap.release()
# close all windows
cv2.destroyAllWindows()