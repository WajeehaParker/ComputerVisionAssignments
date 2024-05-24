import cv2
import os
import numpy as np

####################### START FUNCTIONS ################################

def getFramesAndAngleValuesForPendulum(video_path):
    # Create video capture object
    cap = cv2.VideoCapture(video_path)

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take the first frame and detect corners
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    img_data=[]
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good point
        good_new = p1[0]

        # Draw circle around the tracked point
        a, b = good_new.ravel()
        cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

        # Calculate timestamp
        timestamp = frame_count / fps

        #adding to list
        img_data.append((float(a), float(b), frame, timestamp))

        # Update the previous frame and points
        old_gray = frame_gray.copy()
        p0 = np.array([good_new], dtype=np.float32)

        frame_count += 1

    cap.release()
    return img_data

def getExtremePointData(img_data):    
    min_point = None
    max_point = None
    i=0

    if(img_data[0][0] > img_data[1][0]): #first going left
        while(img_data[i][0] > img_data[i+1][0]):
            min_point = img_data[i+1]
            i += 1
        while(img_data[i][0] < img_data[i+1][0]):
            max_point = img_data[i+1]
            i += 1
    else:                               #first going right
        while(img_data[i][0] < img_data[i+1][0]):
            max_point = img_data[i+1]
            i += 1
        while(img_data[i][0] > img_data[i+1][0]):
            min_point = img_data[i+1]
            i += 1

    return min_point, max_point 

####################### END FUNCTIONS ################################

video_path = 'Simple Pendulum.mp4'

img_data = getFramesAndAngleValuesForPendulum(video_path)
min_point, max_point = getExtremePointData(img_data)

min_a, min_b, min_frame, min_timestamp = min_point
max_a, max_b, max_frame, max_timestamp = max_point

timeperiod = max_timestamp - min_timestamp

print("Time Period for Pendulum: " + str(timeperiod) + " seconds")