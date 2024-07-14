
# importing libraries
import cv2
import numpy as np
from functions import getBGSubtractor, calculate_time
from FrameDifferencing import remove_contained_bboxes, non_max_suppression, draw_bboxes

def get_flow_viz(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb
    

# Create a VideoCapture object
cap = cv2.VideoCapture("C:\CB041446.mp4") #fill with appropiate path

#define substractor
fgbg = getBGSubtractor("MOG2")

# set minimum area size for object detection
min_area = 100 #pixels

# set scale of resized image
scale = 0.5

# Start the timer
start_time = cv2.getTickCount()

# read first frame from video input
ret, frame = cap.read()

# convert first frame to grayscale as optical flow is typically used with grayscale images
prvs = cv2.cvtColor(frame, 
                        cv2.COLOR_BGR2GRAY) 
prvs = cv2.GaussianBlur(prvs, dst=None, ksize=(3,3), sigmaX=5) #larger sigma = more blur

# create HSV image for optical flow visualization
hsv = np.zeros_like(frame) #black image
hsv[..., 1] = 255 #all pixels have maximum saturiation (easier to interpret)

# params for corner detection 
feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 7, 
                       blockSize = 7 ) 
  
# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (5, 5), 
                  maxLevel = 3, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03))


p0 = cv2.goodFeaturesToTrack(prvs, mask = None, 
                             **feature_params) 

# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 

# Create a mask image for drawing purposes 
mask = np.zeros_like(frame) 
# Get the frame rate (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")
# Assuming timestamps are available from the video capture object
start_time = cv2.getTickCount()  # Get start time in ticks

# Desired delay in seconds
desired_delay = 2

# Read until video is completed (ret!=True)
while ret:
    try:
        ret, frame = cap.read()
        # Get current timestamp in ticks
        current_time = cv2.getTickCount()

        # Calculate elapsed time in seconds
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()

        # Exit the loop if desired delay is reached
        if elapsed_time >= desired_delay:
            # convert current frame to grayscale
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next = cv2.GaussianBlur(next, dst=None, ksize=(3,3), sigmaX=5)
            
            # calculate optical flow between previous and current frame
            
            # Calculate optical flow using Lucas-Kanade method
            p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, 
                                                next, 
                                                p0, None, 
                                            **lk_params) 
                        # Select good points 
            good_new = p1[st == 1] 
            good_old = p0[st == 1] 
                # draw the tracks 
            for i, (new, old) in enumerate(zip(good_new,  
                                            good_old)): 
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), 
                                color[i].tolist(), 2) 
                
                frame = cv2.circle(frame, (a, b), 5, 
                                color[i].tolist(), -1) 
                
            #img = cv2.add(frame, mask) 
            #cv2.imshow('frame', img)
            
            # Calculate optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                            pyr_scale=0.75,
                            levels=3,
                            winsize=5,
                            iterations=3,
                            poly_n=10,
                            poly_sigma=1.2,
                            flags=0)
            print("Flow shape",flow.shape)
            rgb = get_flow_viz(flow)
            #cv2.imshow("rgb scaled",rgb*50)
            
            #Choosing dense optical flow as it is more accurate
            
            # Calculate magnitude and angle of optical flow vectors
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            #post processing
            motion_mask = np.uint8(mag > 0.5)*255 # 1 (indicating significant motion) 
            motion_mask = cv2.erode(motion_mask, np.ones((1,1), dtype=np.uint8), iterations=1) #Removes small isolated bright spots (potential noise)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((1,1), dtype=np.uint8), iterations=1) #Removes thin foreground (motion) regions completely surrounded by background.
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, np.ones((1,1), dtype=np.uint8), iterations=3) 
            #cv2.imshow('Motion mask', motion_mask)

            # Find contours in binary image
            contours, hierarchy = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            temp_mask = np.zeros_like(mask) # used to get flow angle of contours
            angle_thresh = 2*ang.std()
            detections = []
            # Loop through contours and filter by minimum area size
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour)
                area = w*h
                cv2.drawContours(temp_mask, [contour], 0, (255,), -1)
                flow_angle = ang[np.nonzero(temp_mask)]
                if (area> 400) and (flow_angle.std() < angle_thresh):
                    detections.append([x,y,x+w,y+h, area])
                    # Draw bounding box around contour
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detections = np.array(detections)
            # separate bboxes and scores
            bboxes = detections[:, :4]
            scores = detections[:, -1]
            
            nms_bboxes = non_max_suppression(bboxes, scores, threshold=0.1)
            
            draw_bboxes(frame, nms_bboxes)
            # display output frame
            cv2.imshow("Frame", frame)
            #cv2.waitKey(0) 
    except not ret:
        print("Error reading frame")
        break
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        break
    except ValueError as ve:
        print(ve)
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        break
    
# Stop the timer
end_time = cv2.getTickCount()

#calculate the execution time
calculate_time(start_time, end_time)

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

