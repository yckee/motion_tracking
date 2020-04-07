import numpy as np
import cv2
from collections import deque


def get_direction(dx, dy, ds, threshold=20):
    if np.abs(dx) > threshold:
        dirX = 'Right' if np.sign(dx) == 1 else 'Left'
    else:
         dirX = ''
    
    if np.abs(dy) > threshold:
        dirY = 'Down' if np.sign(dy) == 1 else 'Up'
    else:
        dirY = '' 

    if np.abs(ds) > threshold**2.4:
        dirS = 'Towards' if np.sign(ds) == 1 else 'Away'
    else:
        dirS = '' 
    
    return dirX, dirY, dirS

pts = deque(maxlen=32)
prev_frames = 10

lower_red = np.array([155,185,60])
upper_red = np.array([179,255,255])

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('motion_tracking_red.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (640, 480))

while True:
    ret, img = cap.read()

    if ret:
        blur = cv2.GaussianBlur(img, (11,11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)    

        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None
        size = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)

            x,y,w,h = cv2.boundingRect(c)
            size = w * h
            if size > 500:          
                center = (x + int(w/2), y + int(h/2))
                
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.circle(img,center,2,(0,255,0),2)

        pts.append((center, size))

        if len(pts) >= 10 and pts[-1][0] and pts[-prev_frames][0]:
            dx = pts[-1][0][0] - pts[-prev_frames][0][0]
            dy = pts[-1][0][1] - pts[-prev_frames][0][1]
            ds = pts[-1][1] - pts[-prev_frames][1]

            dirX, dirY, dirS = get_direction(dx, dy, ds)
            cv2.putText(img, f"{dirX} {dirY}", (10,30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"{dirS}", (10,60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

        writer.write(img)
        cv2.imshow('final', img)

        k = cv2.waitKey(10)
        if k == 27:
            break

cap.release()
writer.release()