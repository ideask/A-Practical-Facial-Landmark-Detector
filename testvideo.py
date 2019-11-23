import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    print(frame.shape[:2])
    print(type(frame))
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
