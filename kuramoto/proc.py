import cv2, sys
import numpy as np

#Process video for playback
fname = sys.argv[1]
path = sys.argv[2]
nearest_scale = 4
upscale = 2

cap = cv2.VideoCapture(sys.argv[1])
shape = None

if not cap.isOpened():
    print("File not found")
while True:
    ret, frame = cap.read()
    if type(frame) == type(None):
        continue
    if type(shape) == type(None):
        shape = (frame.shape[0]*nearest_scale*2**upscale, frame.shape[1]*nearest_scale*2**upscale)
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('X','2','6','4'), 15, shape, 1)

    f_n = cv2.resize(frame, (frame.shape[0]*nearest_scale,frame.shape[1]*nearest_scale), interpolation=cv2.INTER_NEAREST)
    for i in range(upscale):
        f_n = cv2.pyrUp(f_n)
    out.write(f_n)
    #cv2.imshow("Display", f_n)
    #if chr(cv2.waitKey(2) & 0xFF) == "q":
        #break

out.release()
