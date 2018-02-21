import cv2, sys

cap = cv2.VideoCapture(sys.argv[1])
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("File not found")
while True:
    ret, frame = cap.read()
    cv2.imshow("Display", frame)
    if chr(cv2.waitKey(2) & 0xFF) == "q":
        break
