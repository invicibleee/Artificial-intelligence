 import cv2
# frameWidth = 480
# frameHeight = 480
# cap = cv2.VideoCapture(0)
# cap.set(1, frameWidth)
# cap.set(1, frameHeight)
# cap.set(10,150)
# while True:
#     success, img = cap.read()
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


img = cv2.imread("kirii.jpg")

cv2.imshow("kirii",img)
cv2.waitKey(0)