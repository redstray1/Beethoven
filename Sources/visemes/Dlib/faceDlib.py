import cv2
import numpy as np
import matplotlib.pyplot as plt
import landmarksModule as face

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(cap.isOpened())
print('Width: ', cap.get(3))
print('Height: ', cap.get(4))

detector = face.LandmarksDetector()
fig = plt.figure()
while True:
    status, frame = cap.read()
    if not status:
        print('Frame not captured')
        exit()
    img,faces = detector.getFacesLandmarks(frame, screenPositions=False)
    # if faces.size != 0:
    #     fig.clear()
    #     fc = faces[0]
    #     mu = np.mean(fc)
    #     fc -= mu
    #     plt.scatter(fc[:, 0], -fc[:, 1])
    #     fig.canvas.draw()
    #     lmImg = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    #             sep='')
    #     lmImg  = lmImg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     lmImg = cv2.cvtColor(lmImg,cv2.COLOR_RGB2BGR)
    #     cv2.imshow('video feed', lmImg)
    cv2.imshow('video feed', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()