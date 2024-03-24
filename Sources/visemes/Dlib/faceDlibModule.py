import cv2
import dlib
import os
import numpy as np

class LandmarksDetector():
    
    def __init__(self):
        self.Model_PATH = "shape_predictor_68_face_landmarks.dat"
        self.frontalFaceDetector = dlib.get_frontal_face_detector()
        pwd = os.path.relpath(os.path.dirname(__file__))
        self.faceLandmarkDetector = dlib.shape_predictor(pwd + '\\' + self.Model_PATH)

    def drawPoints(self, image, faceLandmarks, startpoint, endpoint, isClosed=False):
        points = []
        for i in range(startpoint, endpoint+1):
            point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
            points.append(point)

        points = np.array(points, dtype=np.int32)
        cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

    def facePoints(self, image, faceLandmarks):
        assert(faceLandmarks.num_parts == 68)
        self.drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
        self.drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
        self.drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
        self.drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
        self.drawPoints(image, faceLandmarks, 30, 35, True)    # Lower nose
        self.drawPoints(image, faceLandmarks, 36, 41, True)    # Left eye
        self.drawPoints(image, faceLandmarks, 42, 47, True)    # Right Eye
        self.drawPoints(image, faceLandmarks, 48, 59, True)    # Outer lip
        self.drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip    

    def getFacesLandmarks(self, img, draw=True, screenPositions=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        allFaces = self.frontalFaceDetector(imgRGB, 0)
        faces = []
        ih, iw, ic = img.shape
        for k in range(0, len(allFaces)):
            faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
                int(allFaces[k].right()),int(allFaces[k].bottom()))
            detectedLandmarks = self.faceLandmarkDetector(imgRGB, faceRectangleDlib)
            face = []
            for i in range(68):
                if screenPositions:
                    face.append([detectedLandmarks.part(i).x, detectedLandmarks.part(i).y])
                else:
                    face.append([detectedLandmarks.part(i).x / iw, detectedLandmarks.part(i).y / ih])
            faces.append(face)
            if draw:
                self.facePoints(img, detectedLandmarks)
        return img, np.array(faces)
