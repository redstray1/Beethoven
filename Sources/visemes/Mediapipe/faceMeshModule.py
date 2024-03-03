import cv2
import numpy as np
import mediapipe as mp

class FaceMeshDetector():

    def __init__(self, staticMode = False, max_faces = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.staticMode = staticMode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            refine_landmarks=True,
            max_num_faces = self.max_faces,
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence = self.min_tracking_confidence)

        self.draw_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def getFacesLandmarks(self, img, draw=True, screenPositions=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.draw_spec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x * iw), int(lm.y*ih)
                    if screenPositions:
                        face.append([x,y])
                    else:
                        face.append([lm.x,lm.y])
            faces.append(face)
        return img, np.array(faces)
