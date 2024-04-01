import cv2
import numpy as np
import mediapipe as mp
import os

class VideoFaceCropper():

    #face box options
    SHORT_RANGE = 0
    LONG_RANGE = 1

    #landmarks options
    STATIC_MODE = True
    TRACKING_MODE = False

    def __init__(self, min_face_detector_confidence=0.5, face_detector_model_selection=LONG_RANGE,
                 landmark_detector_static_image_mode=STATIC_MODE, min_landmark_detector_confidence=0.5):
        self._LEFT_EYE_LANDMARK_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self._RIGHT_EYE_LANDMARK_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_face_detector_confidence,
                                                                       model_selection=face_detector_model_selection)

        self.landmark_detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                                 static_image_mode=landmark_detector_static_image_mode,
                                                                 min_detection_confidence=min_landmark_detector_confidence)

    def _get_bounding_box_inflation_factor(self, eye_coords, amplification=2, base_inflation=1):
        roll_angle = self._get_face_roll_angle([eye_coords[0].x, 1 - eye_coords[0].y], [eye_coords[1].x, 1 - eye_coords[1].y])
        inflation_factor = np.abs(roll_angle) / 90

        return base_inflation + (inflation_factor * amplification)
    
    def _get_inflated_face_image(self, image, face_box, inflation):

        width_inflation, height_inflation = face_box.width * inflation, face_box.height * inflation

        return self._crop_within_bounds(
            image,
            round((face_box.ymin - height_inflation / 2) * image.shape[0]),
            round((face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0]),
            round((face_box.xmin - width_inflation / 2) * image.shape[1]),
            round((face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1])
        )

    def _get_eyes_centres(self, left_eye_landmarks, right_eye_landmarks):
        left_eye_centre = np.array([
            np.sum([lm.x for lm in left_eye_landmarks]) / len(left_eye_landmarks),
            1 - np.sum([lm.y for lm in left_eye_landmarks]) / len(left_eye_landmarks)
        ])

        right_eye_centre = np.array([
            np.sum([lm.x for lm in right_eye_landmarks]) / len(right_eye_landmarks),
            1 - np.sum([lm.y for lm in right_eye_landmarks]) / len(right_eye_landmarks)
        ])
        return left_eye_centre, right_eye_centre

    def _get_eyes_midpoint(self, left_eye_centre, right_eye_centre, image_size):

        return np.ndarray.astype(np.rint(np.array([
            (left_eye_centre[0] + right_eye_centre[0]) * image_size[1] / 2,
            (left_eye_centre[1] + right_eye_centre[1]) * image_size[0] / 2
        ])), np.int32)
    
    def _get_face_roll_angle(self, left_eye_centre, right_eye_centre):

        if right_eye_centre[1] == left_eye_centre[1]:
            if left_eye_centre[0] >= right_eye_centre[0]:
                return 0
            else:
                return 180
        elif right_eye_centre[0] == left_eye_centre[0]:
            if left_eye_centre[1] > right_eye_centre[1]:
                return 90
            else:
                return -90
        else:
            gradient = (left_eye_centre[1] - right_eye_centre[1]) / (left_eye_centre[0] - right_eye_centre[0])
            if left_eye_centre[0] > right_eye_centre[0]:
                return np.degrees(np.arctan(gradient))
            else:
                return 180 + np.degrees(np.arctan(gradient))
            
    def _rotate_landmarks(self, landmarks, rotation_matrix, image_size):
        return np.ndarray.astype(np.rint(np.matmul(rotation_matrix, np.array([
            np.multiply([lm.x for lm in landmarks], image_size[1]),
            np.multiply([lm.y for lm in landmarks], image_size[0]),
            np.ones(len(landmarks))
        ]))), np.int32)

    def _get_roll_corrected_image_and_landmarks(self, face_image, face_landmarks):
        left_eye_centre, right_eye_centre = self._get_eyes_centres(
            [face_landmarks[lm - 1] for lm in self._LEFT_EYE_LANDMARK_INDICES],
            [face_landmarks[lm - 1] for lm in self._RIGHT_EYE_LANDMARK_INDICES]
        )

        eyes_midpoint = self._get_eyes_midpoint(left_eye_centre, right_eye_centre, face_image.shape)
        roll_angle = self._get_face_roll_angle(left_eye_centre, right_eye_centre)
        rotation_matrix = cv2.getRotationMatrix2D((round(eyes_midpoint[0]), round(eyes_midpoint[1])), -roll_angle, 1)

        return cv2.warpAffine(face_image, rotation_matrix, (face_image.shape[1], face_image.shape[0])), self._rotate_landmarks(face_landmarks, rotation_matrix, face_image.shape)

    def _crop_within_bounds(self, image, top, bottom, left, right):

        if top < 0:
            top = 0
        elif top >= image.shape[0]:
            top = image.shape[0] - 1
        
        if bottom < 0: 
            bottom = 0
        elif bottom >= image.shape[0]:
            bottom = image.shape[0] - 1
        
        if left < 0:
            left = 0
        elif left >= image.shape[1]:
            left = image.shape[1] - 1

        if right < 0:
            right = 0
        elif right >= image.shape[1]:
            right = image.shape[1] - 1
        
        return image[top:bottom+1, left:right+1]

    def debug(self, image, *, correct_roll = True):
        face_images = []

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected_faces = self.face_detector.process(imgRGB).detections

        if detected_faces is None:
            return face_images

        image_debug = imgRGB.copy()
        for face in detected_faces:
            #face box
            face_box = face.location_data.relative_bounding_box
            cv2.rectangle(
                image_debug,
                (round(face_box.xmin * image.shape[1]), round(face_box.ymin * image.shape[0])),
                (round((face_box.xmin + face_box.width) * image.shape[1]),
                 round((face_box.ymin + face_box.height) * image.shape[0])),
                (0, 255, 0)
            )

            #eyes
            eyes = face.location_data.relative_keypoints[:2]
            inflation_factor = self._get_bounding_box_inflation_factor(eyes)
            for eye in eyes[:1]:
                cv2.circle(image_debug, (round(eye.x * image.shape[1]), round(eye.y * image.shape[0])), 1, (0, 255, 0))
            for eye in eyes[1:]:
                cv2.circle(image_debug, (round(eye.x * image.shape[1]), round(eye.y * image.shape[0])), 1, (0, 255, 0))

            #eyes midpoint
            detected_landmarks = self.landmark_detector.process(image_debug).multi_face_landmarks
            if detected_landmarks is not None:
                face_landmarks = detected_landmarks[0].landmark
                left_eye_centre, right_eye_centre = self._get_eyes_centres(
                    [face_landmarks[lm] for lm in self._LEFT_EYE_LANDMARK_INDICES],
                    [face_landmarks[lm] for lm in self._RIGHT_EYE_LANDMARK_INDICES]
                )
                eye_midpoint = self._get_eyes_midpoint(left_eye_centre, right_eye_centre, image_debug.shape)
                cv2.circle(image_debug, (eye_midpoint[0], image_debug.shape[0] - eye_midpoint[1]), 2, (255, 0, 255))

            #roll line
            cv2.line(
                image_debug,
                (round(eyes[0].x * image.shape[1]), round(eyes[0].y * image.shape[0])),
                (round(eyes[1].x * image.shape[1]), round(eyes[1].y * image.shape[0])),
                (255, 0, 0)
            )

            # horizontal line
            cv2.line(
                image_debug,
                (round(eyes[0].x * image.shape[1]), round(eyes[0].y * image.shape[0])),
                (round(eyes[1].x * image.shape[1]), round(eyes[0].y * image.shape[0])),
                (255, 0, 0)
            )

            #roll angle text
            cv2.putText(
                image_debug,
                'roll_angle: {:.2f}'.format(self._get_face_roll_angle([eyes[1].x, 1 - eyes[1].y], [eyes[0].x, 1 - eyes[0].y])),
                (round(eyes[0].x * image.shape[1]), round(eyes[0].y * image.shape[0] + 40)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255)
            )

            #inflation box
            width_inflation, height_inflation = face_box.width * inflation_factor, face_box.height * inflation_factor
            cv2.rectangle(
                image_debug,
                (round((face_box.xmin - width_inflation / 2) * image.shape[1]), round((face_box.ymin + face_box.height + height_inflation / 2) * image.shape[0])),
                (round((face_box.xmin + face_box.width + width_inflation / 2) * image.shape[1]), round((face_box.ymin - height_inflation / 2) * image.shape[0])),
                (0, 0, 255)
            )


        #cv2.imshow('image_debug', cv2.cvtColor(image_debug, cv2.COLOR_RGB2BGR))
        face_images.append(image_debug)
        return face_images
    
    def crop_image(self, image, *, correct_roll = True):
        face_images = []

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detected_faces = self.face_detector.process(imgRGB).detections

        if detected_faces is None:
            return face_images

        for face in detected_faces:
            
            if 0 <= face.location_data.relative_bounding_box.xmin <= 1 and 0 <= face.location_data.relative_bounding_box.ymin <= 1:

                inflation_factor = self._get_bounding_box_inflation_factor(face.location_data.relative_keypoints[:2])
                inflated_face_image = self._get_inflated_face_image(imgRGB, face.location_data.relative_bounding_box, inflation_factor)
                detected_landmarks = self.landmark_detector.process(inflated_face_image).multi_face_landmarks

                if detected_landmarks is not None:
                    face_landmarks = detected_landmarks[0].landmark

                    if correct_roll:
                        inflated_face_image, face_landmarks = self._get_roll_corrected_image_and_landmarks(inflated_face_image, face_landmarks)
                    else:
                        face_landmarks = np.ndarray.astype(np.rint(np.array([
                            np.multiply([landmark.x for landmark in face_landmarks], inflated_face_image.shape[1]),
                            np.multiply([landmark.y for landmark in face_landmarks], inflated_face_image.shape[0])])), np.int64)

                    face_images.append(
                        self._crop_within_bounds(
                            inflated_face_image,
                            face_landmarks[1, np.argmin(face_landmarks[1, :])], face_landmarks[1, np.argmax(face_landmarks[1, :])],
                            face_landmarks[0, np.argmin(face_landmarks[0, :])], face_landmarks[0, np.argmax(face_landmarks[0, :])]
                        )
                    )
        return face_images