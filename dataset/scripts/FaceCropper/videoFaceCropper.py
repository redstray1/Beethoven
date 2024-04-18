import cv2
import numpy as np
import mediapipe as mp
import os
import cv2

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
        
        self._FACE_LANDMARKS_INDECES = [33, 263, 1, 61, 291, 199]
        self._YAW_THRESHOLD = 15
        self._PITCH_THRESHOLD = 15

        self.face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=min_face_detector_confidence,
                                                                       model_selection=face_detector_model_selection)

        self.landmark_detector = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                                 static_image_mode=landmark_detector_static_image_mode,
                                                                 min_detection_confidence=min_landmark_detector_confidence)

    def get_face_region_factor(self, eye_coords, amplification=2, base_inflation=1):
        roll_angle = self._get_face_roll_angle([eye_coords[0].x, 1 - eye_coords[0].y], [eye_coords[1].x, 1 - eye_coords[1].y])
        region_factor = np.abs(roll_angle) / 90

        return base_inflation + (region_factor * amplification)
    
    def _get_face_region(self, image, face_box, inflation):

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
            [face_landmarks[lm] for lm in self._LEFT_EYE_LANDMARK_INDICES],
            [face_landmarks[lm] for lm in self._RIGHT_EYE_LANDMARK_INDICES]
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
            region_factor = self.get_face_region_factor(eyes)
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
            width_inflation, height_inflation = face_box.width * region_factor, face_box.height * region_factor
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
            return None

        for face in detected_faces:
            
            if 0 <= face.location_data.relative_bounding_box.xmin <= 1 and 0 <= face.location_data.relative_bounding_box.ymin <= 1:

                region_factor = self.get_face_region_factor(face.location_data.relative_keypoints[:2])
                face_region = self._get_face_region(imgRGB, face.location_data.relative_bounding_box, region_factor)
                detected_landmarks = self.landmark_detector.process(face_region).multi_face_landmarks

                if detected_landmarks is not None:
                    face_landmarks = detected_landmarks[0].landmark

                    if correct_roll:
                        face_region, face_landmarks = self._get_roll_corrected_image_and_landmarks(face_region, face_landmarks)
                    else:
                        face_landmarks = np.ndarray.astype(np.rint(np.array([
                            np.multiply([landmark.x for landmark in face_landmarks], face_region.shape[1]),
                            np.multiply([landmark.y for landmark in face_landmarks], face_region.shape[0])])), np.int64)

                    face_images.append(
                        self._crop_within_bounds(
                            face_region,
                            face_landmarks[1, np.argmin(face_landmarks[1, :])], face_landmarks[1, np.argmax(face_landmarks[1, :])],
                            face_landmarks[0, np.argmin(face_landmarks[0, :])], face_landmarks[0, np.argmax(face_landmarks[0, :])]
                        )
                    )
        return face_images
    
    def get_head_tilt(self, image):
        """
        Return:
        Forward = 0
        Up = 1
        Right = 2
        Down = 3
        Left = 4
        """
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.landmark_detector.process(imgRGB)

        img_h, img_w, img_c = imgRGB.shape
        face_2d = []
        face_3d = []
        head_tilt_type = -1
        if results.multi_face_landmarks:
            for id,lm in enumerate(results.multi_face_landmarks[0].landmark):
                if id in self._FACE_LANDMARKS_INDECES:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length_x = img_w
            focal_length_y = img_h

            focal_length = (focal_length_x + focal_length_y) / 2

            intrinsic_matrix = np.array([[focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])
            distort_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rotation_vector, translation_vector = cv2.solvePnP(face_3d, face_2d, intrinsic_matrix, distort_matrix)

            rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

            angles,R,Q,Qx,Qy,Qz = cv2.RQDecomp3x3(rotation_matrix)

            x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360
            
            head_tilt_type = 0

            if y < -self._YAW_THRESHOLD:
                text = 'Left'
                head_tilt_type = 4
            elif y > self._YAW_THRESHOLD:
                text = 'Right'
                head_tilt_type = 2
            elif x < -self._PITCH_THRESHOLD:
                text = 'Down'
                head_tilt_type = 3
            elif x > self._PITCH_THRESHOLD:
                text = 'Up'
                head_tilt_type = 1
            else:
                text = 'Forward'
                head_tilt_type = 0
        return head_tilt_type

    def crop_video(self, infilename, outfilename, *, correct_roll = True, bad_frames_threshold = 10, show=False):
        cap = cv2.VideoCapture(infilename)
        
        

        face_frames = []

        bad_frames = 0
        many_faces_frames = 0

        while True:
            success, frame = cap.read()

            if not success:
                print('Frame not captured')
                break
            
            face_images = self.crop_image(frame, correct_roll=correct_roll)
            head_tilt = self.get_head_tilt(frame)
            if (face_images is None) or (face_images == []) or ((face_images is not None) and len(face_images) > 1) or (head_tilt != 0):
                #print("No faces detected")
                if ((face_images is not None) and len(face_images) > 1):
                    many_faces_frames += 1
                else:
                    bad_frames += 1
            else:
                if show:
                    for i, face in enumerate(face_images):
                        cv2.imshow(f'Face {i}', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                face_frames.append(cv2.cvtColor(face_images[0], cv2.COLOR_RGB2BGR))
        cap.release()

        if (bad_frames > bad_frames_threshold) or (many_faces_frames > 0):
            return False
        
        output_size = (640, 480)
        video= cv2.VideoWriter(outfilename, cv2.VideoWriter_fourcc(*'mp4v'), 30, output_size)
        for frame in face_frames:
            video.write(cv2.resize(frame, output_size ))
        
        video.release()
        return True