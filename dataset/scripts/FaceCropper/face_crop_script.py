from videoFaceCropper import VideoFaceCropper
import cv2
import sys

input_file = sys.argv[1]

output_file = 'output.mp4'

if len(sys.argv[1]) > 2:
    output_file = sys.argv[2]

vfc = VideoFaceCropper()

cap = cv2.VideoCapture(input_file)

output_size = (640, 480)

video= cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, output_size)

face_frames = []

while True:
    success, frame = cap.read()

    if not success:
        print('Frame not captured')
        break
    
    face_images = vfc.crop_image(frame)
    if face_images is None:
        print("No faces detected")
    else:
        for i, face in enumerate(face_images):
            cv2.imshow(f'Face {i}', cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    face_frames.append(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        

    if cv2.waitKey(5) & 0xFF == 27:
        break
for frame in face_frames:
    video.write(cv2.resize(frame, output_size ))
cap.release()
video.release()
cv2.destroyAllWindows()