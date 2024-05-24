import cv2
import os

#load image
image = cv2.imread('./Images/IMG-20220601-WA0022.jpg')

#load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

# Create a directory for saving detected faces
output_dir = 'Output/detected_faces'
os.makedirs(output_dir, exist_ok=True)

#extract saved faces
for i, (x, y, w, h) in enumerate(faces):
    face = image[y:y+h, x:x+w]
    face_filename = os.path.join(output_dir, f'face_{i}.jpg')
    cv2.imwrite(face_filename, face)

print(str(len(faces)) + " faces detected and saved in folder './Output/detected_faces")