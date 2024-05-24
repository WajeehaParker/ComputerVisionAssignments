import face_recognition
import cv2
import os

def recognizeAndMarkFaceInImage(face_file_name, image_file_name):

    # Load the known face image (the face you want to detect)
    face = face_recognition.load_image_file('./Output/detected_faces/'+face_file_name)
    image = cv2.imread('./Images/'+image_file_name)

    base_name_facefile, extension_facefile = os.path.splitext(face_file_name)
    base_name_image, extension_image = os.path.splitext(image_file_name)

    # Resize the known face image to match the dimensions of the search image
    face = cv2.resize(face, (image.shape[1], image.shape[0]))

    # Find face locations and encodings
    known_face_locations = face_recognition.face_locations(face)
    known_face_encodings = face_recognition.face_encodings(face, known_face_locations)

    search_face_locations = face_recognition.face_locations(image)
    search_face_encodings = face_recognition.face_encodings(image, search_face_locations)

    # Comparing faces
    for (top, right, bottom, left), search_face_encoding in zip(search_face_locations, search_face_encodings):
        # Compare the known face encodings to the search face encoding
        face_distances = face_recognition.face_distance(known_face_encodings, search_face_encoding)
        
        # Define a threshold for considering a match 
        threshold = 0.4

        if any(face_distance <= threshold for face_distance in face_distances):
            # Draw a rectangle around the detected face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Save the marked image with detected faces
            cv2.imwrite('./Output/' + base_name_facefile + '_' + base_name_image + '.' + extension_image, image)

            print(base_name_facefile +" detected in " + base_name_image)
        else:
            print(base_name_facefile +" not detected in " + base_name_image)



# Load the known face image
#face = face_recognition.load_image_file('./Output/detected_faces/face_0.jpg')
#search_image = cv2.imread('./images/IMG-20220327-WA0011.jpg')

for face_file_name in os.listdir('./Output/detected_faces/'):
    if face_file_name.endswith('.jpg'):
        for image_file_name in os.listdir('./Images/'):
            if image_file_name.endswith('.jpg'):
                recognizeAndMarkFaceInImage(face_file_name, image_file_name)
print("Images with marked faces are saved in Output folder")

