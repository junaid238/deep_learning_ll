# face landmark estimation -> 68 landmarks 
# point of various parts of face , example -->nose , eyes , eyebrows . . 
# face alignment --> landmark estimation 
# face should be straight 

import PIL.Image
import PIL.ImageDraw
import face_recognition

file = "/Users/digitallync/Desktop/face_recognition/testdata/unknown_4.JPG"
image = face_recognition.load_image_file(file)
# features of face as array of points 
# eyebrows , eyes , nose and lips
face_landmarks_list = face_recognition.face_landmarks(image)
no_of_faces = len(face_landmarks_list)
print("total faces - %d"%(no_of_faces))

pil_image = PIL.Image.fromarray(image)
draw = PIL.ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    for name , list_of_points in face_landmarks.items():
        draw.line(list_of_points,fill="red",width = 2)
#     print(face_landmarks)
pil_image.show()
