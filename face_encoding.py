# face encoding 
# reducing image of face to measurements of features 
# best features of face measurements --> eyes?? nose?? . . 
# deep metric learning --> decides which measurements to consider 
# IMG_1779.JPG

import PIL.Image
import PIL.ImageDraw
import face_recognition

file = "/Users/digitallync/Desktop/face_recognition/testdata/unknown_4.JPG"
image = face_recognition.load_image_file(file)
f_encodings = face_recognition.face_encodings(image)

if len(f_encodings) == 0 :
    print("face not detected")
else:
    first_face_encodings = f_encodings[0]
    print(len(first_face_encodings)) # 128 length array
    print(first_face_encodings) # encoded face features array

