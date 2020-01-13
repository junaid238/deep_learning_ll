import PIL.Image
import PIL.ImageDraw
import face_recognition

# load images 

image1 = face_recognition.load_image_file("/Users/digitallync/Desktop/face_recognition/person_1.jpg")
image2 = face_recognition.load_image_file("/Users/digitallync/Desktop/face_recognition/person_2.jpg")
image3 = face_recognition.load_image_file("/Users/digitallync/Desktop/face_recognition/person_3.jpg")
image4 = face_recognition.load_image_file("/Users/digitallync/Desktop/face_recognition/IMG_1779.JPG")

# encode the images loaded 
f_encodings1 = face_recognition.face_encodings(image1)[0]
f_encodings2 = face_recognition.face_encodings(image2)[0]
f_encodings3 = face_recognition.face_encodings(image3)[0]
f_encodings4 = face_recognition.face_encodings(image4)[0]

# make list of all encoded images 
known_f_encodings = [
    f_encodings1,
    f_encodings2,
    f_encodings3,
    f_encodings4
]

# load unknown image
unknown = face_recognition.load_image_file("/Users/digitallync/Desktop/face_recognition/unknown_2.jpg")

# encode unknown image
f_un_encodings = face_recognition.face_encodings(unknown)

# compare all faces in unknown encoded array
for un_f_encoding in f_un_encodings:
    # comparing with all encoded images vs unknown image encode
    results = face_recognition.compare_faces(known_f_encodings,un_f_encoding, tolerance=0.6)
    name = "Unknown"
    if results[0]:
        name = "Person1"
    elif results[1]:
        name = "Person2"
    elif results[2]:
        name = "Person3"
    elif results[3]:
        name = "khan"
    print(f"Found {name} in the photo!")
