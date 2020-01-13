import PIL.Image
import PIL.ImageDraw
import face_recognition

# detection of face using face_recognition module
# pass the file name with image below
file = "/Users/digitallync/Desktop/face_recognition/testdata/unknown_4.JPG"
image = face_recognition.load_image_file(file)
face_locations = face_recognition.face_locations(image)
no_of_faces = len(face_locations)
print("total faces detected are %d"%(no_of_faces))

pil_image = PIL.Image.fromarray(image)
for face_location in face_locations:
    top,right,bottom,left = face_location
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([top,right,bottom,left] , outline = "red")
pil_image.show()