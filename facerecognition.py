import PIL.Image
import PIL.ImageDraw
import face_recognition

# detection of face using face_recognition module
image = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1760.JPG")
face_locations = face_recognition.face_locations(image)
no_of_faces = len(face_locations)
print("total faces detected are %d"%(no_of_faces))

pil_image = PIL.Image.fromarray(image)
for face_location in face_locations:
    top,right,bottom,left = face_location
    draw = PIL.ImageDraw.Draw(pil_image)
    draw.rectangle([top,right,bottom,left] , outline = "red")
pil_image.show()


# face landmark estimation -> 68 landmarks 
# point of various parts of face , example -->nose , eyes , eyebrows . . 
# face alignment --> landmark estimation 
# face should be straight 

image = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1760.JPG")
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


# In[27]:


# face encoding 
# reducing image of face to measurements of features 
# best features of face measurements --> eyes?? nose?? . . 
# deep metric learning --> decides which measurements to consider 
# IMG_1779.JPG
image = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1779.JPG")
f_encodings = face_recognition.face_encodings(image)
if len(f_encodings) == 0 :
    print("face not detected")
else:
    first_face_encodings = f_encodings[0]
    print(first_face_encodings)


# In[38]:


image = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1773.JPG")
f_encodings = face_recognition.face_encodings(image)
if len(f_encodings) == 0 :
    print("face not detected")
else:
    first_face_encodings = f_encodings[0]
    print(first_face_encodings)


# In[39]:


image1 = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1779.JPG")
f_encodings1 = face_recognition.face_encodings(image1)
if len(f_encodings1) == 0 :
    print("face not detected")
else:
    first_face_encodings1 = f_encodings1[0]
#     print(first_face_encodings1)

image2 = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1773.JPG")
f_encodings2 = face_recognition.face_encodings(image2)
if len(f_encodings2) == 0 :
    print("face not detected")
else:
    first_face_encodings2 = f_encodings2[0]
#     print(first_face_encodings2)
# print((first_face_encodings1 == first_face_encodings2))


# In[45]:


image1 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_1.jpg")
image2 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_2.jpg")
image3 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_3.jpg")

f_encodings1 = face_recognition.face_encodings(image1)[0]
f_encodings2 = face_recognition.face_encodings(image2)[0]
f_encodings3 = face_recognition.face_encodings(image3)[0]

known_f_encodings = [
    f_encodings1,
    f_encodings2,
    f_encodings3   
]

unknown1 = face_recognition.load_image_file("/Users/digitallync/Desktop/unknown_4.jpg")
f_un_encodings = face_recognition.face_encodings(unknown1)[0]

for un_f_encoding in f_un_encodings:
    results = face_recognition.compare_faces(known_f_encodings,f_un_encodings)
    name = "Unknown"
    if results[0]:
        name = "Person1"
    elif results[1]:
        name = "Person2"
    elif results[2]:
        name = "Person3"

print(name)


# In[64]:


# unknown image has multiple people 

image1 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_1.jpg")
image2 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_2.jpg")
image3 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_3.jpg")
image4 = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1779.JPG")

f_encodings1 = face_recognition.face_encodings(image1)[0]
f_encodings2 = face_recognition.face_encodings(image2)[0]
f_encodings3 = face_recognition.face_encodings(image3)[0]
f_encodings4 = face_recognition.face_encodings(image4)[0]

known_f_encodings = [
    f_encodings1,
    f_encodings2,
    f_encodings3,
    f_encodings4
]

unknown = face_recognition.load_image_file("/Users/digitallync/Desktop/unknown_2.jpg")
f_un_encodings = face_recognition.face_encodings(unknown)
for un_f_encoding in f_un_encodings:
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


# In[66]:


# tuning face recog system 

image1 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_1.jpg")
image2 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_2.jpg")
image3 = face_recognition.load_image_file("/Users/digitallync/Desktop/person_3.jpg")
image4 = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1779.JPG")

f_encodings1 = face_recognition.face_encodings(image1)[0]
f_encodings2 = face_recognition.face_encodings(image2)[0]
f_encodings3 = face_recognition.face_encodings(image3)[0]
f_encodings4 = face_recognition.face_encodings(image4)[0]

known_f_encodings = [
    f_encodings1,
    f_encodings2,
    f_encodings3,
    f_encodings4
]

unknown = face_recognition.load_image_file("/Users/digitallync/Desktop/unknown_7.jpg")
face_locations = face_recognition.face_locations(unknown,number_of_times_to_upsample=2)
f_un_encodings = face_recognition.face_encodings(unknown,known_face_locations=face_locations)

for un_f_encoding in f_un_encodings:
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


# In[77]:


# face make up --> digital 
from PIL import Image
from PIL import ImageDraw

image = face_recognition.load_image_file("/Users/digitallync/Desktop/IMG_1779.JPG")
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image,"RGBA")
for face_landmarks in face_landmarks_list:
    d.line(face_landmarks["left_eyebrow"],fill="purple",width=3)
    d.line(face_landmarks["right_eyebrow"],fill="purple",width=3)
    d.polygon(face_landmarks["top_lip"],fill="purple")
    d.polygon(face_landmarks["bottom_lip"],fill="purple")
pil_image.show()
    


# In[79]:


from PIL import Image
from PIL import ImageDraw
import face_recognition
import os


# In[87]:


mani = face_recognition.load_image_file("/Users/digitallync/Desktop/31122019/100CANON_dump_1/traindata/mani.JPG")
khan = face_recognition.load_image_file("/Users/digitallync/Desktop/31122019/100CANON_dump_1/traindata/khan.JPG")
pavan = face_recognition.load_image_file("/Users/digitallync/Desktop/31122019/100CANON_dump_1/traindata/pavan.JPG")
radha = face_recognition.load_image_file("/Users/digitallync/Desktop/31122019/100CANON_dump_1/traindata/radha.JPG")

mani_encodings = face_recognition.face_encodings(mani)[0]
khan_encodings = face_recognition.face_encodings(khan)[0]
pavan_encodings = face_recognition.face_encodings(pavan)[0]
radha_encodings = face_recognition.face_encodings(radha)[0]

encodings_list = [mani_encodings,khan_encodings,pavan_encodings,radha_encodings]
path = "/Users/digitallync/Desktop/31122019/100CANON_dump_1/testdata/"
for file in os.listdir(path):
    if not file.startswith("."):
#         print(path+file)
        unknown = face_recognition.load_image_file(path+file)
        unknown_encodings = face_recognition.face_encodings(unknown)
        
        
        for each_encoding in unknown_encodings:
            results = face_recognition.compare_faces(encodings_list,each_encoding,tolerance=0.6)
            name = "unknown"
            if results[0]:
                name = "mani"
            elif results[1]:
                name = "khan"
            elif results[2]:
                name = "pavan"
            elif results[3]:
                name = "radha"
            print(f"Found {name} in the "+file)
