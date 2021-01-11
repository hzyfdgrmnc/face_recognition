import face_recognition
from cv2 import cv2
import numpy as np
import os

for j in os.listdir('images'):
    name = j[:-4]
    img_path = 'images\\'+name+'.jpg'
    x_img = face_recognition.load_image_file(img_path)
    x_encode = face_recognition.face_encodings(x_img)[0]
    enc_name = 'encodings\\'+name+'_face_encoding.npy'
    np.save(enc_name,x_encode)


