# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 06:31:12 2021

@author: sacho

inspired by
https://youtu.be/Q1NC3NbmVlc
"""

####IMPORT MODULES
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL
import tensorflow as tf
import cv2
import dlib
import PIL
from PIL import Image, ImageOps
import os
import base64
import imutils
from imutils import face_utils
import argparse
import base64
import pathlib
from PIL import ImageFont, ImageDraw

from keras import metrics
from keras.metrics import categorical_accuracy
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical
import h5py
from keras.models import load_model


#writefile app.py
####################################################################################
### Some functions needed for face dection and inputs
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


def create_image(liste):
    img = np.zeros((48,48),dtype = 'uint8')
    liste = liste[17:]
    for point in liste:
        x = point[0]
        y = point[1]
        if x < 0:
          x = 0
        if x > 47 :
          x= 47
        if y >47 :
          y = 47
        if y <0:
          y=0
        img[y][x] = 1
    #plt.imshow(img, cpam ="gray")
    imList = img.reshape(48*48)
    #print(imList)
    return imList

##################################################################################
#Streamlit application build
#
#inspired by https://youtu.be/Q1NC3NbmVlc 
###################################################################################

st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_models():
    model = load_model(r"test_3_1601_50epoch.h5")
    return model


banner = Image.open(r"banner.JPG")
st.image(banner,use_column_width = True)

st.markdown("""  
         ***
         ### **Welcome to our application**\n
         This application has been created by `Camille Benoit` and `Alexandra Giraud` \n
         Facial emotion recognition can have multiple useful applications\n
         We try to implement it in order to improve road safety
         ***
         """
         )
st.write(""" 
         **Please select an image of a person with a face well visible** \n
         Our app will give you the probable emotion that the person is feeling\n
         On the basis of 6 different emotions : `hapiness`, `sadness`, `suprise`, `fear`, `anger` and `neutral` \n
         Have **fun**
         """
         )
main_bg = r"background2.jpg"
main_bg_ext = "jpg"

side_bg = r"neuralN.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

################################


file = st.file_uploader("Please uplad an image of a face", type = ['jpg','png'])

#this function is just for display/ aesthetic purposes in the webApp
def display_image():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
    shape = None
    image_file = Image.open(file)
    #image = cv2.imread(file)
    image = np.array(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        cv2.putText(image, 'Emotion Recognition in progess', (x - 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            #face_crop = image[(y-h):(y+h+h//2),(x-w):(x+w+w//2)]
            face_crop = image[(y-h):(y+h),(x-w):(x+w)]
            #face_crop = image[(y):(y+h),x:(x+w)]
        #numberpixels = image.size
        #we are going to discrinate wether we are going to display the crop or the entire image 
    st.image(face_crop,use_column_width = True)


def display_emotion(texte):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
    shape = None
    image_file = Image.open(file)
    #image = cv2.imread(file)
    image = np.array(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        x, y, w, h = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (150, 0, 15), 2)
        # show the face number
        cv2.putText(image, texte, (x, y+25),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
            face_crop = image[(y-h):(y+h),(x-w):(x+w)]
    st.image(face_crop,use_column_width = True)


def inputforDL():#,model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
    shape = None
    image_file = Image.open(file)#le file est déjà présent dans le bail
    image = np.array(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        x, y, w, h = rect_to_bb(rect)
        #we have detected the face now we crop the image around it 
        #we create a temporary image with only the face, open it 
        #and then resize and redetect the landmarks 
        gray_crop = gray[y:y+h,x:x+w]
        cv2.imwrite(r"temp.jpg",gray_crop)
        cropped_file = Image.open(r"temp.jpg")
        cropped = cropped_file.resize((48,48),Image.ANTIALIAS)
        cropp = np.array(cropped)
        rects2 = detector(cropp, 1)
        for (j, rect2) in enumerate(rects2):
            shape = predictor(cropp, rect2)
            shape = shape_to_np(shape)
    landmarks = shape
    binary_img = create_image(landmarks)
    inputForDL = np.reshape(binary_img,(1,48,48,1))
    return inputForDL

mapper = {
    0: "Anger",
    1: "Neutral",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Suprise"
    }

def main():
    if file is None:
        st.text('Please upload an image')
    else:
        inputt = inputforDL()
        model = load_models()
        predictions = model.predict_classes(inputt,verbose = 0)
        pred_name = mapper[predictions[0]]
        image = Image.open(file)
        st.image(image,use_column_width = True)
        display_image()
        class_names=['neutral','happy','sad','suprised','angry','fear']
        string="The emotion most likely displayed in this image is: " + pred_name
        st.text(string)
        display_emotion(pred_name)
        
if __name__ == "__main__":
    main()
