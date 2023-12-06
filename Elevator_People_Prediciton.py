import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image

def process_image(uploaded_file):
    # Convert the uploaded image to a cv2-style image
    pil_image = Image.open(uploaded_file)
    cv2_image = np.array(pil_image)
    return cv2_image

def convert_to_streamlit_format(cv2_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return rgb_image

st.title("Elevator People Prediction")
model = YOLO('yolov8m.pt')
detection1 = 0
detection2 = 0

uploaded_file = st.file_uploader("Upload the image for the inside of the elevator ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = process_image(uploaded_file)    
    result = model(image)
    
    result = list(result)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.3)]
    box_annotator = sv.BoxAnnotator()
    image = box_annotator.annotate(scene=image, detections=detections)
    image = convert_to_streamlit_format(image)
    st.write("Output Image:")
    st.image(image)
    detection1 = len(detections)
    st.write("There are", detection1, "people in the elevator")

uploaded_file = st.file_uploader("Upload an image for the people waiting...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = process_image(uploaded_file)    
    result = model(image)
    
    result = list(result)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.3)]
    box_annotator = sv.BoxAnnotator()
    image = box_annotator.annotate(scene=image, detections=detections)
    image = convert_to_streamlit_format(image)
    st.write("Output Image:")
    st.image(image)
    detection2 = len(detections)
    st.write("There are", detection2, "people waiting")

maxPeople = st.number_input("Enter elevator maximum people carrying capacity", step=1, value=0, format="%d")

if(detection1 < maxPeople and detection2 > 0):
    st.write("Elevator make a stop")
elif(detection1 == 0 and detection2 == 0):
    st.write("")
else:
    st.write("Elevator keep going")
