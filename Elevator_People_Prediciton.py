import cv2
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import supervision as sv
from PIL import Image
# streamlit run Elevator_People_Prediciton.py

def process_image(uploaded_file):
    # Convert the uploaded image to a cv2-style image
    pil_image = Image.open(uploaded_file)
    cv2_image = np.array(pil_image)
    return cv2_image

def convert_to_streamlit_format(cv2_image):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return rgb_image

# print(dir(sv))
st.title("Elevator People Prediction")
model = YOLO('yolov8m.pt')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = process_image(uploaded_file)    
    result = model(image)
    
    result = list(result)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.2)]
    
    box_annotator = sv.BoxAnnotator()
    image = box_annotator.annotate(scene=image, detections=detections)
    image = convert_to_streamlit_format(image)
    st.write("Output Image:")
    st.image(image)
    st.write("There are", len(detections), "people in the elevator")
