import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
# streamlit run Elevator_People_Prediciton.py

def process_image(uploaded_file):
    # Convert the uploaded image to a cv2-style image
    pil_image = Image.open(uploaded_file)
    cv2_image = np.array(pil_image)
    return cv2_image

st.title("Elevator People Prediction")
model = YOLO('yolov8m.pt')
detection1 = 0
detection2 = 0

uploaded_file = st.file_uploader("Upload the image for the inside of the elevator ...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = process_image(uploaded_file)
    result = model.predict(source = image, classes = [0], conf = 0.3)
    cls = result[0].boxes.cls
    detection1 = cls.tolist().count(0)
    image_result = result[0].plot()
    st.write("Output Image:")
    st.image(image_result)
    st.write("There are", detection1, "people in the elevator")

uploaded_file = st.file_uploader("Upload an image for the people waiting...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = process_image(uploaded_file)    
    result = model.predict(source = image, classes = [0], conf = 0.3)
    cls = result[0].boxes.cls
    detection2 = cls.tolist().count(0)
    image_result = result[0].plot()
    st.write("Output Image:")
    st.image(image_result)
    st.write("There are", detection2, "people waiting")

maxPeople = st.number_input("Enter elevator maximum people carrying capacity", step=1, value=0, format="%d")

if(detection1 < maxPeople and detection2 > 0):
    st.write("Elevator make a stop")
elif(detection1 == 0 and detection2 == 0):
    st.write("")
else:
    st.write("Elevator keep going")