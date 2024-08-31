import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# Model path
model_path = "yolov8n-face.pt"

# Check if model file exists
if not os.path.exists(model_path):
    st.error(f"Model file {model_path} not found.")
else:
    try:
        model = YOLO(model_path) 
        
        st.title("Real-Time Face Detection in Video")

        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            video_path = uploaded_video.name
            with open(video_path, 'wb') as f:
                f.write(uploaded_video.getbuffer())
            
            video_cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break

                results = model(frame)
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        
                        if confidence > 0.5:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Face: {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                stframe.image(img, caption="Detected faces", use_column_width=True)

            video_cap.release()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
