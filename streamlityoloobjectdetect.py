import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title(" YOLOv8 Object Detection App")

# Sidebar options
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select YOLOv8 model:",
    ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt")
)

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_choice)
st.sidebar.success(f"Loaded model: {model_choice}")

# Choose input type
source_type = st.sidebar.radio("Select input type:", ["Image", "Video"])

# --- IMAGE INPUT ---
if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting..."):
                results = model.predict(image, conf=0.5)
                result_image = results[0].plot()  # Draw bounding boxes
                st.image(result_image, caption="Detection Results", use_container_width=True)

# --- VIDEO INPUT ---
elif source_type == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(tfile.name)

        if st.button("🎥 Run Detection"):
            with st.spinner("Processing video..."):
                cap = cv2.VideoCapture(tfile.name)
                output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = None

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(frame, conf=0.5, verbose=False)
                    annotated_frame = results[0].plot()

                    if out is None:
                        h, w, _ = annotated_frame.shape
                        out = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

                    out.write(annotated_frame)

                cap.release()
                out.release()

                st.success("✅ Detection complete!")
                st.video(output_path)
