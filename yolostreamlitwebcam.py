import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import time

# -----------------------------------
# Streamlit Page Configuration
# -----------------------------------
st.set_page_config(page_title="YOLOv8 Webcam Object Detection", layout="wide")
st.title("🎥 YOLOv8 Real-Time Object Detection (Webcam)")

# -----------------------------------
# Sidebar Controls
# -----------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select YOLOv8 model",
    ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"),
    index=0
)

confidence = st.sidebar.slider("Confidence Threshold", 0.1,0.5, 0.05)

# -----------------------------------
# Load Model (cached)
# -----------------------------------
@st.cache_resource
def load_model(model_name):
    model = YOLO(model_name)
    return model

model = load_model(model_choice)
st.sidebar.success(f"✅ Model {model_choice} loaded successfully!")

# -----------------------------------
# Start Webcam
# -----------------------------------
run = st.checkbox("▶️ Start Webcam Detection")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Please check your camera permissions.")
    else:
        st.info("Press **Stop Webcam Detection** to end stream.")
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read frame from webcam.")
                break

            # YOLOv8 Inference
            results = model.predict(frame, conf=confidence, device='cpu', verbose=False)
            annotated_frame = results[0].plot()

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels="RGB")

            # Small sleep to control frame rate
            time.sleep(0.03)

        cap.release()
        st.success("✅ Webcam stopped.")
else:
    st.info("👈 Check the box to start webcam object detection.")
