from ultralytics import YOLO
import numpy
#load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")  

# predict on an image
detection_output = model.predict(source=r"C:\Users\panip\anaconda3\streamlit\phonegirl.jpg", conf=0.25, save=True)

# Display tensor array
print(detection_output)

