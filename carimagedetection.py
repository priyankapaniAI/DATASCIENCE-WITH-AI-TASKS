import numpy as np
import cv2

# Load the Haar Cascade for face detection
car_classifier = cv2.CascadeClassifier(r"C:\Users\panip\anaconda3\streamlit\computervision\haarcascade_car.xml")
# Load the image
image = cv2.imread(r"C:\Users\panip\Downloads\car1.jpg")

#image = cv2.imread(r'C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\2. DATASCIENCE PROJECT\10. Computer vision\Computer-Vision-Tutorial-master\Computer-Vision-Tutorial-master\image_examples\5.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()  # Exit if image is not loaded
# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cars = car_classifier.detectMultiScale(gray, 1.1, 3)
for (x,y,w,h) in cars:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('Car Detector',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
