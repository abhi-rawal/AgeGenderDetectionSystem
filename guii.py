# Importing Necessary Libraries
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from keras.models import load_model
from keras.metrics import MeanAbsoluteError

# Load the Model and Handle Custom Metrics
custom_objects = {"mae": MeanAbsoluteError()}
model = load_model('Age_Sex_Detection.h5', custom_objects=custom_objects)

# Function to predict age and gender
def predict_age_gender(face_img):
    face_img = cv2.resize(face_img, (48, 48))  # Resize to match model input size
    face_img = np.array(face_img).astype('float32') / 255.0  # Normalize
    face_img = np.expand_dims(face_img, axis=0)
    
    prediction = model.predict(face_img)
    age = int(np.round(prediction[1][0]))  # Age prediction
    gender = "Male" if int(np.round(prediction[0][0])) == 0 else "Female"
    return age, gender

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

# Create a Label widget for displaying the video feed
video_label = Label(top)
video_label.pack(pady=20)

# Variable to control the video loop
running = False

# Data storage for logging
data = []

# Function to start real-time detection
def start_detection():
    global running
    running = True
    cap = cv2.VideoCapture(0)
    
    def update_frame():
        if not running:
            cap.release()
            return
        
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                age, gender = predict_age_gender(face_img)
                
                color = (0, 255, 0) if age > 60 else (255, 0, 0)  # Green for senior citizens, Blue otherwise
                label = f"{gender}, Age: {age}"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Log data if age > 60
                if age > 60:
                    data.append({
                        'Age': age,
                        'Gender': gender,
                        'Time of Visit': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Convert the frame to RGB and display it in the Tkinter window
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        
        video_label.after(10, update_frame)
    
    update_frame()

# Function to stop detection and save data to CSV
def stop_detection():
    global running
    running = False
    if data:
        df = pd.DataFrame(data)
        df.to_csv('senior_citizens_log.csv', index=False)
        print("Data saved to senior_citizens_log.csv")

# Create Start and Stop Detection Buttons
start_btn = Button(top, text="Start Detection", command=start_detection, padx=10, pady=5)
start_btn.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
start_btn.pack(side='left', padx=20, pady=20)

stop_btn = Button(top, text="Stop Detection", command=stop_detection, padx=10, pady=5)
stop_btn.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
stop_btn.pack(side='right', padx=20, pady=20)

# Add Heading to the Window
heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Run the Application
top.mainloop()
