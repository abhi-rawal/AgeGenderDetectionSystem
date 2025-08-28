# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras.metrics import MeanAbsoluteError  # Use built-in metric

# Initialize the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

# Initialize Labels
label1 = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
label2 = Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = Label(top)

# Load the Model
try:
    model = load_model('Age_Sex_Detection.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    top.destroy()  # Close the GUI if the model fails to load

# Detect Function
def Detect(file_path):
    try:
        # Open and preprocess the image
        image = Image.open(file_path)
        image = image.resize((48, 48))  # Resize to match model input size
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.array(image)  # Convert to numpy array
        image = np.delete(image, 0, 1)  # Remove unnecessary dimension
        image = np.resize(image, (48, 48, 3))  # Reshape to (48, 48, 3)
        print("Processed Image Shape:", image.shape)

        # Normalize the image
        image = np.array([image]) / 255.0

        # Predict age and gender
        pred = model.predict(image)
        age = int(np.round(pred[1][0]))  # Predicted age
        sex = int(np.round(pred[0][0]))  # Predicted gender (0: Male, 1: Female)
        sex_f = ["Male", "Female"]

        # Display results
        print("Predicted Age is " + str(age))
        print("Predicted Gender is " + sex_f[sex])

        label1.configure(foreground="#011638", text=f"Age: {age}")
        label2.configure(foreground="#011638", text=f"Gender: {sex_f[sex]}")

    except Exception as e:
        print(f"Error during detection: {e}")

# Show Detect Button Function
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

# Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()  # Open file dialog
        uploaded = Image.open(file_path)  # Open the uploaded image
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))  # Resize for display
        im = ImageTk.PhotoImage(uploaded)  # Convert to PhotoImage
        sign_image.configure(image=im)  # Display the image
        sign_image.image = im  # Keep a reference to avoid garbage collection

        # Clear previous results
        label1.configure(text='')
        label2.configure(text='')

        # Show the Detect button
        show_Detect_button(file_path)

    except Exception as e:
        print(f"Error uploading image: {e}")

# Upload Button
upload = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

# Image and Labels Placement
sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

# Heading
heading = Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

# Run ```python
# Run the main loop
top.mainloop()