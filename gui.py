from tkinter import PhotoImage, messagebox
from customtkinter import *
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from PIL import Image

# Load model and scaler
model = load_model("my_model.h5")
scaler = joblib.load('scaler.pkl')

# Define all 30 feature names
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 
    'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 
    'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Create GUI
app = CTk()
app.geometry("600x800")
set_appearance_mode("dark")
app.title("Breast Cancer Prediction")

icon_image = PhotoImage(file='./assests/brain.png')
app.iconphoto(False, icon_image)

# Load an image for the GUI
my_image = CTkImage(light_image=Image.open("./assests/ml.jpg"),
                    dark_image=Image.open("./assests/ml.jpg"),
                    size=(600, 500))

# Global storage for input fields
entry_fields = {}

def predict():
    # Collect input data
    input_data = []
    for feature in feature_names:
        value = entry_fields[feature].get()
        try:
            input_data.append(float(value))  # Convert input to float
        except ValueError:
            messagebox.showerror("Input Error", f"Invalid input for {feature}. Please enter a numeric value.")
            return

    # Scale the data and predict
    input_data = np.array([input_data])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    prediction_label = np.argmax(prediction)

    # Interpret result
    result = 'Malignant' if prediction_label == 0 else 'Benign'
    messagebox.showinfo("Prediction Result", f"The tumor is: {result}")

def load_input_fields():
    # Clear previous content
    for widget in app.winfo_children():
        widget.destroy()

    # Title Label
    title_label = CTkLabel(app, text="Enter Feature Values for Prediction", font=("Arial", 20, "bold"))
    title_label.pack(pady=10)

    # Input Frame
    input_frame = CTkScrollableFrame(app, width=550, height=500)
    input_frame.pack(pady=10)

    # Dynamically create input fields
    global entry_fields
    entry_fields = {}  # Reset storage
    for feature in feature_names:
        frame = CTkFrame(input_frame)
        frame.pack(pady=5, anchor="w")
        label = CTkLabel(frame, text=feature + ":", font=("Arial", 14), width=250)
        label.pack(side="left", padx=5)
        entry = CTkEntry(frame, width=200)
        entry.pack(side="left", padx=5)
        entry_fields[feature] = entry  # Store entry field in the dictionary

    # Predict Button
    predict_btn = CTkButton(app, text="Predict", command=predict, corner_radius=32, 
                            fg_color="#1F6AA5", font=("Arial", 18, "bold"), height=40)
    predict_btn.pack(pady=20)

    # Back Button
    back_btn = CTkButton(app, text="Back", command=load_initial_content, corner_radius=32,
                         fg_color="transparent", hover_color="#4158D0", border_color="#4158D0",
                         border_width=2, font=("Arial", 16, "bold"))
    back_btn.pack()

def load_initial_content():
    # Clear existing content
    for widget in app.winfo_children():
        widget.destroy()

    image_label = CTkLabel(app, image=my_image, text="")
    image_label.image = my_image
    image_label.pack()

    # Single Predict Button
    predict_btn = CTkButton(app, text="Start Prediction", corner_radius=32, 
                            fg_color="#1F6AA5", font=("Arial", 20, "bold"),
                            command=load_input_fields)
    predict_btn.pack(pady=20)

# Initial content
load_initial_content()

app.mainloop()
