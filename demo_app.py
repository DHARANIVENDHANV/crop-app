import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(32, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(4, 4),

    nn.Flatten(),
    nn.Linear(29*29*64, 512),
    nn.ReLU(),
    nn.Linear(512, 38)
).to(device)

# Load model weights with CPU mapping
PATH = 'C:/Users/User/Desktop/vscode_jupyter/crop_disease_detection_model.pth'
model.load_state_dict(torch.load(PATH, map_location=device))

# Define class names
classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Streamlit app
st.title("Crop Disease Detector")

uploaded_file = st.file_uploader("Upload an image of a crop leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    crop_image = Image.open(uploaded_file).convert('RGB')
    st.image(crop_image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Process Image"):
        # Define the image preprocessing transform
        preprocess = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Preprocess the image
        input_image = preprocess(crop_image).unsqueeze(0).to(device)
        
        # Perform inference
        model.eval()
        with torch.no_grad():
            outputs = model(input_image)
            _, predicted = torch.max(outputs, 1)
        
        # Get the predicted class
        remarks = classes[predicted.item()]
        st.write('Result:', remarks)
