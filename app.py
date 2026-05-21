import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ----------------------- Model Definition ------------------------
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ----------------------- Load Model ------------------------
model = BrainTumorCNN()
model.load_state_dict(torch.load('brain_tumor_model.pth', map_location=torch.device('cpu')))
model.eval()

# ----------------------- App Config ------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title(" Brain Tumor MRI Image Classifier")
st.markdown("Upload an MRI image and this model will predict the tumor type with high confid    ence.")

# ----------------------- Class Labels ------------------------
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
from torchvision import datasets

dataset = datasets.ImageFolder("C:/Users/Dell/Documents/Internship_project/Brain Tumor MRI Image Classification/train/train")
class_names = dataset.classes  # auto-detects correct order


# ----------------------- File Upload ------------------------
uploaded_file = st.file_uploader(" Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]
        _, prediction = torch.max(output, 1)
        predicted_class = class_names[prediction.item()]
        confidence = probs[prediction.item()]

    #  Show prediction
    st.success(f" **Predicted Tumor Type:** {predicted_class}")
    st.info(f" **Model Confidence:** {confidence * 100:.2f}%")

    #  Bar Chart of Probabilities
    fig, ax = plt.subplots()
    ax.bar(class_names, probs, color='skyblue')
    ax.set_ylabel("Confidence")
    ax.set_title("Model Confidence per Class")
    st.pyplot(fig)

    # Class Probabilities List
    st.subheader(" Detailed Class Probabilities")
    for i, prob in enumerate(probs):
        st.write(f" {class_names[i]}: **{prob * 100:.2f}%**")

# ----------------------- Model Info ------------------------
st.markdown("---")
st.subheader(" Model Information")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", "80%")
col2.metric("Total Classes", "4")
col3.metric("Trained Epochs", "20")

with st.expander("About This Model"):
    st.markdown("""
    - This model classifies brain MRI images into **4 tumor categories**:
      - Glioma
      - Meningioma
      - Pituitary
      - No Tumor
    - Built with **PyTorch CNN** and trained on a real brain MRI dataset.

    """)

# Feedback Section
feedback = st.text_input(" Was the prediction helpful?")
if feedback:
    st.success("Thank you for your feedback!")
