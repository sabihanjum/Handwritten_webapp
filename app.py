import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt

# Same CVAE class as above (include again)

# Load trained model
model = CVAE()
model.load_state_dict(torch.load("cvae_mnist.pth", map_location='cpu'))
model.eval()

def generate_images(digit, num=5):
    y = torch.eye(10)[digit].repeat(num, 1)
    z = torch.randn(num, 20)
    with torch.no_grad():
        imgs = model.decode(z, y).view(-1, 28, 28)
    return imgs

# Streamlit app
st.title("MNIST Digit Generator")
digit = st.selectbox("Select a digit (0–9)", list(range(10)))

if st.button("Generate Images"):
    images = generate_images(digit)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
