import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Function to overlay passport image on selected credit card
def overlay_passport_on_card(card_image_path, passport_image, output_path="final_card.png"):
    # Load the credit card image
    card = cv2.imread(card_image_path)

    # Convert uploaded passport image to OpenCV format
    passport = np.array(passport_image.convert("RGB"))
    passport = cv2.cvtColor(passport, cv2.COLOR_RGB2BGR)

    # Resize the passport image to fit on the credit card (e.g., 110x90 pixels)
    passport_resized = cv2.resize(passport, (110, 90))

    # Define position on the credit card (Bottom right corner)
    x_offset, y_offset = card.shape[1] - 260, card.shape[0] - 100  # Adjust as needed
    h, w = passport_resized.shape[:2]

    # Overlay the passport image onto the card
    card[y_offset:y_offset+h, x_offset:x_offset+w] = passport_resized

    # Save final image
    cv2.imwrite(output_path, card)
    return output_path

# Streamlit UI
st.title("Custom Credit Card Generator")

# Select credit card pattern
card_folder = "card_patterns"  # Folder containing card pattern images
card_files = [f for f in os.listdir(card_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

selected_card = st.selectbox("Choose a credit card pattern:", card_files)

# Upload passport-size image
uploaded_passport = st.file_uploader("Upload your passport-size photo", type=["png", "jpg", "jpeg"])

if uploaded_passport and selected_card:
    # Load passport image using PIL
    passport_image = Image.open(uploaded_passport)

    # Generate final image
    card_path = os.path.join(card_folder, selected_card)
    final_image_path = overlay_passport_on_card(card_path, passport_image)

    # Display the final card
    st.image(final_image_path, caption="Final Credit Card", use_column_width=True)

    # Provide a download button
    with open(final_image_path, "rb") as file:
        st.download_button(label="Download Your Custom Card", data=file, file_name="custom_credit_card.png", mime="image/png")


