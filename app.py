import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# --- Load the Keras model ---
model = load_model("faceMaskModel.h5")

# --- Title ---
st.title("😷 Face Mask Detection App")
st.write("Upload an image and click the button to check if the person is wearing a mask.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Placeholder for result ---
prediction_result = st.empty()

# --- Prediction Logic ---
if uploaded_file is not None:
    # Read image only after prediction
    img = Image.open(uploaded_file)
    img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    input_image = np.reshape(img_array, (1, 128, 128, 3))

    # --- Button to trigger prediction ---
    if st.button("Predict"):
        # Show image *after* prediction
        st.image(uploaded_file, caption="Uploaded Image", width=250)

        prediction = model.predict(input_image)
        pred_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # --- Show result ---
        if pred_label == 1:
            prediction_result.success(f"✅ The person **is wearing** a mask.\n\n🟢 Confidence: {confidence:.2f}%")
        else:
            prediction_result.error(f"❌ The person **is NOT wearing** a mask.\n\n🔴 Confidence: {confidence:.2f}%")
