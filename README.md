Face Mask Detection using CNN 🧠😷
This project is a deep learning-based face mask detection system that identifies whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) trained on a labeled dataset. The solution also includes a Streamlit-based frontend for real-time user interaction and prediction.

📦 Features
Uses CNN for binary classification: With Mask (1) vs Without Mask (0).

Dataset sourced from Kaggle and preprocessed using PIL and NumPy.

Trained using TensorFlow/Keras.

Visualizes training history (accuracy and loss).

Supports image prediction via uploaded images.

Integrated Streamlit UI for an interactive and simple web app.

🛠 Tech Stack
Python

TensorFlow / Keras

NumPy, OpenCV, Matplotlib

Streamlit (Frontend)

🚀 How to Run
Clone the repo:

git clone https://github.com/your-username/face-mask-detection.git
Install dependencies:
pip install -r requirements.txt
Launch the Streamlit app:
streamlit run app.py

📁 Model
Trained model saved as:
faceMaskModel.h5

📷 Sample Output
The model predicts:

✅ "Wearing a Mask"

❌ "Not Wearing a Mask"
