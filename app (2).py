import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Skin Lesion Classification App")
st.write("Upload an image of a skin lesion to get a prediction from our trained CNN model.")

# Load the saved model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_lesion_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Class names ordered logically
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_map = {
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

uploaded_file = st.file_uploader("Choose a skin lesion image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("Classifying...")
    
    # Preprocess image to match training: 28x28 RGB normalized to [0,1]
    resized_image = image.resize((28, 28)).convert('RGB')
    img_array = np.array(resized_image) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 28, 28, 3)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx] * 100
    
    st.subheader(f"Prediction: {class_map[predicted_class_name]} ({predicted_class_name})")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Display all probabilities
    st.write("### Class Probabilities:")
    for name, prob in zip(class_names, predictions[0]):
        st.write(f"- **{class_map[name]} ({name})**: {prob*100:.2f}%")
