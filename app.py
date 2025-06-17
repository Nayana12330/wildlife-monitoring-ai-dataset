import streamlit as st
from PIL import Image
import joblib
from feature_extraction import extract_features

st.title("🐾 AI-based Wildlife Monitoring System")

# Load model and label encoder
try:
    model = joblib.load('model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    try:
        features = extract_features(image).reshape(1, -1)
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"✅ Predicted Animal: **{predicted_label}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
