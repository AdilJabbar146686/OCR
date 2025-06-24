import streamlit as st
from PIL import Image
import pytesseract
from io import BytesIO
import sounddevice as sd
import numpy as np
from TTS.api import TTS

# Initialize TTS model (Glow-TTS pretrained)
@st.cache_resource
def load_tts_model():
    return TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

tts_model = load_tts_model()

def extract_text(image: Image.Image) -> str:
    """Extract text from a PIL image using Tesseract."""
    return pytesseract.image_to_string(image).strip()

def speak(text: str):
    """Convert text to speech and play audio."""
    if not text:
        st.warning("No text to speak.")
        return
    st.info("🔊 Speaking text...")
    audio_data = tts_model.tts(text)
    sd.play(audio_data, samplerate=22050)
    sd.wait()

# --- Streamlit UI ---

st.title("📘 Image-to-Text + Text-to-Speech App")

uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("🔍 Extracting text..."):
        image = Image.open(uploaded_file)
        extracted_text = extract_text(image)
    
    if extracted_text:
        st.success("✅ Text Extracted:")
        st.text_area("Extracted Text", extracted_text, height=200)

        if st.button("🔊 Speak"):
            speak(extracted_text)
    else:
        st.error("❌ No text detected in the image.")
else:
    st.info("Upload an image to begin.")
