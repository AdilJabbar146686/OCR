import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import tempfile
import os

def extract_text(image: Image.Image) -> str:
    """Extract text from image using Tesseract OCR."""
    return pytesseract.image_to_string(image).strip()

def text_to_speech(text: str) -> str:
    """Convert text to speech and save as a WAV file."""
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        temp_filename = tmp_file.name
    engine.save_to_file(text, temp_filename)
    engine.runAndWait()
    return temp_filename

# --- Streamlit UI ---

st.set_page_config(page_title="OCR + TTS App", page_icon="🗣️")
st.title("🧠 Image-to-Text + Text-to-Speech")

uploaded_file = st.file_uploader("📤 Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("🔍 Extracting text..."):
        image = Image.open(uploaded_file)
        extracted_text = extract_text(image)

    if extracted_text:
        st.success("✅ Text Extracted!")
        st.text_area("📝 Extracted Text", extracted_text, height=200)

        if st.button("🔊 Convert to Speech"):
            with st.spinner("🎙️ Generating audio..."):
                audio_path = text_to_speech(extracted_text)

            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.download_button(label="⬇️ Download Audio", data=audio_bytes, file_name="speech.wav", mime="audio/wav")

            os.remove(audio_path)
    else:
        st.warning("⚠️ No text detected in the image.")
else:
    st.info("Upload an image to get started.")
