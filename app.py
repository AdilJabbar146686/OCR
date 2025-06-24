import streamlit as st
from PIL import Image
import pytesseract
from gtts import gTTS
import tempfile
import base64

def extract_text(image: Image.Image) -> str:
    """Extract text from image using Tesseract OCR."""
    return pytesseract.image_to_string(image).strip()

def text_to_speech(text: str) -> bytes:
    """Convert text to speech and return audio bytes."""
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)
        audio_bytes = tmp_file.read()
    return audio_bytes

# --- Streamlit UI ---

st.set_page_config(page_title="OCR + TTS", page_icon="🧠")
st.title("📷 Image to Text + Text to Speech")

image_input = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

if image_input == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
else:
    uploaded_file = st.camera_input("📸 Take a photo")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("🔍 Extracting text..."):
        extracted_text = extract_text(image)

    if extracted_text:
        st.success("✅ Text Extracted:")
        st.text_area("📝 Extracted Text", extracted_text, height=200)

        if st.button("🎧 Convert to Speech"):
            with st.spinner("🔊 Generating Audio..."):
                audio_bytes = text_to_speech(extracted_text)

            st.audio(audio_bytes, format="audio/mp3")

            st.download_button(
                label="⬇️ Download Audio",
                data=audio_bytes,
                file_name="speech.mp3",
                mime="audio/mpeg"
            )
    else:
        st.warning("⚠️ No text detected in the image.")
else:
    st.info("Upload an image or take a photo to begin.")
