import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import cv2
from gtts import gTTS
import tempfile

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for OCR using OpenCV."""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize to enhance small text
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return gray

def extract_text_with_easyocr(image_array: np.ndarray) -> str:
    """Use EasyOCR to extract text from image."""
    results = reader.readtext(image_array, detail=0)
    return " ".join(results).strip()

def text_to_speech(text: str) -> bytes:
    """Convert text to speech and return audio bytes."""
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)
        return tmp_file.read()

# ---------------- UI ----------------

st.set_page_config(page_title="📷 SpeakText for the Blind", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧠 SpeakText</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Scan printed text with your camera — it will be read aloud automatically.</p>", unsafe_allow_html=True)

# Capture Image from Camera
camera_image = st.camera_input("📸 Tap below to take a picture", key="camera")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    with st.spinner("🧠 Reading text..."):
        preprocessed = preprocess_image(image)
        extracted_text = extract_text_with_easyocr(preprocessed)

    if extracted_text:
        st.success("✅ Text detected and spoken!")

        st.markdown("### 📝 Extracted Text")
        st.text_area("Extracted Text", extracted_text, height=200)

        audio_bytes = text_to_speech(extracted_text)

        st.audio(audio_bytes, format="audio/mp3")

        st.download_button(
            label="⬇️ Download Speech",
            data=audio_bytes,
            file_name="speech.mp3",
            mime="audio/mpeg",
            use_container_width=True
        )
    else:
        st.error("❌ No readable text found. Try taking the photo again in better lighting.")
else:
    st.info("Take a photo to begin.")

# --- Styling for mobile (larger UI) ---
st.markdown("""
    <style>
    .stButton > button {
        font-size: 20px !important;
        padding: 0.75em 2em;
    }
    .stTextArea textarea {
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)
