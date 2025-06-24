import streamlit as st
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
from paddleocr import PaddleOCR
import tempfile

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

ocr = load_ocr()

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

def extract_text(image: np.ndarray) -> str:
    result = ocr.ocr(image, cls=True)
    if result and isinstance(result[0], list):
        lines = [line[1][0] for line in result[0]]
        return " ".join(lines)
    return ""

def text_to_speech(text: str) -> bytes:
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)
        return tmp_file.read()

# Streamlit UI
st.set_page_config(page_title="SpeakText OCR", layout="centered")

st.markdown("<h1 style='text-align: center;'>📷 SpeakText</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Capture text with your camera. It will be read out automatically.</p>", unsafe_allow_html=True)

camera_image = st.camera_input("📸 Tap to Take Picture")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    with st.spinner("🔍 Extracting text..."):
        preprocessed = preprocess_image(image)
        text = extract_text(preprocessed)

    if text:
        st.success("✅ Text detected and spoken!")
        st.text_area("📝 Extracted Text", text, height=200)

        audio = text_to_speech(text)
        st.audio(audio, format="audio/mp3")
        st.download_button("⬇️ Download Audio", data=audio, file_name="speech.mp3", mime="audio/mpeg")
    else:
        st.error("❌ No readable text found.")
else:
    st.info("Use the camera above to scan your document.")

# Mobile styling
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
