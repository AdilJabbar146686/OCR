import streamlit as st
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
from paddleocr import PaddleOCR
from symspellpy import SymSpell, Verbosity
import tempfile
import pkg_resources
import re

# ----- Caching -----
@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

@st.cache_resource
def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    return sym_spell

ocr = load_ocr()
sym_spell = load_symspell()

# ----- Image Preprocessing -----
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Improve contrast with adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

# ----- OCR + Cleanup -----
def extract_text(image: np.ndarray) -> str:
    result = ocr.ocr(image, cls=True)
    if not result or not isinstance(result[0], list):
        return ""

    raw_text = " ".join([line[1][0] for line in result[0]])
    raw_text = re.sub(r"[^A-Za-z0-9.,;:!?()\"'’\s]", "", raw_text)

    suggestions = sym_spell.lookup_compound(raw_text, max_edit_distance=2)
    corrected = suggestions[0].term if suggestions else raw_text

    return corrected.strip().capitalize()

# ----- Text-to-Speech -----
def text_to_speech(text: str) -> bytes:
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)
        return tmp_file.read()

# ----- Streamlit Interface -----
st.set_page_config(page_title="SpeakText", layout="centered")

st.markdown("<h1 style='text-align: center;'>📷 SpeakText</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Take a picture and the app will read printed text aloud — designed for visually impaired users.</p>", unsafe_allow_html=True)

camera_image = st.camera_input("📸 Tap below to take a picture")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    with st.spinner("🧠 Processing and reading text..."):
        preprocessed = preprocess_image(image)
        extracted_text = extract_text(preprocessed)

    if extracted_text:
        st.success("✅ Text read successfully!")

        st.markdown("### 📝 Detected & Corrected Text")
        st.text_area("Output", extracted_text, height=250)

        audio = text_to_speech(extracted_text)
        st.audio(audio, format="audio/mp3")
        st.download_button("⬇️ Download Audio", data=audio, file_name="speech.mp3", mime="audio/mpeg", use_container_width=True)
    else:
        st.error("❌ No text found. Try better lighting or a clearer capture.")
else:
    st.info("Take a picture to get started.")

# ----- Styling for Mobile -----
st.markdown("""
    <style>
    .stButton > button {
        font-size: 22px !important;
        padding: 1em 2em;
    }
    .stTextArea textarea {
        font-size: 18px !important;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)
