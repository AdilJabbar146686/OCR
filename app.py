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

# ----- Loaders -----

@st.cache_resource
def load_ocr():
    return PaddleOCR(use_angle_cls=True, lang='en')

@st.cache_resource
def load_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    return sym_spell

ocr = load_ocr()
sym_spell = load_symspell()

# ----- Functions -----

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2))
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

def extract_text(image: np.ndarray) -> str:
    result = ocr.ocr(image, cls=True)
    if not result or not isinstance(result[0], list):
        return ""

    # Join text lines
    raw_text = " ".join([line[1][0] for line in result[0]])

    # Clean up unwanted characters
    raw_text = re.sub(r"[^A-Za-z0-9.,;:!?()\"'’\s]", "", raw_text)

    # Correct spelling, spacing
    suggestions = sym_spell.lookup_compound(raw_text, max_edit_distance=2)
    corrected_text = suggestions[0].term if suggestions else raw_text

    return corrected_text.strip().capitalize()

def text_to_speech(text: str) -> bytes:
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)
        return tmp_file.read()

# ----- Streamlit UI -----

st.set_page_config(page_title="SpeakText", layout="centered")

st.markdown("<h1 style='text-align: center;'>📷 SpeakText</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Take a photo and the app will read printed text aloud — ideal for visually impaired users.</p>", unsafe_allow_html=True)

camera_image = st.camera_input("📸 Tap below to take a picture")

if camera_image:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)

    with st.spinner("🧠 Extracting and correcting text..."):
        preprocessed = preprocess_image(image)
        final_text = extract_text(preprocessed)

    if final_text:
        st.success("✅ Text extracted and spoken successfully!")

        st.markdown("### 📝 Extracted & Corrected Text")
        st.text_area("Output", final_text, height=200)

        audio_data = text_to_speech(final_text)
        st.audio(audio_data, format="audio/mp3")

        st.download_button("⬇️ Download Audio", data=audio_data, file_name="speech.mp3", mime="audio/mpeg", use_container_width=True)
    else:
        st.error("❌ No readable text found. Try again with better lighting or closer framing.")
else:
    st.info("Take a picture to begin.")

# --- Style for Mobile ---
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
