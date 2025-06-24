import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
from gtts import gTTS
import tempfile

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for better OCR using OpenCV."""
    img = np.array(image.convert("RGB"))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Remove noise
    blur = cv2.medianBlur(gray, 3)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Dilation to enhance characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    return dilated

def extract_text(preprocessed_img: np.ndarray) -> str:
    """Extract text from preprocessed image using Tesseract."""
    return pytesseract.image_to_string(preprocessed_img).strip()

def text_to_speech(text: str) -> bytes:
    """Convert text to speech and return audio bytes."""
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tts.save(tmp_file.name)
        tmp_file.seek(0)
        audio_bytes = tmp_file.read()
    return audio_bytes

# --- Streamlit UI ---

st.set_page_config(page_title="Text Reader for the Blind", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>📷 SpeakText - Scan and Listen</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center;'>Point your camera at printed text. The app will read it out loud automatically.</p>",
    unsafe_allow_html=True,
)

# --- Camera input only ---
camera_image = st.camera_input("📸 Tap to Capture", key="camera")

if camera_image:
    image = Image.open(camera_image)

    with st.spinner("🧠 Processing image and extracting text..."):
        preprocessed = preprocess_image(image)
        extracted_text = extract_text(preprocessed)

    if extracted_text:
        st.success("✅ Text detected and spoken!")
        st.markdown("### 📝 Extracted Text")
        st.text_area("Extracted Text", extracted_text, height=200)

        audio_bytes = text_to_speech(extracted_text)

        st.audio(audio_bytes, format="audio/mp3")

        st.download_button(
            label="⬇️ Download Audio",
            data=audio_bytes,
            file_name="speech.mp3",
            mime="audio/mpeg",
            use_container_width=True
        )
    else:
        st.error("❌ No readable text found. Try again.")
else:
    st.info("Use the camera above to scan your document.")

# --- Optional: Add spacing and large buttons for mobile UX ---
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
