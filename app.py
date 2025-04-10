import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Configure page
st.set_page_config(page_title="AI Image Processor", layout="wide", page_icon="🎨")
st.title("🎨 AI-Powered Image Processing")

# Custom CSS for modern UI
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(45deg, #1a1a1a, #2a2a2a) !important;
        color: white !important;
    }
    .stButton>button {border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# Helper functions
def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def negative_image(img):
    return 255 - img

def custom_convolution(image, kernel):
    kh, kw = kernel.shape
    pad = kh // 2
    image = to_grayscale(image)
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
    output = np.zeros_like(image)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = image_padded[y:y+kh, x:x+kw]
            output[y, x] = np.clip(np.sum(region * kernel), 0, 255)
    return output

def average_smoothing_5x5(img):
    kernel = np.ones((5, 5), np.float32) / 25
    return custom_convolution(img, kernel)

def median_smoothing(img, size=3):
    return cv2.medianBlur(to_grayscale(img), size)

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))

def gaussian_smoothing(img, sigma=1.0):
    return cv2.GaussianBlur(to_grayscale(img), (5, 5), sigma)

def edge_detection(img):
    return cv2.Canny(to_grayscale(img), 100, 200)

def resize_image(img, scale_factor):
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

def convert_to_image_bytes(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img)
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()

# Sidebar UI
with st.sidebar:
    st.header("📤 Upload Your Image")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        st.header("🎚 Processing Controls")
        processor = st.radio("Select Operation:", [
            "Negative Image", "Average Smoothing (5x5)", "Median Smoothing",
            "Rotate Image", "Gaussian Smoothing", "Edge Detection", "Resize Image", "Custom Convolution"
        ])
        
        if processor == "Rotate Image":
            angle = st.slider("Rotation Angle", -180, 180, 0, 5)
        elif processor == "Gaussian Smoothing":
            sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
        elif processor == "Resize Image":
            scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
        elif processor == "Custom Convolution":
            kernel_input = st.text_area("Enter 3x3 Kernel (comma-separated rows)", "0,-1,0\n-1,5,-1\n0,-1,0")

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("🖼 Original Image")
        st.image(image, use_column_width=True, clamp=True)
        st.text(f"Dimensions: {image.shape[1]}x{image.shape[0]}")
        st.text(f"Channels: {'Grayscale' if len(image.shape)==2 else 'RGB'}")

    processed = image.copy()
    try:
        if processor == "Negative Image":
            processed = negative_image(image)
        elif processor == "Average Smoothing (5x5)":
            processed = average_smoothing_5x5(image)
        elif processor == "Median Smoothing":
            processed = median_smoothing(image, 3)
        elif processor == "Rotate Image":
            processed = rotate_image(image, angle)
        elif processor == "Gaussian Smoothing":
            processed = gaussian_smoothing(image, sigma)
        elif processor == "Edge Detection":
            processed = edge_detection(image)
        elif processor == "Resize Image":
            processed = resize_image(image, scale_factor)
        elif processor == "Custom Convolution":
            kernel = np.array([[float(num) for num in row.split(",")] for row in kernel_input.split("\n")])
            if kernel.shape == (3, 3):
                processed = custom_convolution(image, kernel)
            else:
                st.sidebar.error("Kernel must be 3x3.")
    except Exception as e:
        st.error(f"Processing failed: {e}")

    with col2:
        st.subheader("🎨 Processed Image")
        st.image(processed, use_column_width=True, clamp=True)
        st.download_button("⬇️ Download Processed Image", convert_to_image_bytes(processed), file_name="processed_image.png", mime="image/png")

else:
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px">
        <h2 style="color: #666">📁 Upload an Image to Start</h2>
        <p style="color: #444">Supports JPG, PNG, JPEG formats</p>
    </div>
    """, unsafe_allow_html=True)
