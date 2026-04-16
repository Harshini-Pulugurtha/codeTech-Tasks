import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import time
import io

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Neural Style Transfer", layout="centered")

st.title("🎨 Neural Style Transfer")
st.write("Upload a content image and a style image to create artistic output")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

model = load_model()

# -------------------- IMAGE PROCESSING --------------------
def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    return img[np.newaxis, :]

# -------------------- FILE UPLOAD --------------------
st.subheader("📸 Upload Content Image")
content_file = st.file_uploader("Choose a content image", type=["jpg", "png"], key="content")

st.subheader("🎨 Upload Style Image")
style_file = st.file_uploader("Choose a style image", type=["jpg", "png"], key="style")

# -------------------- DISPLAY INPUTS --------------------
if content_file:
    st.image(content_file, caption="Content Image", use_column_width=True)

if style_file:
    st.image(style_file, caption="Style Image", use_column_width=True)

# -------------------- PROCESS BUTTON --------------------
if st.button("✨ Apply Style Transfer"):
    if content_file is None or style_file is None:
        st.warning("⚠️ Please upload both images")
    else:
        with st.spinner("Applying style... ⏳"):
            start = time.time()

            # Load images
            content_image = load_image(content_file)
            style_image = load_image(style_file)

            # ✅ CORRECT MODEL CALL (for your setup)
            stylized_image = model(
                tf.constant(content_image),
                tf.constant(style_image)
            )[0]

            output = stylized_image.numpy().squeeze()

            end = time.time()

        # -------------------- OUTPUT --------------------
        st.subheader("🖼️ Stylized Output")
        st.image(output, use_column_width=True)

        st.success("✅ Style Transfer Completed!")
        st.info(f"⏱️ Time Taken: {round(end - start, 2)} seconds")

        # -------------------- DOWNLOAD --------------------
        output_img = Image.fromarray((output * 255).astype(np.uint8))
        buf = io.BytesIO()
        output_img.save(buf, format="PNG")

        st.download_button(
            label="📥 Download Image",
            data=buf.getvalue(),
            file_name="stylized.png",
            mime="image/png"
        )