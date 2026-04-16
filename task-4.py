import streamlit as st
from transformers import pipeline
import time

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="GPT Text Generator", layout="centered")

st.title("🤖 AI Text Generator")
st.write("Generate coherent paragraphs using GPT model")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

generator = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")

max_len = st.sidebar.slider("Max Length", 50, 300, 120)
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.5, 0.8)
num_outputs = st.sidebar.slider("Number of Outputs", 1, 3, 1)

# -------------------- INPUT --------------------
prompt = st.text_area("✍️ Enter your topic or prompt:", height=150)

# -------------------- GENERATE --------------------
if st.button("✨ Generate Text"):
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt")
    else:
        with st.spinner("Generating... ⏳"):
            start = time.time()

            outputs = generator(
                prompt,
                max_length=max_len,
                num_return_sequences=num_outputs,
                temperature=temperature
            )

            end = time.time()

        st.subheader("📄 Generated Text")

        for i, out in enumerate(outputs):
            st.markdown(f"### ✨ Output {i+1}")
            st.write(out["generated_text"])

        st.info(f"⏱️ Time Taken: {round(end - start, 2)} sec")

        # -------------------- DOWNLOAD --------------------
        full_text = "\n\n".join([o["generated_text"] for o in outputs])

        st.download_button(
            label="📥 Download Text",
            data=full_text,
            file_name="generated_text.txt",
            mime="text/plain"
        )