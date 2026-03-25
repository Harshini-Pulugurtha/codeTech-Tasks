import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="🧠",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🧠 AI Text Summarization Tool")
st.markdown("Generate concise summaries using Transformer-based NLP models")

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_model(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.replace("\n", " ").strip()
    return text

# -------------------- SUMMARIZATION --------------------
def summarize_text(text, model, tokenizer, max_len, min_len):
    text = clean_text(text)
    input_ids = tokenizer.encode(
        f"summarize: {text}",
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    outputs = model.generate(
        input_ids,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- SIDEBAR --------------------
st.sidebar.header("⚙️ Settings")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["t5-small", "t5-base"]
)

length_option = st.sidebar.selectbox(
    "Summary Length Preset",
    ["Short", "Medium", "Long"]
)

# Length presets
if length_option == "Short":
    max_len, min_len = 50, 20
elif length_option == "Medium":
    max_len, min_len = 100, 40
else:
    max_len, min_len = 150, 60

# -------------------- LOAD MODEL --------------------
model, tokenizer = load_model(model_option)

# -------------------- INPUT METHOD --------------------
option = st.radio("📥 Choose input method:", ["Upload File", "Enter Text"])

text = ""

if option == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        st.subheader("📄 Uploaded Text")
        st.text_area("", text, height=200)

else:
    text = st.text_area("✍️ Enter your text here:", height=200)

# -------------------- TEXT STATS --------------------
if text:
    word_count = len(text.split())
    st.info(f"📊 Word Count: {word_count}")

# -------------------- SUMMARIZE BUTTON --------------------
if st.button("✨ Generate Summary"):
    if text.strip() == "":
        st.warning("⚠️ Please provide some text!")
    else:
        with st.spinner("Generating summary... ⏳"):
            start = time.time()

            summary = summarize_text(
                text[:1000],  # limit input
                model,
                tokenizer,
                max_len,
                min_len
            )

            end = time.time()

        st.subheader("📌 Summary")
        st.success(summary)

        # -------------------- METRICS --------------------
        st.markdown(f"⏱️ Time Taken: `{round(end - start, 2)} sec`")
        st.markdown(f"📝 Summary Length: `{len(summary.split())} words`")

        # -------------------- DOWNLOAD BUTTON --------------------
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt",
            mime="text/plain"
        )