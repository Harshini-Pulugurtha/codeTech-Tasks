import streamlit as st
import speech_recognition as sr
import tempfile

# Page config
st.set_page_config(page_title="Speech-to-Text", layout="centered")

st.title("🎧 Speech-to-Text System")
st.write("Upload an audio file and get transcription")

# Initialize recognizer
recognizer = sr.Recognizer()

# Upload audio file
uploaded_file = st.file_uploader("📂 Upload audio file", type=["wav"])

if uploaded_file is not None:
    # Play audio
    st.audio(uploaded_file, format="audio/wav")

    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    # Transcribe button
    if st.button("🧠 Transcribe"):
        with st.spinner("Processing audio..."):
            with sr.AudioFile(temp_path) as source:
                audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data)
                st.success("📝 Transcription:")
                st.write(text)

            except sr.UnknownValueError:
                st.error("❌ Could not understand audio")

            except sr.RequestError:
                st.error("❌ API error / No internet")