# run_frontend.py
import streamlit as st
import requests
from PIL import Image
import os
import json

# Backend API URL
BACKEND_URL = "http://127.0.0.1:8002"

# Streamlit UI Title
st.title("🎶 Adaptive Neural Audio Mixing System 🎵")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    # ✅ FIX: Play the uploaded audio file
    st.audio(uploaded_file, format="audio/wav")
    st.write(f"✅ Uploaded File: **{uploaded_file.name}**")

# Sidebar for real-time adjustments
st.sidebar.title("🎛️ Adjust Mix Preferences")

# Mixing controls
volume = st.sidebar.slider("🔊 Volume", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
panning = st.sidebar.slider("🎚️ Panning (Left: -1.0, Center: 0.0, Right: 1.0)", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
bass_boost = st.sidebar.slider("🎸 Bass Boost (dB)", min_value=-10, max_value=10, value=0)
midrange_boost = st.sidebar.slider("🎼 Midrange Boost (dB)", min_value=-10, max_value=10, value=0)
treble_boost = st.sidebar.slider("🎶 Treble Boost (dB)", min_value=-10, max_value=10, value=0)
add_reverb = st.sidebar.checkbox("🎵 Add Reverb")

# User ID input
user_id = st.text_input("👤 Enter your User ID", "default_user")

# Ensure a file is uploaded before using its name
file_name = uploaded_file.name if uploaded_file else "Unknown"

# Save User Preferences Button (Logging)
if st.sidebar.button("💾 Save Adjustments"):
    if uploaded_file:
        mix_settings = {
            "volume": volume,
            "panning": panning,
            "bass_boost": bass_boost,
            "midrange_boost": midrange_boost,
            "treble_boost": treble_boost,
            "add_reverb": add_reverb
        }

        try:
            response = requests.post(
                f"{BACKEND_URL}/log_feedback/",
                data={
                    "user_id": user_id,  
                    "track_name": file_name,  
                    "final_thoughts": "User adjusted mix",  
                    "mix_settings": json.dumps(mix_settings)  # ✅ FIX: Send mix settings properly
                }
            )
            if response.status_code == 200:
                st.sidebar.success("✅ Mix adjustments saved successfully!")
            else:
                st.sidebar.error("❌ Failed to save adjustments.")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"⚠️ Network Error: {e}")

    else:
        st.sidebar.warning("⚠️ Please upload a file before saving adjustments.")

# Retrain AI Model Button
if st.sidebar.button("🚀 Retrain AI Model"):
    with st.spinner("⏳ Retraining AI model... This may take some time!"):
        try:
            response = requests.post(f"{BACKEND_URL}/retrain_ai/")
            if response.status_code == 200:
                st.sidebar.success("✅ AI Model Updated!")
            else:
                st.sidebar.error("❌ AI Training Failed.")
        except requests.exceptions.RequestException as e:
            st.sidebar.error(f"⚠️ Network Error: {e}")

# **New: Save Initial Thoughts Before Processing**
st.subheader("📝 Share your initial thoughts before processing...")
initial_thoughts = st.text_area("Write your thoughts here...", key="initial_thoughts")

if st.button("💾 Save Initial Thoughts"):
    if uploaded_file:
        try:
            response = requests.post(
                f"{BACKEND_URL}/log_initial_thoughts/",
                data={"user_id": user_id, "track_name": file_name, "initial_thoughts": initial_thoughts},
            )
            if response.status_code == 200:
                st.success("✅ Initial thoughts saved successfully!")
            else:
                st.error("❌ Failed to save initial thoughts.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network Error: {e}")
    else:
        st.warning("⚠️ Please upload a file before saving initial thoughts.")


# Processing Button
if st.button("🚀 Process Audio"):
    if uploaded_file:
        with st.spinner("🎧 Processing... Please wait!"):
            # Prepare file and data for backend processing
            files = {"file": (file_name, uploaded_file.getvalue())}
            data = {"user_id": user_id, "feedback": initial_thoughts}

            try:
                response = requests.post(f"{BACKEND_URL}/process_audio/", files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ **Processing Complete!**\nProcessed File: **{result.get('processed_path', 'N/A')}**")

                    # Display waveform visualization
                    if "waveform_path" in result and os.path.exists(result["waveform_path"]):
                        st.subheader("📊 Waveform Visualization")
                        st.image(Image.open(result["waveform_path"]))
                    else:
                        st.warning("⚠️ Waveform visualization not available.")

                    # Display spectrogram visualization
                    if "spectrogram_path" in result and os.path.exists(result["spectrogram_path"]):
                        st.subheader("📈 Spectrogram Visualization")
                        st.image(Image.open(result["spectrogram_path"]))
                    else:
                        st.warning("⚠️ Spectrogram visualization not available.")

                    # Display evaluation metrics
                    if "sdr" in result and result["sdr"] is not None:
                        st.write(f"🎚️ **Signal-to-Distortion Ratio (SDR):** {result['sdr']:.2f} dB")
                    else:
                        st.warning("⚠️ SDR metric not available.")
                else:
                    st.error("❌ Audio processing failed. Please check the backend server.")
            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ Network Error: {e}")
    else:
        st.warning("⚠️ Please upload an audio file before processing.")

# **Feedback submission section (After Processing)**
st.subheader("💬 Provide Feedback on the AI Mixing 🎵")
feedback_text_after = st.text_area("📝 Share your thoughts on the AI-mixed audio after listening...", key="feedback_after_processing")

if st.button("📨 Submit Feedback"):
    if uploaded_file:
        # Ensure mix_settings is defined (even if the user hasn't adjusted anything)
        mix_settings = {
            "volume": volume,
            "panning": panning,
            "bass_boost": bass_boost,
            "midrange_boost": midrange_boost,
            "treble_boost": treble_boost,
            "add_reverb": add_reverb
        }

        try:
            response = requests.post(
                f"{BACKEND_URL}/log_feedback/",
                data={
                    "user_id": user_id,  # Ensure user_id is sent
                    "track_name": file_name,  # Ensure track_name is sent
                    "final_thoughts": feedback_text_after,  # Ensure final_thoughts is sent
                    "mix_settings": json.dumps(mix_settings)  # FIX: Define mix_settings properly
                }
            )
            if response.status_code == 200:
                st.success("✅ Feedback submitted successfully!")
            else:
                st.error("❌ Failed to submit feedback. Please try again.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network Error: {e}")
    else:
        st.warning("⚠️ Please upload a file before submitting feedback.")


# View Feedback Logs
if st.button("📜 View Feedback Logs"):
    with st.spinner("📖 Fetching feedback logs..."):
        try:
            response = requests.get(f"{BACKEND_URL}/get_feedback/")
            if response.status_code == 200:
                logs = response.json().get("logs", [])
                st.subheader("🗒️ **User Feedback Logs**")
                if logs:
                    for log in logs:
                        st.write(log)
                else:
                    st.warning("⚠️ No feedback logs available yet.")
            else:
                st.error("❌ Failed to fetch feedback logs.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network Error: {e}")

# View Processed Files
if st.button("📂 View Processed Files"):
    with st.spinner("🔍 Searching for processed files..."):
        try:
            response = requests.get(f"{BACKEND_URL}/get_processed/", params={"filename": file_name})
            if response.status_code == 200:
                st.write(f"🎵 **Processed File:** {response.json().get('processed_file', 'N/A')}")
            else:
                st.error("❌ Error fetching processed file.")
        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ Network Error: {e}")




