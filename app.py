# app.py
from fastapi import FastAPI, UploadFile, Form
from fastapi import Request
from datetime import datetime
import os
import librosa
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from mir_eval.separation import bss_eval_sources
import demucs.separate
from stable_baselines3 import PPO
import json 
import subprocess
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3 import PPO
import gymnasium as gym
from mixing_env import AudioMixingEnv

app = FastAPI()

# Directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
VISUALIZATION_DIR = os.path.join(PROCESSED_DIR, "visualizations")
LOG_FILE = "feedback_log.csv"
COMBINED_FEEDBACK_FILE = "combined_feedback_log.csv"  # Stores both initial & final feedback
MIX_ADJUSTMENTS_FILE = "mix_adjustments_log.csv"  # Stores only mix adjustment feedback

# Ensure files exist and have the correct headers
for file in [LOG_FILE, COMBINED_FEEDBACK_FILE, MIX_ADJUSTMENTS_FILE]:
    if not os.path.exists(file) or os.stat(file).st_size == 0:
        with open(file, "w", encoding="utf-8") as f:
            f.write("Timestamp\tUser ID\tTrack Name\tInitial Thoughts\tFinal Thoughts\tMix Settings\n")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Function to retrain the AI model
def retrain_model():
    try:
        global MODEL_PATH
        MODEL_PATH = "adaptive_mixing_model/ppo_mixing_model.zip"
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

        # ✅ Use the real audio mixing environment
        env = AudioMixingEnv()  

        if os.path.exists(MODEL_PATH):
            print("🔄 Loading existing model for retraining...")
            try:
                model = PPO.load(MODEL_PATH, env=env)
                if model.observation_space.shape != env.observation_space.shape:
                    print("⚠️ Observation space mismatch! Creating a new model.")
                    model = PPO("MlpPolicy", env, verbose=1)
            except Exception as e:
                print(f"⚠️ Model loading failed: {e}. Creating a new model...")
                model = PPO("MlpPolicy", env, verbose=1)
        else:
            print("⚠️ Model file not found, creating a new one...")
            model = PPO("MlpPolicy", env, verbose=1)  

        # ✅ Retrain for 50,000 steps
        print("🎛️ Retraining AI model on real audio mixing data...")
        model.learn(total_timesteps=50000)

        # ✅ Save the trained model
        model.save(MODEL_PATH)
        print("✅ AI model retrained and saved with real audio environment!")

        return True
    except Exception as e:
        print(f"❌ AI retraining failed: {e}")
        return False

@app.get("/")
async def root():
    return {"message": "Welcome to the Adaptive Neural Audio Mixing API!"}

# SDR Calculation
def compute_sdr(original_audio, mixed_audio):
    try:
        sdr, _, _, _ = bss_eval_sources(original_audio, mixed_audio)
        return np.mean(sdr)
    except Exception as e:
        print(f"⚠️ SDR Calculation Failed: {e}")
        return None

# Function to save waveform visualization
def save_waveform(audio: np.ndarray, sr: int, output_file: str):
    plt.figure(figsize=(12, 4))
    if np.max(np.abs(audio)) > 1:
        print("⚠️ Audio exceeds [-1, 1] range. Scaling down for visualization.")
        audio = audio / np.max(np.abs(audio))
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    time_axis = np.linspace(0, len(audio) / sr, num=len(audio))
    plt.plot(time_axis, audio, alpha=0.75)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.ylim(-1.1, 1.1)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

# Function to save spectrogram visualization
def save_spectrogram(audio: np.ndarray, sr: int, output_file: str, n_fft=2048, hop_length=512):
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

# AI Model: Apply Source Separation and Adaptive Mixing
def apply_mixing_adjustments(file_path):
    """
    Applies AI-based mixing adjustments after separating the audio using Demucs.
    """
    try:
        output_dir = os.path.join(PROCESSED_DIR, "demucs_output")
        os.makedirs(output_dir, exist_ok=True)

        print("🔍 Running Demucs for source separation...")
        demucs.separate.main(["-n", "htdemucs", "--out", output_dir, file_path])

        # Load the separated stems
        track_name = os.path.splitext(os.path.basename(file_path))[0]
        sources = ["vocals", "drums", "bass", "other"]
        separated_tracks = {}
        sr = None

        for src in sources:
            stem_path = os.path.join(output_dir, "htdemucs", track_name, f"{src}.wav")
            if os.path.exists(stem_path):
                separated_tracks[src], sr = librosa.load(stem_path, sr=None, mono=False)
            else:
                print(f"⚠️ Missing stem: {src}. Check Demucs output.")

        if len(separated_tracks) != 4:
            raise ValueError("❌ Demucs separation failed: Not all stems were processed.")

        # ✅ Load the AI model for adaptive mixing
        if os.path.exists(MODEL_PATH):
            model = PPO.load(MODEL_PATH)
            print("🎛️ Using AI model for mixing adjustments...")

            # Extracting features for AI decision-making
            feature_vector = np.array([
                np.mean(np.abs(separated_tracks["vocals"])),
                np.mean(np.abs(separated_tracks["drums"])),
                np.mean(np.abs(separated_tracks["bass"])),
                np.mean(np.abs(separated_tracks["other"]))
            ]).reshape(1, -1)

            action, _ = model.predict(feature_vector)
            gain_factors = np.clip(action, 0.5, 2.0).flatten()

            if gain_factors.shape != (4,):  # Fix shape mismatch
                print("⚠️ AI output shape mismatch. Reshaping to correct form.")
                gain_factors = gain_factors.reshape((4,))

            print(f"✅ AI Adjustments Applied: {gain_factors}")

            # Apply AI-based volume adjustments to each track
            for i, src in enumerate(sources):
                if separated_tracks[src].ndim == 2:  # Stereo audio
                    separated_tracks[src] *= gain_factors[i]
                else:  # Mono audio
                    separated_tracks[src] *= gain_factors[i]

        # Mix tracks together
        mixed_audio = np.sum(list(separated_tracks.values()), axis=0)

        # Normalize audio
        mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))

        return mixed_audio, sr

    except Exception as e:
        print(f"❌ Mixing Adjustment Failed: {e}")
        return None, None

# Logging function
def log_feedback(user_id, track_name, feedback, mix_settings=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if mix_settings and isinstance(mix_settings, dict):
        mix_settings_json = json.dumps(mix_settings, indent=None)
    else:
        mix_settings_json = "{}"
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp}\t{user_id}\t{track_name}\t{feedback}\t{mix_settings_json}\n")

@app.post("/process_audio/")
async def process_audio(request: Request, file: UploadFile, user_id: str = Form(...), feedback: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    processed_path = os.path.join(PROCESSED_DIR, f"processed_{file.filename}")
    waveform_path = os.path.join(VISUALIZATION_DIR, f"waveform_{file.filename}.png")
    spectrogram_path = os.path.join(VISUALIZATION_DIR, f"spectrogram_{file.filename}.png")

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"✅ File saved successfully: {file_path}")
    except Exception as e:
        return {"error": f"File save error: {str(e)}"}

    # Apply AI-Based Adaptive Mixing
    try:
        adapted_audio, sr = apply_mixing_adjustments(file_path)
        if adapted_audio is None or sr is None:
            return {"error": "AI mixing failed, no processed audio available."}
    except Exception as e:
        return {"error": f"AI Mixing Model Error: {str(e)}"}

    # Save processed audio
    try:
        sf.write(processed_path, adapted_audio.T, sr)
        print(f"✅ Processed audio saved: {processed_path}")
    except Exception as e:
        print(f"❌ Failed to save processed audio: {e}")
        return {"error": "Failed to save processed audio"}

    # Compute SDR (Evaluate AI Mixing Quality)
    try:
        reference_audio, _ = librosa.load(file_path, sr=sr, mono=False)
        sdr_value = compute_sdr(reference_audio, adapted_audio)
        print(f"✅ Computed SDR: {sdr_value:.2f} dB")
    except Exception as e:
        print(f"⚠️ SDR Calculation Failed: {e}")
        sdr_value = None

    # Generate visualizations
    try:
        save_waveform(adapted_audio, sr, waveform_path)
        save_spectrogram(adapted_audio, sr, spectrogram_path)
        print("✅ Waveform and Spectrogram generated successfully.")
    except Exception as e:
        print(f"⚠️ Visualization generation failed: {e}")

    return {
        "message": f"🎵 Processed {file.filename}",
        "processed_path": processed_path,
        "waveform_path": waveform_path,
        "spectrogram_path": spectrogram_path,
        "sdr": sdr_value,
    }

@app.get("/get_processed/")
async def get_processed(filename: str):
    processed_file_path = os.path.join(PROCESSED_DIR, f"processed_{filename}")
    if os.path.exists(processed_file_path):
        return {"processed_file": processed_file_path}
    return {"error": "Processed file not found"}

@app.post("/retrain_ai/")
async def retrain_ai():
    global MODEL_PATH
    MODEL_PATH = "adaptive_mixing_model/ppo_mixing_model.zip"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    success = retrain_model()
    if success:
        return {"message": "✅ AI retraining completed successfully (or new model created)."}
    else:
        return {"error": "❌ AI retraining failed. Check logs for details."}

# Function to log initial thoughts
@app.post("/log_initial_thoughts/")
async def log_initial_thoughts(
    user_id: str = Form(...), 
    track_name: str = Form(...), 
    initial_thoughts: str = Form(...)
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp}\t{user_id}\t{track_name}\t{initial_thoughts}\t\t\n")
    return {"message": "✅ Initial thoughts logged successfully"}

# Function to log final thoughts and save both initial & final feedback together
@app.post("/log_feedback/")
async def log_feedback(
    user_id: str = Form(...), 
    track_name: str = Form(...), 
    final_thoughts: str = Form(...), 
    mix_settings: str = Form("{}")
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        mix_settings_dict = json.loads(mix_settings)
    except json.JSONDecodeError:
        mix_settings_dict = {}

    # Read and ensure correct CSV format for feedback_log.csv
    expected_feedback_columns = ["Timestamp", "User ID", "Track Name", "Initial Thoughts", "Final Thoughts", "Mix Settings"]
    try:
        feedback_df = pd.read_csv(LOG_FILE, sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip")
        if list(feedback_df.columns) != expected_feedback_columns:
            raise ValueError("Incorrect columns in feedback_log.csv")
    except (pd.errors.ParserError, ValueError):
        print(f"⚠️ Resetting corrupted {LOG_FILE}")
        feedback_df = pd.DataFrame(columns=expected_feedback_columns)
        feedback_df.to_csv(LOG_FILE, sep="\t", index=False, encoding="utf-8")

    # Read and ensure correct CSV format for mix_adjustments_log.csv
    expected_mix_columns = ["Timestamp", "User ID", "Track Name", "Mix Settings"]
    try:
        mix_df = pd.read_csv(MIX_ADJUSTMENTS_FILE, sep="\t", encoding="utf-8", dtype=str, on_bad_lines="skip")
        if list(mix_df.columns) != expected_mix_columns:
            raise ValueError("Incorrect columns in mix_adjustments_log.csv")
    except (pd.errors.ParserError, ValueError):
        print(f"⚠️ Resetting corrupted {MIX_ADJUSTMENTS_FILE}")
        mix_df = pd.DataFrame(columns=expected_mix_columns)
        mix_df.to_csv(MIX_ADJUSTMENTS_FILE, sep="\t", index=False, encoding="utf-8")

    # Append feedback to feedback_log.csv
    new_feedback_entry = pd.DataFrame([[timestamp, user_id, track_name, "", final_thoughts, json.dumps(mix_settings_dict)]], 
                                      columns=feedback_df.columns)
    feedback_df = pd.concat([feedback_df, new_feedback_entry], ignore_index=True)
    feedback_df.to_csv(LOG_FILE, sep="\t", index=False, encoding="utf-8")

    # Append mix adjustments to mix_adjustments_log.csv if settings exist
    if mix_settings_dict:
        new_mix_entry = pd.DataFrame([[timestamp, user_id, track_name, json.dumps(mix_settings_dict)]], 
                                     columns=mix_df.columns)
        mix_df = pd.concat([mix_df, new_mix_entry], ignore_index=True)
        mix_df.to_csv(MIX_ADJUSTMENTS_FILE, sep="\t", index=False, encoding="utf-8")

    return {"message": "✅ Feedback & mix adjustments logged successfully"}

# Endpoint to retrieve all feedback logs
@app.get("/get_feedback/")
async def get_feedback():
    if os.path.exists(COMBINED_FEEDBACK_FILE):
        with open(COMBINED_FEEDBACK_FILE, "r", encoding="utf-8") as log_file:
            logs = log_file.readlines()
        return {"logs": logs}
    return {"error": "No feedback logged yet."}

# Endpoint to retrieve mix adjustments logs separately
@app.get("/get_mix_adjustments/")
async def get_mix_adjustments():
    if os.path.exists(MIX_ADJUSTMENTS_FILE):
        try:
            mix_df = pd.read_csv(MIX_ADJUSTMENTS_FILE, sep="\t", encoding="utf-8")
            mix_logs = mix_df.to_dict(orient="records")
            return {"mix_adjustments": mix_logs}
        except Exception as e:
            return {"error": f"Failed to read mix adjustments: {e}"}
    return {"error": "No mix adjustments logged yet."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)


