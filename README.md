# 🎵 Adaptive Neural Audio Mixing with Human-in-the-Loop Feedback 🎛️ 

An AI-driven **adaptive audio mixing system** that integrates **human-in-the-loop feedback** using **reinforcement learning (RL)** to continuously refine audio mixing quality based on user interactions.

## 🚀 Features

- **AI-Powered Mixing** 🎼: Uses **Demucs** for source separation and applies AI-driven mix adjustments.
- **Interactive UI** 🎚️: Built with **Streamlit** for real-time mixing, EQ, and effects controls.
- **Reinforcement Learning (RL) Optimization** 🧠: AI model adapts based on user feedback using **Stable-Baselines3 PPO**.
- **Audio Visualization** 📊: Live waveform & spectrogram analysis for better decision-making.

---

## 📌 Table of Contents

- [🎯 Introduction](#-introduction)
- [🛠️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [🔬 System Architecture](#-system-architecture)
- [🧪 Model Training](#-model-training)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [📝 Future Work](#-future-work)
- [📚 References](#-references)
- [💡 Contributors](#-contributors)
- [🌟 Support & Feedback](#-support--feedback)

---

## 🎯 Introduction

### **Problem Statement**
Traditional **audio mixing** requires expert knowledge, while automated tools (e.g., LANDR, iZotope) optimize technical parameters but fail to **capture artistic intent**. This project introduces an AI-powered **adaptive neural mixing system** that combines **automation with human feedback**.

### **Objectives**
- 🧠 **AI-Generated Mixing**: AI creates an initial mix from multi-track audio.
- 🎚️ **Interactive Mixing UI**: Users adjust **volume, EQ, panning, and effects**.
- 📈 **Reinforcement Learning (RL)**: AI **learns from user interactions** to improve mixing decisions.

---

## 🛠️ Installation Guide

This document provides a step-by-step guide to installing and setting up the **Adaptive Neural Audio Mixing System**.

### 1️⃣ Prerequisites

Ensure you have the following installed:
- **Python 3.8+**
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Virtual environment (Recommended)**

### 2️⃣ Clone the Repository

```sh
git clone https://github.com/YOUR-USERNAME/adaptive-neural-audio-mixing.git
cd adaptive-neural-audio-mixing
```

### 3️⃣ Setup a Virtual Environment (Optional but Recommended)
```
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 4️⃣ Install Dependencies
```
pip install -r requirements.txt
```

### 5️⃣ Running the Streamlit UI
```
streamlit run app.py
```
Your AI audio mixing system is now ready to use! 🚀

## 1️⃣ Launching the App
To start the interactive mixing system:
```sh
streamlit run app.py
```

