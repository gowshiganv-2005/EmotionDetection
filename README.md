# RajGowthamTune 🎭✨

**RajGowthamTune** is a high-precision, AI-driven Tamil music recommendation system. It uses computer vision to detect your face, analyze your emotion, gender, and age, and then suggests the latest Tamil hit songs (2021–2025) suited to your mood.

## 🚀 Features
- **🎯 High-Precision Detection:** Powered by **YOLOv8-face** for robust face localization.
- **🧠 Emotion Intelligence:** Custom Keras model + DeepFace for emotion, gender, and age prediction.
- **🎵 Tamil Music Engine:** Curated database of 50+ latest Tamil tracks mapped to user moods.
- **✨ Premium UI:** Modern midnight-teal aesthetic with **Aero-Glassmorphism** and responsive design.
- **⚡ Auto-Sync Mode:** Live tracking that updates recommendations automatically as your expression changes.

## 🛠 Tech Stack
- **Frontend:** React (Vite), Vanilla CSS (Premium Glassmorphism).
- **Backend:** Flask (Python).
- **AI Models:** YOLOv8 (Ultralytics), TensorFlow/Keras, DeepFace.
- **Tools:** OpenCV, PIL.

## 📦 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/gowshiganv-2005/EmotionDetection.git
cd EmotionDetection
```

### 2. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Flask API
python app.py
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🎥 Usage
1. Open `http://localhost:5173` in your browser.
2. Click **Start Camera**.
3. Click **Sync Emotion** or playback a song to trigger **Auto-Sync**.
4. Enjoy your personalized Tamil playlist!

---
Created with ❤️ by RajGowtham.
