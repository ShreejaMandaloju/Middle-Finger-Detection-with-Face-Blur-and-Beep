# 🖐️ FingerSense: Middle Finger & Face Blur Detection with Beep

FingerSense is a fun and privacy-focused computer vision tool that detects when someone shows the middle finger in front of a webcam — and censors it **with a blur and a beep**!  
Additionally, it blurs **faces** to protect identities in real time.

---

## ✨ Features

- ✅ **Real-time middle finger detection** using MediaPipe
- 🧠 Smart **gesture voting system** to reduce flickering
- 🔊 **Beep sound alert** when middle finger is detected
- 🧍 **Automatic face blur** using Haar Cascades (OpenCV)
- 🔒 Keeps users’ faces private along with gesture moderation
- 🖥️ Simple UI with OpenCV preview window
- 💻 Lightweight & easy to run on most systems

---

## 📦 Installation

**1. Clone the repo**
```bash
git clone https://github.com/ShreejaMandaloju/Middle-Finger-Detection-with-Face-Blur-and-Beep.git
cd Middle-Finger-Detection-with-Face-Blur-and-Beep
```
**2. Create a virtual environment** (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
```

**3. Install the dependencies**
```bash
pip install -r requirements.txt
```

### 🚀 Usage
Run the application with:
```bash
python main.py
```
## Press ***Ctrl+C or Esc*** to exit the application window.

## 🧠 How it Works
- Uses MediaPipe Hands to track hand landmarks
- Tracks the Y-position of key fingers to detect the middle finger only
- Applies Gaussian blur over the region of the finger
- Uses OpenCV Haar cascade to detect and blur faces
- Plays a beep sound when the middle finger is detected

