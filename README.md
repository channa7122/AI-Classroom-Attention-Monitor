# 🧠 AI Classroom Attention Monitor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![DeepFace](https://img.shields.io/badge/DeepFace-Emotion%20Analysis-orange)

A real-time computer vision system designed to analyze student engagement and emotional states in a classroom setting.

---

## 🚀 Features

- **Real-time Face Detection**  
  Uses OpenCV Haar Cascades for fast and efficient face detection.

- **Emotion Recognition**  
  Leverages `DeepFace` (VGG-Face) to classify emotions (Happy, Neutral, Surprise, Sad, Angry, etc.).

- **Attention Tracking**  
  Calculates a dynamic **Focus Score** based on gaze direction and head pose estimation.

- **Drowsiness Detection**  
  Detects prolonged eye closure and triggers alerts.

- **Automated Reporting**  
  Generates visual analytics (charts and graphs) after each session.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Libraries:** OpenCV, DeepFace, Pandas, Matplotlib
- **Data Logging:** CSV (Timestamped session logs)

---

## 📂 Project Structure

```text
AI_Classroom_Monitor/
├── src/
│   ├── detector.py       # Core AI logic (Face, Eyes, Emotion)
│   └── __init__.py
├── assets/
│   └── known_faces/      # Add reference images for face recognition
├── logs/                 # Session CSV logs (auto-generated)
├── main.py               # Main application entry point
├── analysis.py           # Report & analytics generator
└── README.md
```

---

## ⚡ How to Run

### 1️⃣ Install Dependencies

```bash
pip install opencv-python deepface pandas matplotlib
```

### 2️⃣ Start the Monitor

```bash
python main.py
```

Press **`q`** to quit the session.

### 3️⃣ Generate the Report

```bash
python analysis.py
```

---

## 📊 Analytics Dashboard

The system automatically:

- Logs session data into CSV files
- Generates visual reports using Matplotlib
- Saves analytics charts (e.g., `latest_report.png`)

> 📌 Add a screenshot of `latest_report.png` here for better presentation.

---

## 👨‍💻 Author

**Mohamed Channa**

---

## 📌 Future Improvements (Optional Section)

- Head pose estimation using MediaPipe
- Web dashboard version (Streamlit or Flask)
- Database integration (PostgreSQL / MongoDB)
- Cloud deployment support

---

⭐ If you find this project useful, consider giving it a star!
