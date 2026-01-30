# SMART-INVIGILATOR
# Real-Time Classroom Mobile Phone Detection System

**Smart Invigilator** is an AI system developed to maintain academic integrity by detecting mobile phone usage in classrooms in real time. By combining object detection, pose estimation, and behavioral analysis, this system differentiates between active phone use (cheating) and passive holding, providing automated alerts and evidence logging.

---

## 🚀 How to Run

Follow this step-by-step guide to set up the project on your local machine.

 1️⃣ Clone the Repository


2️⃣ Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
venv\Scripts\activate
```

3️⃣ Install Required Libraries
You can install the dependencies using one of the following methods:

*Option A:* Using requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```
*option B:* Manual Installation
```bash
pip install ultralytics numpy opencv-python mediapipe==0.10.21 scikit-learn flask
```
4️⃣ Run the System
Once the environment is set up and libraries are installed, run the main detection script:
```bash
python main.py
```

⚡ Features
🎯 Real-Time Detection
Integrated YOLOv8 / YOLOv9 models for high-accuracy detection of students and mobile devices.
🧠 Behavioral Analysis
Implemented MediaPipe hand tracking and pose estimation to identify head tilt and hand-to-phone interactions.
👤 Student–Phone Association
Built logic using IoU (Intersection over Union) and geometric matching to correctly associate a detected phone with the specific student holding it.
📊 Activity Classification
Designed a scikit-learn classification model to classify behavior into:
Active Phone Use
Passive Holding
No-Phone Condition
💡 Screen Glow Detection
Added a dedicated algorithm to detect screen light/glow, enabling detection even in:
Low-light environments
Phones hidden under desks
🚨 Alerts & Logging
Real-time Audio & Visual Alerts when cheating is detected.
Automated incident logging using SQLite and CSV for record-keeping.
📡 Live Dashboard
Optional Flask-based live dashboard for remote monitoring and data visualization.
🛠️ Tech Stack
Python
YOLOv8 / YOLOv9 (Ultralytics)
OpenCV
MediaPipe
Scikit-learn
Flask (Dashboard)
SQLite / CSV
