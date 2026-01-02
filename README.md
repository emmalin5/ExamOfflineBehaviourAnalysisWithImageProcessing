# Cheating Detection for ASU Classroom

## 🔎 Overview

This project presents a **Computer Vision (CV)** based solution designed to enhance cheating detection during exams in classroom settings at **Arizona State University (ASU)**. By leveraging CV technology to monitor students' **head and eye movements**, the solution aims to ensure **academic integrity**, promote **honesty**, and encourage the development of fair study habits.

### 🚀 Streamlit App

1. Install dependencies: `pip install -r requirements.txt`
2. Run the UI: `streamlit run app.py`
3. Grant camera access in your OS prompt. The page shows real-time webcam video with status badges for cheating/not cheating. Adjust yaw sensitivity in the sidebar if detection feels too strict or lenient.

---

### 📝 Problem Definition

Maintaining academic integrity during exams is a persistent challenge in educational institutions. Traditional surveillance methods may fail to detect **covert cheating behaviors**, such as secret interactions between students or sneaking glances at unauthorized materials. This project seeks to address these gaps using **computer vision technology**.

---

### 💡 Importance of the Solution

Implementing a real-time monitoring system using CV can help in:
- **Enhanced cheating detection**: Monitoring student behavior more effectively.
- **Fostering a culture of honesty**: Building trust between students and faculty.
- **Promoting integrity**: Encouraging students to focus on fair study habits.

---

### 👥 Stakeholders and Beneficiaries

- **Stakeholders**: Faculty members, Office of Academic Integrity, Information Technology Department.
- **Beneficiaries**: Students and the academic community benefiting from fair assessments.

---

### ⚙️ Current Solutions and Limitations

Current solutions like browser lockdowns and seating arrangements cannot detect **subtle behavioral indicators** of cheating, such as:
- Sneaking glances at unauthorized materials.
- Secret interactions between students.
- Lack of live feedback for proctors during the exam.

---

### 💻 Proposed Solution (Role of Computer Vision)

Our proposed solution uses **CV algorithms** to track and analyze students' head and eye movements in real-time during exams. This system identifies suspicious behavior such as:
- Continuously looking away from the front view.
- Glancing in directions unrelated to the exam task.

The system employs **high-resolution webcams** to capture data, which is processed in real-time to detect and flag potential cheating patterns.

---

### 🔄 E2E Product Workflow

1. **Real-Time Monitoring**: Captures students' head and eye movements.
2. **Immediate Analysis**: CV model analyzes live feed for cheating indicators.
3. **Live Alerts**: Sends alerts when potential cheating is detected.
4. **Intervention Protocol**: Proctors are alerted to take real-time action.
5. **Confirmation & Action**: Proctors confirm and act on the identified cheating cases.

---

### 📊 Models Used

1. **YOLOv4 - Object Detection**
   - Detects human figures and objects such as phones or notes in real time.
   - Tracks the presence and location of unauthorized materials.

<img src="assets/Phone.png" align="center" width="541" height="305" style="margin-top: 40px;">

<img src="assets/MultipleFace.png" align="center" width="541" height="305" style="margin-top: 40px;">

2. **Shape_Predictor_68_Face_Landmarks - Face Detection**
   - Monitors **eye movement direction** and **head orientation** to detect cheating behavior.

<img src="assets/Left.png" align="center" width="576" height="360" style="margin-top: 40px;">
   
<img src="assets/NoFace.png" align="center" width="576" height="360" style="margin-top: 40px;">
   
<img src="assets/Right.png" align="center" width="576" height="360" style="margin-top: 40px;">

---

### ⚠️ Possible Limitations/Risks

- **Narrow Object Detection**: Limited object recognition capabilities.
- **Limited Accuracy**: Potential operational errors in identifying cheating behavior.
- **Device Usage Limitation**: Limitations on compatible devices and hardware.

---

### 📚 References

- Facial landmarks with dlib: [https://pyimagesearch.com](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/)
- YOLO - Object Detection: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- OpenCV for face detection: [OpenCV Documentation](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html)
