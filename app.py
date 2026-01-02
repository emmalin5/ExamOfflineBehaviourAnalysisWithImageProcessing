from typing import Tuple

import av
import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoTransformerBase, WebRtcMode, webrtc_streamer

st.set_page_config(page_title="Cheating Detection", layout="wide")
st.title("Cheating Detection (Streamlit)")

st.markdown(
    "Use your webcam to monitor face presence and head direction. The app flags potential cheating when no face is seen, multiple faces appear, or the head looks away from center."
)

DEFAULT_YAW_THRESHOLD = 0.06


@st.cache_resource(show_spinner=False)
def load_models():
    """Load dlib face detector and landmark predictor once per session."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor


def analyze_frame(
    frame: np.ndarray, detector: dlib.fhog_object_detector, predictor: dlib.shape_predictor, yaw_threshold: float
) -> Tuple[np.ndarray, str, bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    num_faces = len(faces)
    cheating_flag = False

    if num_faces == 0:
        cheat_status = "No face detected - possible cheating"
        cheating_flag = True
    elif num_faces > 1:
        cheat_status = "Multiple faces detected - possible cheating"
        cheating_flag = True
    else:
        cheat_status = "One face detected"

    for idx, face in enumerate(faces):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        nose_tip = landmarks.part(30).x
        left_eye = sum(landmarks.part(n).x for n in range(36, 42)) // 6
        right_eye = sum(landmarks.part(n).x for n in range(42, 48)) // 6

        mid_eye = (left_eye + right_eye) // 2
        face_width = max(1, x2 - x1)
        yaw_ratio = (nose_tip - mid_eye) / face_width

        if yaw_ratio < -yaw_threshold:
            direction = "Looking Left"
            color = (0, 0, 255)
            cheating_flag = True
        elif yaw_ratio > yaw_threshold:
            direction = "Looking Right"
            color = (0, 0, 255)
            cheating_flag = True
        else:
            direction = "Looking Forward"
            color = (0, 255, 255)

        label = f"Face {idx + 1}: {direction}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if cheating_flag:
        cheat_status += " - cheating"
        status_color = (0, 0, 255)
    else:
        status_color = (0, 255, 0)

    cv2.putText(frame, cheat_status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
    return frame, cheat_status, cheating_flag


class CheatingDetector(VideoTransformerBase):
    def __init__(self):
        self.detector, self.predictor = load_models()
        self.cheat_status = "Waiting for camera..."
        self.cheating_flag = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        yaw_threshold = st.session_state.get("yaw_threshold", DEFAULT_YAW_THRESHOLD)
        annotated, status, flag = analyze_frame(img, self.detector, self.predictor, yaw_threshold)
        self.cheat_status = status
        self.cheating_flag = flag
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


st.sidebar.header("Controls")
st.session_state.setdefault("yaw_threshold", DEFAULT_YAW_THRESHOLD)
st.session_state["yaw_threshold"] = st.sidebar.slider(
    "Yaw threshold (sensitivity)",
    min_value=0.02,
    max_value=0.18,
    value=st.session_state["yaw_threshold"],
    step=0.01,
)

rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

st.caption("Status appears on the video overlay. Stop the camera with the control bar above.")

webrtc_streamer(
    key="cheating-detector",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=CheatingDetector,
)
