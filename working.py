import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# YOLOv4 for phone detection
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    YOLO_CLASSES = [line.strip() for line in f.readlines()]
layer_names = yolo_net.getLayerNames()
YOLO_OUTPUT_LAYERS = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
PHONE_CLASS_ID = YOLO_CLASSES.index("cell phone") if "cell phone" in YOLO_CLASSES else None

# Start the webcam (try macOS AVFoundation first, then fallback)
def open_camera():
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    for backend in backends:
        cap_try = cv2.VideoCapture(0, backend)
        if cap_try.isOpened():
            return cap_try
        cap_try.release()
    return cv2.VideoCapture(0)

cap = open_camera()
if not cap.isOpened():
    raise RuntimeError("Camera could not be opened. Check macOS camera permissions for Terminal/VS Code and ensure no other app is using the camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # upsample once to help detect multiple faces
    num_faces = len(faces)

    if num_faces == 0:
        cv2.putText(frame, "No face detected - possible cheating", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        if num_faces > 1:
            cv2.putText(frame, "Multiple faces detected - possible cheating", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        phone_detected = False

        # Phone detection via YOLO
        if PHONE_CLASS_ID is not None:
            blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
            yolo_net.setInput(blob)
            outputs = yolo_net.forward(YOLO_OUTPUT_LAYERS)

            boxes = []
            confs = []
            class_ids = []
            h, w = frame.shape[:2]
            for output in outputs:
                for detect in output:
                    scores = detect[5:]
                    class_id = int(np.argmax(scores))
                    conf = float(scores[class_id])
                    if class_id == PHONE_CLASS_ID and conf > 0.4:
                        center_x = int(detect[0] * w)
                        center_y = int(detect[1] * h)
                        bw = int(detect[2] * w)
                        bh = int(detect[3] * h)
                        x = int(center_x - bw / 2)
                        y = int(center_y - bh / 2)
                        boxes.append([x, y, bw, bh])
                        confs.append(conf)
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.4, nms_threshold=0.3)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, bw, bh = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
                    cv2.putText(frame, "Phone", (x, max(y - 5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    phone_detected = True
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predictor(gray, face)
            nose_tip = landmarks.part(30).x
            left_eye = sum(landmarks.part(n).x for n in range(36, 42)) // 6
            right_eye = sum(landmarks.part(n).x for n in range(42, 48)) // 6

            mid_eye = (left_eye + right_eye) // 2
            face_width = max(1, x2 - x1)
            yaw_ratio = (nose_tip - mid_eye) / face_width

            yaw_threshold = 0.06
            if yaw_ratio < -yaw_threshold:
                direction = "Looking Left"
                cheating_turn = True
            elif yaw_ratio > yaw_threshold:
                direction = "Looking Right"
                cheating_turn = True
            else:
                direction = "Looking Forward"
                cheating_turn = False

            status_color = (0, 0, 255) if cheating_turn else (0, 255, 255)
            label = f"Face {idx + 1}: {direction}"
            if cheating_turn:
                label += " - possible cheating"
            cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        status_msgs = []
        if num_faces == 0:
            status_msgs.append("No face detected - possible cheating")
        elif num_faces > 1:
            status_msgs.append("Multiple faces detected - possible cheating")
        else:
            status_msgs.append("One face detected")

        if phone_detected:
            status_msgs.append("Phone detected - cheating")

        status_text = " | ".join(status_msgs)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if phone_detected else (0, 255, 255), 2, cv2.LINE_AA)
