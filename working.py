import cv2
import dlib

# Initialize dlib's face detector and the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start the webcam
cap = cv2.VideoCapture(0)

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

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Facial landmarks
            landmarks = predictor(gray, face)
            nose_tip = landmarks.part(30).x
            left_eye = sum(landmarks.part(n).x for n in range(36, 42)) // 6
            right_eye = sum(landmarks.part(n).x for n in range(42, 48)) // 6

            # Yaw estimate: nose relative to eye midpoint, normalized by face width
            mid_eye = (left_eye + right_eye) // 2
            face_width = max(1, x2 - x1)
            yaw_ratio = (nose_tip - mid_eye) / face_width

            yaw_threshold = 0.06  # tweak higher (0.08) if still too sensitive
            if yaw_ratio < -yaw_threshold:
                direction = "Looking Left"
                cheating = True
            elif yaw_ratio > yaw_threshold:
                direction = "Looking Right"
                cheating = True
            else:
                direction = "Looking Forward"
                cheating = False

            status_color = (0, 0, 255) if cheating else (0, 255, 255)
            label = f"Face {idx + 1}: {direction}"
            if cheating:
                label += " - possible cheating"
            cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
