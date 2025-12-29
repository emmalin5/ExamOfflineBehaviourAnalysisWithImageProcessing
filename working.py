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
    faces = detector(gray)
    num_faces = len(faces)

    # Logic to determine if the scenario might be cheating
    cheating_flag = False

    if num_faces == 0:
        cheat_status = "No face detected - possible cheating"
        cheating_flag = True
    elif num_faces > 1:
        cheat_status = "Multiple faces detected - possible cheating"
        cheating_flag = True
    else:
        cheat_status = "One face detected"

    # Analyze and draw each face
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
        cv2.putText(frame, label, (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Draw this status on the frame
    if cheating_flag:
        cheat_status += " - cheating"
        status_color = (0, 0, 255)
    else:
        status_color = (0, 255, 0)

    cv2.putText(frame, cheat_status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
