import cv2
import numpy as np
import os
import sys

def load_yolo():
    # Verify files exist
    required_files = ["yolov4.weights", "yolov4.cfg", "coco.names"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            f"Missing YOLO files: {', '.join(missing)}. Place them in the project root."
        )

    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    # Handle possible nested indices from getUnconnectedOutLayers
    uol = net.getUnconnectedOutLayers()
    output_layers = []
    for idx in uol:
        if isinstance(idx, (list, tuple, np.ndarray)):
            output_layers.append(layer_names[int(idx[0]) - 1])
        else:
            output_layers.append(layer_names[int(idx) - 1])
    return net, classes, output_layers

def detect_objects(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    # Flatten indexes to a simple list of ints
    if len(indexes) > 0:
        try:
            indexes = [int(i) for i in indexes.flatten()]
        except Exception:
            indexes = [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in indexes]
    else:
        indexes = []

    font = cv2.FONT_HERSHEY_PLAIN
    num_faces = 0
    phone_detected = False
    face_direction = "Forward"  # Default assumption

    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        # Use class-based color to avoid index out of range
        color = colors[class_ids[i] % len(colors)]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), font, 1, color, 1)

        if label == "person":
            num_faces += 1
            centroid_x = x + w // 2
            if centroid_x < img.shape[1] // 3:
                face_direction = "Left"
            elif centroid_x > 2 * img.shape[1] // 3:
                face_direction = "Right"
            else:
                face_direction = "Forward"
        if label == "cell phone":
            phone_detected = True

    cheat_status = "Normal - No cheating detected"
    if num_faces == 0:
        cheat_status = "No face detected - possible cheating"
    elif num_faces > 1:
        cheat_status = "Multiple faces detected - possible cheating"
    if phone_detected:
        cheat_status = "Phone detected - cheating"

    cheat_status += " | Direction: " + face_direction
    cv2.putText(img, cheat_status, (10, 50), font, 3, (0, 0, 255), 2)
    return img

def webcam_detect(camera_index: int = 0):
    model, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(camera_index)
    # Optional: reduce resolution for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        frame = draw_labels(boxes, confs, colors, class_ids, classes, frame)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam = 0
    if len(sys.argv) > 1:
        try:
            cam = int(sys.argv[1])
        except ValueError:
            pass
    webcam_detect(cam)
