import cv2
import sys

def open_camera(camera_index: int = 0):
    indices_to_try = [camera_index, 0, 1, 2]
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    for idx in indices_to_try:
        for backend in backends:
            cap_try = cv2.VideoCapture(idx, backend)
            if cap_try.isOpened():
                cap_try.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap_try.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap_try
            cap_try.release()
    return None

def webcam_preview(camera_index: int = 0):
    cap = open_camera(camera_index)
    if cap is None:
        print("[Camera] Could not open camera. Enable camera permissions for your terminal/VS Code in System Settings → Privacy & Security → Camera.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Camera] Frame read failed. Exiting.")
            break
        cv2.imshow("Preview", frame)
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
    webcam_preview(cam)
