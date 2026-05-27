from ultralytics import YOLO
import cv2

model = YOLO("best_openvino_model")

cap = cv2.VideoCapture("test_vid.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, device="cpu")

    annotated = results[0].plot()
    cv2.imshow("Gate Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
