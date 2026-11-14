
from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open video source (0 = laptop camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open video source")

while True:
    success, frame = cap.read()

    if not success:
        break

    # Run object detection
    results = model.predict(
        frame,
        classes=[0],  # Person only
        conf=0.5,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    cv2.imshow("Restaurant - Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
