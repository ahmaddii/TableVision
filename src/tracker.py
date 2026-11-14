
from ultralytics import YOLO
import cv2

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Unable to open camera")

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detect and track people
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.5,
        verbose=False
    )

    # Display tracking IDs
    annotated_frame = results[0].plot()

    cv2.imshow("Restaurant - Person Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# using byte track to assign id so that we can say that a person in frame 1 is same as frame 2