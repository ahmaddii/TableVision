
import cv2

# Open the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera started. Press Q to exit.")

while True:

    # Read one frame
    success, frame = cap.read()

    if not success:
        print("Could not read frame")
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Display frame information
    cv2.putText(
        frame,
        f"Width: {width} Height: {height}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Display video
    cv2.imshow("Restaurant Vision - Experiment 01", frame)

    # Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()
