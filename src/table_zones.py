
import cv2
import json
import numpy as np
from pathlib import Path

# ---------------- CONFIGURATION ----------------

VIDEO_PATH = "data/videos/restaurant.mp4"
OUTPUT_PATH = "outputs/table_zones.json"

# -----------------------------------------------

tables = {}
current_points = []
current_table_id = 1

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Could not read the first video frame.")

height, width = frame.shape[:2]

print(f"Video resolution: {width}x{height}")


def mouse_callback(event, x, y, flags, param):
    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        print(f"Point added: ({x}, {y})")


def draw_label(image, text, center, color):
    """Draw a high-contrast label centered at a table's polygon centroid."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x = int(center[0] - text_width / 2)
    y = int(center[1] + text_height / 2)
    padding = 6

    # Keep the complete label background inside the video frame.
    x = max(padding, min(x, image.shape[1] - text_width - padding))
    y = max(text_height + padding, min(y, image.shape[0] - baseline - padding))

    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(
        image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA
    )


def draw_annotations():
    display = frame.copy()

    # Draw completed table polygons
    for table_id, points in tables.items():
        polygon = np.array(points, dtype=np.int32)

        cv2.polylines(
            display,
            [polygon],
            True,
            (0, 255, 0),
            3
        )

        center = np.mean(polygon, axis=0).astype(int)

        draw_label(display, table_id, center, (0, 255, 0))

    # Draw current polygon
    if current_points:
        points = np.array(current_points, dtype=np.int32)

        for point in current_points:
            cv2.circle(
                display,
                tuple(point),
                6,
                (0, 0, 255),
                -1
            )

        if len(current_points) >= 2:
            cv2.polylines(
                display,
                [points],
                False,
                (0, 0, 255),
                2
            )

    instructions = [
        "LEFT CLICK: Add polygon point",
        "N: Finish current table",
        "U: Undo last point",
        "R: Reset current polygon",
        "S: Save all tables",
        "Q: Quit"
    ]

    for i, instruction in enumerate(instructions):
        cv2.putText(
            display,
            instruction,
            (20, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    return display


# Create window
window_name = "Restaurant Table Zone Annotation"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, mouse_callback)

print("\n--- TABLE ANNOTATION STARTED ---")
print("Click around the boundary of each table.")
print("Press N when you finish drawing a table.\n")

while True:

    display = draw_annotations()
    cv2.imshow(window_name, display)

    key = cv2.waitKey(20) & 0xFF

    # Finish current table
    if key == ord("n"):

        if len(current_points) >= 3:

            table_id = f"T{current_table_id:02d}"
            tables[table_id] = current_points.copy()

            print(f"{table_id} saved with {len(current_points)} points.")

            current_points = []
            current_table_id += 1

        else:
            print("A table needs at least 3 points.")

    # Undo last point
    elif key == ord("u"):

        if current_points:
            current_points.pop()
            print("Last point removed.")

    # Reset current polygon
    elif key == ord("r"):

        current_points = []
        print("Current polygon reset.")

    # Save all polygons
    elif key == ord("s"):

        if current_points:
            print("Finish the current polygon by pressing N first.")
            continue

        if not tables:
            print("No table polygons to save.")
            continue

        Path(OUTPUT_PATH).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "video": VIDEO_PATH,
            "frame_width": width,
            "frame_height": height,
            "tables": tables
        }

        with open(OUTPUT_PATH, "w") as file:
            json.dump(data, file, indent=4)

        print(f"\nTable zones saved to {OUTPUT_PATH}")
        break

    # Quit
    elif key == ord("q"):
        print("Annotation cancelled.")
        break


cv2.destroyAllWindows()
