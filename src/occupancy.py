import cv2
import json
import os
import sys
import time
import numpy as np
import torch

from collections import defaultdict, deque
from ultralytics import YOLO


# ---------------- CONFIGURATION ----------------

ZONES_PATH = "outputs/table_zones_1.json"
OUTPUT_PATH = "outputs/restaurant_occupancy.mp4"

# Pose model separates seated guests from standing people much better.
# Set USE_POSE = False and MODEL_PATH = "yolo26s.pt" for a plain detector.
MODEL_PATH = "yolo26s-pose.pt"
USE_POSE = True

# ---- Speed settings ----
IMG_SIZE = 800               # 640 = fastest, 960+ = most accurate for small people
DEVICE = 0 if torch.cuda.is_available() else "cpu"
HALF = DEVICE != "cpu"       # FP16 on GPU is much faster

# Run detection on every Nth frame and skip the rest (video keeps real speed).
# 1 = every frame, 2 = half the work, 3 = a third. Tracking/timers adapt.
DETECT_EVERY_N = 2

CONFIDENCE = 0.30
TRACKER = "bytetrack.yaml"   # tip: copy it and set track_buffer: 60

# ---- Timing (seconds) ----
# Waiters usually stop at a table for a few seconds (order, food, bill).
# A guest must look seated for this long before being counted.
SEAT_CONFIRM_SEC = 5.0
EXIT_DELAY_SEC = 3.0         # table frees after nobody confirmed for this long
STABILITY_WINDOW_SEC = 1.0   # window used for the "stationary" test
CANDIDATE_GAP_SEC = 0.6      # tolerated dropout before the seat timer resets
COUNT_SMOOTH_SEC = 1.0       # smoothing for the displayed people count
HANDOFF_SEC = 2.0            # window to pass a guest's status to a new track ID
HANDOFF_DIST_RATIO = 0.8     # max distance (x box height) for an ID handoff

# ---- Waiter / staff filtering ----
# 1) Anyone who stops at 2+ different tables without being a confirmed guest
#    is treated as staff.
STAFF_VISIT_LIMIT = 2
# 2) Optional: polygons (in ORIGINAL frame coordinates) where only staff
#    stand, e.g. behind the counter / kitchen door. Anyone seen inside one
#    is staff forever. Example:
#    STAFF_ZONES = [[[700, 200], [1000, 200], [1000, 290], [700, 290]]]
STAFF_ZONES = []

# ---- False-positive filtering (posters, menu boards, statues...) ----
# Polygons in ORIGINAL frame coordinates. Any "person" whose box centre or
# bottom-centre falls inside one is dropped completely (no box, no ID).
# Get the coordinates with:  python occupancy_detection.py pick
# Example: IGNORE_ZONES = [[[20, 240], [140, 240], [140, 520], [20, 520]]]
IGNORE_ZONES = [
    [
        [1, 245],
        [366, 100],
        [372, 176],
        [13, 381],
    ]
]

# Automatic filter: a track that never moves AND never goes near a table for
# this many seconds is treated as a fake person (poster/picture) and hidden.
# Set 0 to disable.
GHOST_FILTER_SEC = 20

# ---- Seating zone = table polygon expanded by a margin ----
SEAT_MARGIN_RATIO = 0.45     # of sqrt(table area)
MIN_MARGIN_FRAME_RATIO = 0.04
MAX_MARGIN_FRAME_RATIO = 0.10

# ---- Stationary test (relative to the person's box height) ----
MOVE_RATIO = 0.20
MIN_MOVE_PX = 8

# ---- Seated tests ----
MAX_SEATED_ASPECT = 1.95     # fallback only (weak)
SEATED_KNEE_RATIO = 0.55     # |knee_y - hip_y| / torso below this = seated
STANDING_KNEE_RATIO = 0.80   # above this = standing
KP_CONF = 0.40

# Shows pose (S seated, T standing, ? unknown) and still/moving per person
DEBUG_LABELS = True

# ------------------------------------------------


def load_zones():
    with open(ZONES_PATH, "r") as file:
        return json.load(file)


def scale_points(points, scale_x, scale_y):
    return np.array(
        [[x * scale_x, y * scale_y] for x, y in points],
        dtype=np.float32
    )


def build_tables(zone_data, scale_x, scale_y, width):
    """Scale polygons to the video size and compute each table's seat margin."""

    tables = {}

    for table_id, points in zone_data["tables"].items():

        poly = scale_points(points, scale_x, scale_y)

        area = cv2.contourArea(poly)

        margin = SEAT_MARGIN_RATIO * np.sqrt(max(area, 1.0))
        margin = float(np.clip(
            margin,
            MIN_MARGIN_FRAME_RATIO * width,
            MAX_MARGIN_FRAME_RATIO * width
        ))

        tables[table_id] = {
            "contour": poly.reshape(-1, 1, 2),
            "draw": poly.astype(np.int32),
            "margin": margin,
        }

    return tables


def assign_table(point, tables):
    """Nearest table whose seating zone contains the point."""

    best_table = None
    best_dist = -float("inf")

    for table_id, table in tables.items():

        # >0 inside polygon, <0 distance outside
        dist = cv2.pointPolygonTest(
            table["contour"],
            (float(point[0]), float(point[1])),
            True
        )

        if dist >= -table["margin"] and dist > best_dist:
            best_dist = dist
            best_table = table_id

    return best_table


def in_any_zone(point, contours):
    for contour in contours:
        if cv2.pointPolygonTest(
            contour,
            (float(point[0]), float(point[1])),
            False
        ) >= 0:
            return True
    return False


def is_stationary(history, min_len):
    """history items: (cx, cy, box_height)."""

    if len(history) < min_len:
        return False

    data = np.array(history, dtype=np.float32)

    centers = data[:, :2]
    heights = data[:, 2]

    threshold = max(MIN_MOVE_PX, MOVE_RATIO * float(np.median(heights)))

    center = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - center, axis=1)

    return float(np.percentile(distances, 90)) <= threshold


def pose_seated(kpts):
    """
    kpts: (17, 3) array [x, y, conf].
    Returns True (seated), False (standing) or None (unknown).
    """

    def visible(indices):
        return [
            kpts[i][:2] for i in indices
            if kpts[i][2] >= KP_CONF
        ]

    shoulders = visible((5, 6))
    hips = visible((11, 12))
    knees = visible((13, 14))

    if not shoulders or not hips or not knees:
        return None

    shoulder = np.mean(shoulders, axis=0)
    hip = np.mean(hips, axis=0)
    knee = np.mean(knees, axis=0)

    torso = float(np.linalg.norm(shoulder - hip))

    if torso < 5:
        return None

    ratio = abs(float(knee[1] - hip[1])) / torso

    if ratio <= SEATED_KNEE_RATIO:
        return True

    if ratio >= STANDING_KNEE_RATIO:
        return False

    return None


def main():

    zone_data = load_zones()
    video_path = zone_data["video"]

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = OUTPUT_PATH
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )
    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {output_path}")

    scale_x = width / zone_data["frame_width"]
    scale_y = height / zone_data["frame_height"]

    tables = build_tables(zone_data, scale_x, scale_y, width)

    staff_contours = [
        scale_points(poly, scale_x, scale_y).reshape(-1, 1, 2)
        for poly in STAFF_ZONES
    ]

    ignore_contours = [
        scale_points(poly, scale_x, scale_y).reshape(-1, 1, 2)
        for poly in IGNORE_ZONES
    ]

    # Frame-based timings (counted in PROCESSED frames)
    eff_fps = fps / max(1, DETECT_EVERY_N)

    history_size = max(6, int(eff_fps * STABILITY_WINDOW_SEC))
    min_history = max(4, history_size // 2)
    confirm_frames = max(1, int(SEAT_CONFIRM_SEC * eff_fps))
    exit_frames = max(1, int(EXIT_DELAY_SEC * eff_fps))
    gap_frames = max(1, int(CANDIDATE_GAP_SEC * eff_fps))
    smooth_frames = max(1, int(COUNT_SMOOTH_SEC * eff_fps))
    handoff_frames = max(1, int(HANDOFF_SEC * eff_fps))
    ghost_frames = int(GHOST_FILTER_SEC * eff_fps)

    track_history = defaultdict(lambda: deque(maxlen=history_size))
    track_last_seen = {}
    track_pos = {}                     # id -> (cx, cy, box_h)
    track_tables = defaultdict(set)    # tables where an id stood still
    staff_ids = set()

    # Ghost (poster / picture) filter state
    ghost_ids = set()
    track_origin = {}          # id -> (x, y, first_frame)
    track_max_disp = {}        # id -> max distance moved from origin
    track_near_table = set()   # ids that were ever inside a table's zone

    seat_since = {}   # (table, id) -> frame when seated-look began
    seat_last = {}    # (table, id) -> last frame it looked seated

    # table -> {id: {"last": frame, "pos": (cx, cy, h)}}
    confirmed = {t: {} for t in tables}
    occupied_state = {t: False for t in tables}
    count_history = {t: deque(maxlen=smooth_frames) for t in tables}

    def is_confirmed_anywhere(track_id):
        return any(track_id in confirmed[t] for t in tables)

    frame_number = 0
    processing_started = time.perf_counter()

    print("Restaurant occupancy detection started.")
    print(f"Tables loaded: {len(tables)}")
    print(f"Output: {output_path}")

    try:
        while True:

            # Skip frames cheaply (grab = decode only, no processing)
            for _ in range(max(1, DETECT_EVERY_N) - 1):
                if not cap.grab():
                    break

            success, frame = cap.read()

            if not success or frame is None:
                print("Input video ended or a frame could not be decoded.")
                break

            frame_number += 1

            results = model.track(
                source=frame,
                persist=True,
                tracker=TRACKER,
                classes=[0],
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            # table -> {track_id: is_seated_candidate}
            zone_people = {t: {} for t in tables}

            # (box, track_id, label, color) drawn after blending
            detections = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
                and result.boxes.id is not None
            ):

                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()

                keypoints = None

                if USE_POSE and result.keypoints is not None:
                    keypoints = result.keypoints.data.cpu().numpy()

                for i, (box, track_id) in enumerate(zip(boxes, track_ids)):

                    x1, y1, x2, y2 = map(int, box)

                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # Fake people (posters, menu boards): drop completely
                    if ignore_contours and (
                        in_any_zone((cx, cy), ignore_contours)
                        or in_any_zone((cx, y2), ignore_contours)
                    ):
                        continue

                    if track_id in ghost_ids:
                        continue

                    track_last_seen[track_id] = frame_number
                    track_history[track_id].append((cx, cy, box_h))
                    track_pos[track_id] = (cx, cy, box_h)

                    # Staff zone (counter / kitchen door)
                    if staff_contours and in_any_zone((cx, cy), staff_contours):
                        staff_ids.add(track_id)

                    stationary = is_stationary(
                        track_history[track_id],
                        min_history
                    )

                    pose_state = None   # True seated, False standing, None unknown

                    if keypoints is not None and i < len(keypoints):
                        pose_state = pose_seated(keypoints[i])

                    posture = pose_state

                    if posture is None:
                        posture = (box_h / box_w) <= MAX_SEATED_ASPECT

                    table_id = assign_table((cx, cy), tables)

                    # Ghost filter: never moves and never near a table = fake
                    if table_id is not None:
                        track_near_table.add(track_id)

                    ox, oy, first_frame = track_origin.setdefault(
                        track_id, (cx, cy, frame_number)
                    )

                    track_max_disp[track_id] = max(
                        track_max_disp.get(track_id, 0.0),
                        float(np.hypot(cx - ox, cy - oy))
                    )

                    if (
                        ghost_frames > 0
                        and frame_number - first_frame >= ghost_frames
                        and track_id not in track_near_table
                        and track_max_disp[track_id] <= 0.15 * box_h
                    ):
                        ghost_ids.add(track_id)

                    if track_id in ghost_ids:
                        continue

                    # Waiter rule: stopping at 2+ tables without being a guest
                    if table_id is not None and stationary:

                        track_tables[track_id].add(table_id)

                        if (
                            len(track_tables[track_id]) >= STAFF_VISIT_LIMIT
                            and not is_confirmed_anywhere(track_id)
                        ):
                            staff_ids.add(track_id)

                    label = None
                    color = (255, 255, 0)

                    if track_id in staff_ids:

                        label = "STAFF (ignored)"
                        color = (255, 0, 255)

                    elif table_id is not None:

                        candidate = bool(posture and stationary)

                        zone_people[table_id][track_id] = candidate

                        if candidate:
                            label = "SEATED"
                            color = (0, 255, 255)
                        else:
                            label = "NEAR TABLE"

                        if DEBUG_LABELS:
                            pose_txt = {True: "S", False: "T", None: "?"}[pose_state]
                            label += (
                                f" p:{pose_txt} "
                                f"{'still' if stationary else 'moving'}"
                            )

                    detections.append(
                        ((x1, y1, x2, y2), track_id, label, color)
                    )

            # ---------- update table states ----------

            occupied_count = 0
            table_info = {}

            for table_id in tables:

                people = zone_people[table_id]

                # Seat timers (strict: seated AND stationary)
                for track_id, is_candidate in people.items():

                    if is_candidate:

                        key = (table_id, track_id)

                        if key not in seat_since:
                            seat_since[key] = frame_number

                        seat_last[key] = frame_number

                for key in [k for k in seat_since if k[0] == table_id]:

                    if frame_number - seat_last[key] > gap_frames:
                        del seat_since[key]
                        del seat_last[key]
                        continue

                    if (
                        seat_last[key] == frame_number
                        and frame_number - seat_since[key] >= confirm_frames
                    ):
                        confirmed[table_id][key[1]] = {
                            "last": frame_number,
                            "pos": track_pos[key[1]],
                        }

                # ID handoff: ByteTrack sometimes gives a seated guest a new ID
                # (e.g. after being hidden by a passing waiter). A seated, still
                # person who appears right where a confirmed guest was just lost
                # inherits that guest's status immediately.
                for track_id, is_candidate in people.items():

                    if not is_candidate or track_id in confirmed[table_id]:
                        continue

                    px, py, _ = track_pos[track_id]

                    best_old = None
                    best_dist = float("inf")

                    for old_id, info in confirmed[table_id].items():

                        if old_id in people:
                            continue

                        if frame_number - info["last"] > handoff_frames:
                            continue

                        ox, oy, oh = info["pos"]

                        dist = float(np.hypot(px - ox, py - oy))

                        if dist <= HANDOFF_DIST_RATIO * oh and dist < best_dist:
                            best_dist = dist
                            best_old = old_id

                    if best_old is not None:

                        del confirmed[table_id][best_old]

                        confirmed[table_id][track_id] = {
                            "last": frame_number,
                            "pos": track_pos[track_id],
                        }

                # Confirmed guests stay alive while they remain in the zone
                for track_id in people:

                    if track_id in confirmed[table_id]:
                        confirmed[table_id][track_id]["last"] = frame_number
                        confirmed[table_id][track_id]["pos"] = track_pos[track_id]

                # Expire guests who left
                for track_id, info in list(confirmed[table_id].items()):

                    if frame_number - info["last"] > exit_frames:
                        del confirmed[table_id][track_id]

                occupied = len(confirmed[table_id]) > 0

                if occupied_state[table_id] and not occupied:

                    for key in [k for k in seat_since if k[0] == table_id]:
                        del seat_since[key]
                        del seat_last[key]

                occupied_state[table_id] = occupied

                # Only confirmed guests are ever counted - waiters never are
                visible = [
                    track_id for track_id in people
                    if track_id in confirmed[table_id]
                ]

                count_history[table_id].append(len(visible))

                if occupied:
                    people_count = int(round(
                        float(np.median(count_history[table_id]))
                    ))
                    people_count = max(1, people_count)
                    occupied_count += 1
                else:
                    people_count = 0

                table_info[table_id] = (occupied, people_count)

            # Prune old tracks
            if frame_number % 100 == 0:

                stale = [
                    tid for tid, seen in track_last_seen.items()
                    if frame_number - seen > eff_fps * 5
                ]

                for tid in stale:
                    track_history.pop(tid, None)
                    track_last_seen.pop(tid, None)
                    track_pos.pop(tid, None)
                    track_origin.pop(tid, None)
                    track_max_disp.pop(tid, None)

            # ---------- drawing ----------

            overlay = frame.copy()

            for table_id, table in tables.items():

                occupied, _ = table_info[table_id]
                color = (0, 0, 255) if occupied else (0, 255, 0)

                cv2.fillPoly(overlay, [table["draw"]], color)

            frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

            for contour in staff_contours:
                cv2.polylines(
                    frame,
                    [contour.astype(np.int32)],
                    True,
                    (255, 0, 255),
                    1
                )

            if DEBUG_LABELS:
                for contour in ignore_contours:
                    cv2.polylines(
                        frame,
                        [contour.astype(np.int32)],
                        True,
                        (128, 128, 128),
                        1
                    )

            for table_id, table in tables.items():

                occupied, people_count = table_info[table_id]

                color = (0, 0, 255) if occupied else (0, 255, 0)
                status = "OCCUPIED" if occupied else "AVAILABLE"

                cv2.polylines(frame, [table["draw"]], True, color, 3)

                center = np.mean(table["draw"], axis=0).astype(int)

                cv2.putText(
                    frame,
                    f"{table_id}: {status}",
                    (int(center[0]) - 80, int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"People: {people_count}",
                    (int(center[0]) - 80, int(center[1]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            for (x1, y1, x2, y2), track_id, label, color in detections:

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

                if label:
                    cv2.putText(
                        frame,
                        label,
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        2
                    )

            total_tables = len(tables)
            available_count = total_tables - occupied_count

            cv2.rectangle(frame, (10, 10), (370, 110), (30, 30, 30), -1)

            cv2.putText(
                frame, f"Total Tables: {total_tables}", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            cv2.putText(
                frame, f"Occupied: {occupied_count}", (25, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
            )
            cv2.putText(
                frame, f"Available: {available_count}", (25, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )

            video_writer.write(frame)

            elapsed = time.perf_counter() - processing_started
            processing_fps = frame_number / elapsed if elapsed > 0 else 0.0
            if frame_number == 1 or frame_number % max(1, int(eff_fps)) == 0:
                if total_frames > 0:
                    source_frame = min(frame_number * max(1, DETECT_EVERY_N), total_frames)
                    progress = 100.0 * source_frame / total_frames
                    print(
                        f"Progress: {progress:5.1f}% "
                        f"({source_frame}/{total_frames} source frames), "
                        f"processing: {processing_fps:.2f} FPS",
                        flush=True
                    )
                else:
                    print(
                        f"Processed {frame_number} frames, "
                        f"processing: {processing_fps:.2f} FPS",
                        flush=True
                    )
    finally:
        video_writer.release()
        cap.release()

    elapsed = time.perf_counter() - processing_started
    processing_fps = frame_number / elapsed if elapsed > 0 else 0.0
    print(f"Processed video saved: {output_path}")
    print(
        f"Processed {frame_number} frames in {elapsed:.1f}s "
        f"({processing_fps:.2f} processing FPS)."
    )

def pick_points():
    """
    Click on the first video frame to print coordinates for IGNORE_ZONES /
    STAFF_ZONES. Left-click adds a point, R resets, Q quits.
    """

    zone_data = load_zones()

    cap = cv2.VideoCapture(zone_data["video"])
    success, base = cap.read()
    cap.release()

    if not success:
        raise RuntimeError("Cannot read the first frame.")

    points = []
    window = "Pick points (click, R = reset, Q = quit)"

    def on_click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Point: [{x}, {y}]   ->   {points}")

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    while True:

        view = base.copy()

        for p in points:
            cv2.circle(view, tuple(p), 4, (0, 0, 255), -1)

        if len(points) > 1:
            cv2.polylines(
                view,
                [np.array(points, dtype=np.int32)],
                len(points) > 2,
                (0, 255, 255),
                2
            )

        cv2.imshow(window, view)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("r"):
            points.clear()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    print("\nPaste this into IGNORE_ZONES (inside the outer list):")
    print(points)


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "pick":
        pick_points()
    else:
        main()
