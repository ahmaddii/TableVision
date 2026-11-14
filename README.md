# Restaurant Table Occupancy Detection

An experimental computer vision project that analyzes restaurant CCTV footage to detect and track people, then estimate which tables are available or occupied.

The prototype combines YOLO26 pose detection, ByteTrack tracking, and manually drawn table regions to turn video into an annotated occupancy view.

![Restaurant occupancy detection demo](assets/occupancy-demo.png)

## What it does

- Detects people and estimates body pose with YOLO26 Pose.
- Tracks people across video frames with ByteTrack.
- Maps people to manually annotated table polygons.
- Estimates seated guests and table occupancy over time.
- Displays estimated people counts and exports an annotated video.
- Supports staff filtering, ignore zones, and temporal confirmation to reduce short-lived false assignments.

> **Status:** Experimental prototype under active development. Occupancy and customer counts are estimates and are not guaranteed to be accurate in every scene.

## How it works

```text
CCTV video → YOLO26 Pose → ByteTrack → pose and movement analysis
           → table-zone assignment → occupancy state → annotated video
```

Tables are defined as polygons for a specific camera view. The occupancy logic uses proximity to these zones, pose cues, movement history, and confirmation/exit timers to estimate whether guests are seated.

## Project layout

```text
.
├── assets/                 # README demo image
├── data/
│   ├── images/             # Demo footage
│   └── videos/             # Test videos
├── outputs/                # Table-zone JSON and generated video
├── src/
│   ├── detector.py         # Person detection
│   ├── occupancy.py        # Occupancy pipeline and configuration
│   ├── table_zones.py      # Table polygon annotation
│   ├── tracker.py          # Tracking helpers
│   └── video.py            # Camera/video input experiment
├── yolo26s-pose.pt         # Pose model weights
└── yolo26n.pt              # Detection model weights
```

## Requirements

- Python 3.10 or newer
- A supported OS with OpenCV display support for the interactive annotation tool
- CPU for basic runs; CUDA-capable NVIDIA GPU recommended for faster inference
- Dependencies: `ultralytics`, `opencv-python`, `numpy`, and `torch` (plus tracker dependencies required by your Ultralytics setup)

If the repository has a `requirements.txt` in your checkout, install from it. Otherwise install the core packages:

```bash
python -m pip install ultralytics opencv-python numpy torch lap
```

## Run the project

Clone the repository and create an environment:

```bash
git clone https://github.com/ahmaddii/TableVision
cd Restaurent-Table-Detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install the dependencies using the instructions above. Make sure the model weights and input video are available at the paths configured by the scripts.

### 1. Configure table zones

Open `src/occupancy.py` and set `ZONES_PATH` to the zone file for the camera view you want to process. The current default is `outputs/table_zones_1.json`.

To create or update polygon annotations, configure the video and output paths in `src/table_zones.py`, then run:

```bash
python src/table_zones.py
```

Save the zones to the JSON path selected in the occupancy configuration. Each camera view needs its own calibrated zones; changing camera position or zoom generally requires drawing them again.

### 2. Configure and process a video

In `src/occupancy.py`, check `MODEL_PATH`, `ZONES_PATH`, and `OUTPUT_PATH`. The default output is `outputs/restaurant_occupancy.mp4`. The input path and processing settings are also configured in this script.

Run:

```bash
python src/occupancy.py
```

The annotated video is written to the configured output path. Open the result with a video player such as VLC.

## Occupancy states

- **Available:** no confirmed seated guests are currently assigned to the table.
- **Occupied:** one or more guests have met the configured seating confirmation logic.
- **Checking:** the state is being evaluated or is waiting for its confirmation/exit timer.

The exact state transitions and timing thresholds are configurable in `src/occupancy.py`.

## Limitations

- Occlusion by people, tables, and furniture can reduce detection and counting accuracy.
- Camera angle, perspective, lighting, and image quality affect zone assignment and pose estimates.
- Staff can be mistaken for guests, and guest/staff filtering depends on the scene and configuration.
- Detection results and tracking IDs can change between runs.
- Performance depends on input resolution and hardware.
- Generated or sample footage is not a substitute for evaluation on representative restaurant video.

## Roadmap

- [ ] Improve person-to-table association and seated-person classification.
- [ ] Support multiple camera views through clearer configuration.
- [ ] Evaluate against authorized, representative restaurant footage.
- [ ] Add real-time camera stream support and occupancy analytics.

## Privacy and responsible use

Use camera footage only with the appropriate authorization and safeguards. Follow applicable privacy requirements, limit access to video and derived data, and evaluate the system carefully before relying on its estimates for operational decisions.

## Author

**Ahmad** — Computer Vision, Machine Learning, and Software Engineering

Feedback and contributions are welcome.
