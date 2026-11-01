# Computer Vision: Coffee Detection Agent

A Python project for detecting configured items in camera or video streams. It uses Ultralytics YOLO for object detection and a lightweight tracker to follow detections across frames, reduce repeat counts, and write session counts to daily JSON and CSV reports.

> **Status:** Experimental. Detection quality depends on the trained model, class mapping, camera position, and lighting. This release contains source code only; it does not include training images or model weights.

## Features

- Detect objects in a still image with `CoffeeDetectionAgent.test_on_image()`.
- Process a camera, video file, or stream with `CoffeeDetectionAgent.start_detection()`.
- Track detections between frames and count newly created tracks.
- Read current in-memory counts with `snapshot()`.
- Write a JSON and CSV count report when a live detection session ends.
- Prepare, train, and inspect YOLO-format datasets with the included helper scripts.

## Requirements

- Python 3.10 or newer is recommended.
- A working webcam for live camera use. OpenCV must support the chosen camera or stream.
- A custom YOLO model trained for the classes you intend to detect.

Install the primary dependencies:

```bash
python -m pip install ultralytics opencv-python numpy PyYAML requests
```

The tracker can use SciPy and FilterPy when installed, and falls back to simpler matching when they are unavailable:

```bash
python -m pip install scipy filterpy
```

## Configure a model and dataset

Place your trained model at `models/best.pt` and create a `data.yaml` beside `app.py`. The order of the class names must match the zero-based class indices used when training the model. Keep indices contiguous; do not leave gaps.

Example for two classes:

```yaml
path: ./dataset
train: images/train
val: images/val
names:
  0: cup
  1: espresso
```

This public release excludes local datasets and model files. Use images and weights that you have permission to use, and check the relevant framework and model license terms.

**Important:** If `models/best.pt` is missing, the current code falls back to the generic `yolov8n.pt` model. That model is not trained for your custom classes, and the fallback class-name mapping can make predictions misleading. Add and verify a compatible custom model before relying on the detections or counts.

## Detect objects in an image

```python
from app import CoffeeDetectionAgent

agent = CoffeeDetectionAgent(
    model_path="models/best.pt",
    data_yaml="data.yaml",
)

for detection in agent.test_on_image("path/to/image.jpg"):
    print(detection["label"], detection["conf"], detection["bbox"])
```

Each detection contains a `label`, confidence score (`conf`), and bounding box (`bbox`). Set `show=True` to display an annotated image; this requires a graphical desktop.

## Run a camera or video source

Run the default camera (index `0`):

```bash
python app.py
```

Or select a camera, video path, or stream URL accepted by OpenCV:

```python
from app import CoffeeDetectionAgent

agent = CoffeeDetectionAgent(model_path="models/best.pt", data_yaml="data.yaml")
agent.start_detection(source=0)
print(agent.snapshot())
```

Press `q` in the detection window to stop. Press `c` to log the current counts. On shutdown, reports are written to `reports/coffee_report_YYYY-MM-DD.json` and `.csv`. Counts reset when the process restarts, and a later session on the same date overwrites that date's report files.

## Train or retrain

Prepare a YOLO-format dataset and `data.yaml`, then run:

```bash
python retrain.py --data data.yaml --epochs 50 --imgsz 640
```

Optional arguments include `--device cpu` or a GPU index such as `--device 0`, and `--resume`. The current training helper logs exceptions instead of returning a failing exit status, so a successful process exit does not prove training completed. Review the output and resulting weights yourself.

Several dataset helpers rewrite label files or move files. Back up your dataset and review each script before running it.

## Included source

The release includes the root-level Python source scripts, including the detection agent, tracker, training/data-preparation helpers, and manual demonstration scripts. The `test_*.py` files are manual scripts that may need a camera or local sample data; they are not an automated test suite.

The repository does not include the nested `coffai-entr` application, local datasets or augmented images, model weights, training runs, reports, caches, generated files, local backups, or the machine-specific training output directory.

## Known limitations

- There is no automated test suite, dependency lockfile, or measured accuracy benchmark.
- Tracking uses image-space detections; camera movement and overlapping objects can affect counts.
- Counts are held in memory and the daily report is a session snapshot, not a persistent daily total.
- The checked-in local dataset configuration was not included. Create and validate your own `data.yaml`, ensuring the class order matches the model.
- No `LICENSE` file is included. Public visibility alone does not grant permission to reuse or redistribute the code; choose and add a license before inviting reuse.

## License

No license has been selected for this project yet.
