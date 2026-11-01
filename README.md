# Computer Vision: Coffee Detection Agent

A Python project for detecting configured items in camera or video streams. It uses Ultralytics YOLO for object detection and a lightweight tracker to follow detections across frames, reduce repeat counts, and write session counts to daily JSON and CSV reports.

> **Status:** Experimental. Detection quality depends on the trained model, class mapping, camera position, and lighting. This release contains source code only; it does not include training images or model weights.

## Features

- Detect objects in a still image with `CoffeeDetectionAgent.test_on_image()`.
- Process a camera, video file, or stream with `CoffeeDetectionAgent.start_detection()`.
- Track detections between frames and count newly created tracks.
- Read current in-memory counts with `snapshot()`.
- Write a JSON and CSV count report when a live detection session ends.
- Prepare, train, and inspect YOLO-format datasets with the supplementary helper scripts.

## Requirements

- Python 3.10 or newer is recommended.
- A working webcam for live camera use. OpenCV must support the chosen camera or stream.
- A custom YOLO model trained for the classes you intend to detect.

Create an environment and install the pinned runtime dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\\Scripts\\Activate.ps1`. On macOS or Linux, use `source .venv/bin/activate`.

The tracker can use SciPy and FilterPy for Kalman filtering and optimal assignment. The default tracker still works without them:

```bash
python -m pip install -r requirements-tracking.txt
```

## Configure a model and dataset

Copy `data.yaml.example` to `data.yaml`, then edit it for your dataset. By default, the app looks for custom weights at `coffee_model_final.pt` in the project root. If you store them at `models/best.pt`, pass that path to `CoffeeDetectionAgent` as shown below. The order of class names must match the zero-based class indices used when training the model. Keep indices contiguous; the application rejects gaps instead of silently compressing them.

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

**Important:** When custom weights are missing or fail to load, the application may load generic `yolov8n.pt` weights as a training starting point. It does not relabel those generic model classes, and both inference methods refuse to run until compatible custom weights load successfully. This avoids treating generic detections as your custom classes.

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

Optional arguments include `--device cpu` or a GPU index such as `--device 0`, and `--resume`. Training errors are logged and re-raised, so failed training returns a non-zero process status. Review the output and resulting weights yourself.

Several dataset helpers rewrite label files or move files. Back up your dataset and review each script before running it.

## Code layout

The reusable core is `app.py` (detection, counting, reports), `tracker.py` (track association), and `retrain.py` (training command). `data.yaml.example`, `requirements.txt`, and `tests/` provide the starter configuration, pinned direct dependencies, and automated unit tests.

The other root-level Python files are supplementary data-preparation, reporting, tuning, or manual demo scripts. Some rewrite or move dataset files; back up your dataset and inspect a helper before running it. Three scripts with hard-coded local product classes (`get_coffee_dataset.py`, `generate_synthetic_coffee.py`, and `test_app_run.py`) are excluded. The included root-level `test_*.py` files are manual demos, not the automated suite.

The repository does not include the nested `coffai-entr` application, local datasets or augmented images, model weights, training runs, reports, caches, generated files, local backups, or the machine-specific training output directory.

## Known limitations

- Run the automated unit suite with `python -m unittest discover -s tests -v`.
- Runtime dependencies are directly pinned; pip resolves transitive dependencies, so there is no platform-specific full lockfile.
- No measured accuracy benchmark is included.
- Tracking uses image-space detections; camera movement and overlapping objects can affect counts.
- Counts are held in memory and the daily report is a session snapshot, not a persistent daily total.
- The checked-in local dataset configuration was not included. Create and validate your own `data.yaml`, ensuring the class order matches the model.
- No `LICENSE` file is included. Public visibility alone does not grant permission to reuse or redistribute the code; choose and add a license before inviting reuse.

## License

No license has been selected for this project yet.
