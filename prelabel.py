"""
prelabel.py

Run the current model over all images in the dataset and write YOLO-format label files
from predictions. This accelerates manual labeling (human must still verify).

Usage:
  python prelabel.py --conf 0.5 --subset train

This will create/overwrite files under `coffee_dataset/labels/<subset>/*.txt`.
"""
from pathlib import Path
import argparse
from app import CoffeeDetectionAgent


def load_name_to_id(data_yaml_path: str = "data.yaml"):
    """Parse a minimal data.yaml and return name -> id mapping.

    Falls back to an empty dict if file missing or parsing fails.
    """
    mapping = {}
    p = Path(data_yaml_path)
    if not p.exists():
        return mapping

    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        in_names = False
        for ln in lines:
            stripped = ln.strip()
            if not stripped:
                continue
            if stripped.startswith("names:"):
                in_names = True
                continue
            if in_names:
                # Expect lines like '0: espresso' or '  0: espresso'
                if ":" not in stripped:
                    break
                key, val = stripped.split(":", 1)
                try:
                    idx = int(key.strip())
                    name = val.strip()
                    # remove optional quotes
                    if name.startswith(('"', "'")) and name.endswith(('"', "'")):
                        name = name[1:-1]
                    mapping[name] = idx
                except Exception:
                    # stop parsing names if unexpected format
                    continue
    except Exception:
        return {}

    return mapping


def image_paths(subset: str):
    base = Path("coffee_dataset") / "images" / subset
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])


def write_yolo_label(path: Path, detections):
    # Each detection: {'bbox': (x1,y1,x2,y2), 'cls_id': int, 'conf': conf}
    # Need to convert to normalized YOLO format: class_id x_c y_c w h
    lines = []
    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        h = abs(y2 - y1)
        w = abs(x2 - x1)
        x_c = x1 + w / 2.0
        y_c = y1 + h / 2.0
        img = d.get('_img_shape')
        if img is None:
            continue
        ih, iw = img
        # normalize
        xcn = x_c / iw
        ycn = y_c / ih
        wn = w / iw
        hn = h / ih
        cls_id = d.get('cls_id')
        if cls_id is None:
            continue
        lines.append(f"{cls_id} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--subset", choices=["train", "val", "all"], default="train")
    args = p.parse_args()

    agent = CoffeeDetectionAgent()
    model = agent._model

    # Build mapping from dataset names -> class id (0..N-1)
    name_to_id = load_name_to_id('data.yaml')

    subsets = [args.subset] if args.subset != "all" else ["train", "val"]
    for s in subsets:
        imgs = image_paths(s)
        label_dir = Path("coffee_dataset") / "labels" / s
        label_dir.mkdir(parents=True, exist_ok=True)
        for img_path in imgs:
            results = model(str(img_path), conf=args.conf)
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names.get(cls_id, str(cls_id))
                # Only include detections that map to our dataset names and allowed classes
                if name in agent.config.allowed_classes and name in name_to_id and conf >= args.conf:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    ih, iw = results[0].orig_shape[:2]
                    detections.append({'bbox': (float(x1), float(y1), float(x2), float(y2)), 'cls_id': int(name_to_id[name]), 'conf': conf, '_img_shape': (ih, iw)})

            label_path = label_dir / (img_path.stem + ".txt")
            write_yolo_label(label_path, detections)
            print(f"Wrote {label_path} ({len(detections)} objects)")


if __name__ == '__main__':
    main()
