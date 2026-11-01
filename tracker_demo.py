"""
tracker_demo.py

Quick demo: run model on validation images, update tracker, print track IDs and vitesse.

Usage:
  python tracker_demo.py
"""
import time
from app import CoffeeDetectionAgent


def run_demo():
    agent = CoffeeDetectionAgent()
    model = agent._model
    val_images = [
        'coffee_dataset/images/val/cappucino_002.jpg',
        'coffee_dataset/images/val/espresso_006.jpg'
    ]

    print('Running tracker demo on validation images...')
    now = time.time()
    for img in val_images:
        print('\nImage:', img)
        results = model(img, conf=agent.config.confidence_threshold)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names.get(cls_id, str(cls_id))
            if label in agent.config.allowed_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({'bbox': (float(x1), float(y1), float(x2), float(y2)), 'label': label, 'conf': float(conf)})

        agent.tracker.update(detections, timestamp=now)
        for tr in agent.tracker.get_active_tracks():
            vx, vy, speed = tr.get_velocity()
            print(f" Track id={tr.id} label={tr.label} conf={tr.last_conf:.2f} speed={speed:.2f} vx={vx:.2f} vy={vy:.2f} counted={tr.counted}")
        now += 0.5


if __name__ == '__main__':
    run_demo()
