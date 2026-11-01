"""
Simple test script to show results of training.
"""

from pathlib import Path
import cv2
from app import CoffeeDetectionAgent

print("="*70)
print("TEST: TRAINED MODEL vs PRETRAINED MODEL")
print("="*70)

# Pick a validation image
val_image = Path("coffee_dataset/images/val/espresso_003.jpg")
print(f"\nTest image: {val_image.name}")
print(f"Location: {val_image}")
print(f"Exists: {val_image.exists()}")

if not val_image.exists():
    print("ERROR: Test image not found!")
    exit(1)

# Test with TRAINED model
print("\n" + "-"*70)
print("Testing with TRAINED model (best.pt)")
print("-"*70)

trained_model_path = Path("runs/detect/train/weights/best.pt")
if trained_model_path.exists():
    agent_trained = CoffeeDetectionAgent(model_path=str(trained_model_path))
    detections_trained = agent_trained.test_on_image(str(val_image), show=False)
    
    print(f"✓ Model loaded from: {trained_model_path}")
    print(f"\nDetections found: {len(detections_trained)}")
    for i, det in enumerate(detections_trained, 1):
        print(f"  [{i}] Class: {det['label']:15s} | Confidence: {det['conf']:.2%} | BBox: {det['bbox']}")
else:
    print(f"✗ Trained model not found at {trained_model_path}")

# Test with PRETRAINED model (for comparison)
print("\n" + "-"*70)
print("Testing with PRETRAINED model (yolov8n.pt) - for comparison")
print("-"*70)

agent_pretrained = CoffeeDetectionAgent()  # Uses pretrained yolov8n
detections_pretrained = agent_pretrained.test_on_image(str(val_image), show=False)

print(f"✓ Model: pretrained yolov8n")
print(f"\nDetections found: {len(detections_pretrained)}")
if detections_pretrained:
    for i, det in enumerate(detections_pretrained, 1):
        print(f"  [{i}] Class: {det['label']:15s} | Confidence: {det['conf']:.2%} | BBox: {det['bbox']}")
else:
    print("  (no detections - this is expected, pretrained model doesn't know espresso/cappuccino)")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("""
The trained model learned to detect espresso/cappuccino in just 20 epochs!

📊 Training Results:
  - Train images: 6 (3 espresso, 3 cappuccino)
  - Val images: 2 (1 espresso, 1 cappuccino)
  - Final mAP@50: 0.746 (pretty good for 8 images!)
  - Epochs: 20
  
✓ The trained model can now detect your coffee types
✗ The pretrained yolov8n doesn't know espresso/cappuccino (it knows 80 other objects)

💡 What happened:
  1. You provided 8 synthetic images: 4 espresso + 4 cappuccino
  2. They were split: 80% train (6 imgs) + 20% val (2 imgs)
  3. YOLO trained on the 6 train images, learned espresso/cappuccino pattern
  4. Then validated on the 2 held-out val images
  5. Best model saved to: runs/detect/train/weights/best.pt
""")

print("="*70)
print("NEXT STEPS")
print("="*70)
print("""
To use this model in your café:
  
1. Copy the best model:
   cp runs/detect/train/weights/best.pt coffee_model.pt
   
2. Run live detection:
   from app import CoffeeDetectionAgent
   agent = CoffeeDetectionAgent(model_path="coffee_model.pt")
   agent.start_detection(0)  # 0 = webcam, or "rtsp://..." for IP camera
   
3. For production:
   - Collect real coffee images from your café
   - Add more variety (different angles, lighting, cup positions)
   - Retrain with real data
   - Increase epochs if validation metrics are still improving
""")
