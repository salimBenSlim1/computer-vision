"""
Generate synthetic espresso/cappuccino images, create YOLO dataset, train model.
This is a complete end-to-end example for beginners.
"""

import cv2
import numpy as np
from pathlib import Path
import json
from app import CoffeeDetectionAgent

# ============================================================================
# STEP 1: Generate synthetic training images
# ============================================================================

def generate_synthetic_image(coffee_type: str, image_id: int) -> tuple:
    """
    Generate a synthetic image of a coffee cup.
    Returns (image_array, bbox_normalized) where bbox is (class_id, x_center, y_center, width, height).
    
    coffee_type: 'espresso' or 'cappuccino'
    """
    # Create blank image (white background)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Random noise/texture
    noise = np.random.randint(200, 256, (480, 640, 3), dtype=np.uint8)
    img = cv2.addWeighted(img, 0.8, noise, 0.2, 0)
    
    # Add some café background elements (blurred shapes)
    cv2.circle(img, (100, 100), 40, (200, 180, 150), -1)
    cv2.rectangle(img, (450, 50), (600, 150), (220, 200, 180), -1)
    
    # Cup color and position (randomize a bit)
    cup_x = np.random.randint(200, 400)
    cup_y = np.random.randint(150, 350)
    cup_width = np.random.randint(80, 150)
    cup_height = np.random.randint(100, 180)
    
    if coffee_type == "espresso":
        # Dark brown cup with small amount of coffee
        cv2.ellipse(img, (cup_x, cup_y), (cup_width // 2, cup_height // 3), 0, 0, 360, (100, 80, 60), -1)
        cv2.ellipse(img, (cup_x, cup_y), (cup_width // 2, cup_height // 4), 0, 0, 360, (50, 40, 30), -1)
        # Handle
        cv2.ellipse(img, (cup_x + cup_width // 2 + 20, cup_y), (25, 40), 0, 0, 360, (150, 120, 100), 3)
        class_id = 0  # espresso
    else:  # cappuccino
        # Cup with more milk (lighter color)
        cv2.ellipse(img, (cup_x, cup_y), (cup_width // 2, cup_height // 3), 0, 0, 360, (180, 150, 120), -1)
        cv2.ellipse(img, (cup_x, cup_y), (cup_width // 2, cup_height // 4), 0, 0, 360, (200, 180, 160), -1)
        # Foam on top (white)
        cv2.ellipse(img, (cup_x, cup_y - cup_height // 5), (cup_width // 2 + 10, 20), 0, 0, 360, (245, 245, 245), -1)
        # Handle
        cv2.ellipse(img, (cup_x + cup_width // 2 + 20, cup_y), (25, 40), 0, 0, 360, (200, 170, 140), 3)
        class_id = 1  # cappuccino
    
    # Normalize bbox: (x_center, y_center, width, height) in [0, 1]
    x_center = cup_x / 640
    y_center = cup_y / 480
    width = cup_width / 640
    height = cup_height / 480
    
    bbox = (class_id, x_center, y_center, width, height)
    return img, bbox


def create_minimal_dataset():
    """Create folder structure and generate 8 images: 4 espresso, 4 cappuccino."""
    
    base = Path("coffee_dataset")
    (base / "images" / "train").mkdir(parents=True, exist_ok=True)
    (base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (base / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print("✓ Folder structure created")
    
    # Generate images
    images_data = []
    img_id = 0
    
    # 4 espresso images (80% train, 20% val = 3 train, 1 val)
    for i in range(4):
        img, bbox = generate_synthetic_image("espresso", i)
        split = "train" if i < 3 else "val"
        img_name = f"espresso_{i:03d}.jpg"
        img_path = base / "images" / split / img_name
        cv2.imwrite(str(img_path), img)
        
        # Write label file
        label_path = base / "labels" / split / img_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        images_data.append((img_name, split, "espresso", bbox))
        print(f"  ✓ {split}/{img_name}")
    
    # 4 cappuccino images (80% train, 20% val = 3 train, 1 val)
    for i in range(4):
        img, bbox = generate_synthetic_image("cappuccino", i)
        split = "train" if i < 3 else "val"
        img_name = f"cappuccino_{i:03d}.jpg"
        img_path = base / "images" / split / img_name
        cv2.imwrite(str(img_path), img)
        
        # Write label file
        label_path = base / "labels" / split / img_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")
        
        images_data.append((img_name, split, "cappuccino", bbox))
        print(f"  ✓ {split}/{img_name}")
    
    print(f"\n✓ Generated 8 images: 6 train, 2 val")
    return images_data


# ============================================================================
# STEP 2: Create data.yaml
# ============================================================================

def create_data_yaml():
    """Write data.yaml pointing to train/val folders."""
    yaml_content = """train: ./coffee_dataset/images/train
val: ./coffee_dataset/images/val
names:
  0: espresso
  1: cappuccino
"""
    with open("data.yaml", "w") as f:
        f.write(yaml_content)
    print("✓ Created data.yaml")
    print("  Content:")
    for line in yaml_content.strip().split("\n"):
        print(f"    {line}")


# ============================================================================
# STEP 3: Train model
# ============================================================================

def train():
    """Train YOLOv8 nano on the minimal dataset."""
    print("\n" + "="*70)
    print("TRAINING YOLOV8 ON MINIMAL DATASET")
    print("="*70)
    
    agent = CoffeeDetectionAgent()
    try:
        agent.train_model("data.yaml", epochs=20, imgsz=640)
        print("\n✓ Training completed!")
        print("  Best model saved to: runs/train/exp/weights/best.pt")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("  Make sure you have ultralytics installed:")
        print("    python -m pip install -U ultralytics")


# ============================================================================
# STEP 4: Test on validation image
# ============================================================================

def test_results():
    """Test the trained model on a validation image."""
    print("\n" + "="*70)
    print("TESTING RESULTS")
    print("="*70)
    
    val_image = Path("coffee_dataset/images/val/espresso_003.jpg")
    if not val_image.exists():
        print(f"✗ Validation image not found: {val_image}")
        return
    
    # Load the trained model
    trained_model_path = Path("runs/train/exp/weights/best.pt")
    if not trained_model_path.exists():
        print(f"✗ Trained model not found. Did training complete successfully?")
        return
    
    # Test with trained model
    agent = CoffeeDetectionAgent(model_path=str(trained_model_path))
    detections = agent.test_on_image(str(val_image), show=False)
    
    print(f"\n✓ Detections on {val_image.name}:")
    if detections:
        for i, det in enumerate(detections, 1):
            print(f"  [{i}] {det['label']:15s} confidence={det['conf']:.2%} bbox={det['bbox']}")
    else:
        print("  (no detections)")
    
    # Also test with pretrained model for comparison
    print(f"\n✓ Comparison: pretrained yolov8n on same image:")
    agent_pretrained = CoffeeDetectionAgent()
    detections_pretrained = agent_pretrained.test_on_image(str(val_image), show=False)
    if detections_pretrained:
        for i, det in enumerate(detections_pretrained, 1):
            print(f"  [{i}] {det['label']:15s} confidence={det['conf']:.2%} bbox={det['bbox']}")
    else:
        print("  (no detections)")


# ============================================================================
# STEP 5: Show what's in train vs val
# ============================================================================

def show_folder_contents():
    """Explain what's in train and val."""
    print("\n" + "="*70)
    print("WHAT'S IN TRAIN VS VAL FOLDERS")
    print("="*70)
    
    base = Path("coffee_dataset")
    
    print("\n📁 TRAIN folder (used to teach the model):")
    train_images = list((base / "images" / "train").glob("*.jpg"))
    for img in sorted(train_images):
        label_file = base / "labels" / "train" / img.name.replace(".jpg", ".txt")
        with open(label_file) as f:
            label_content = f.read().strip()
        print(f"  {img.name:25s} → label: {label_content}")
    
    print(f"\n  Total: {len(train_images)} images (model learns from these)")
    
    print("\n📁 VAL folder (used to test if model learned correctly):")
    val_images = list((base / "images" / "val").glob("*.jpg"))
    for img in sorted(val_images):
        label_file = base / "labels" / "val" / img.name.replace(".jpg", ".txt")
        with open(label_file) as f:
            label_content = f.read().strip()
        print(f"  {img.name:25s} → label: {label_content}")
    
    print(f"\n  Total: {len(val_images)} images (model tests on these, never seen during training)")
    
    print("\n📊 LABEL FORMAT (each line: class_id x_center y_center width height, all normalized [0,1]):")
    print("   class_id: 0=espresso, 1=cappuccino")
    print("   x_center, y_center: center of box as fraction of image width/height")
    print("   width, height: box size as fraction of image width/height")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("MINIMAL COFFEE DETECTION TRAINING EXAMPLE")
    print("="*70)
    
    # Step 1: Generate images
    print("\n[1/5] Generating synthetic images...")
    images_data = create_minimal_dataset()
    
    # Step 2: Show what's in train vs val
    show_folder_contents()
    
    # Step 3: Create data.yaml
    print("\n[2/5] Creating data.yaml...")
    create_data_yaml()
    
    print("\n[3/5] Ready to train. Run:")
    print("  Option A (CLI):")
    print("    yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=20 imgsz=640")
    print("\n  Option B (Python):")
    print("    python -c \"from prepare_and_train import train; train()\"")
    
    print("\n[4/5] Starting training (this may take 1-5 minutes)...")
    train()
    
    print("\n[5/5] Testing results on validation image...")
    test_results()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
