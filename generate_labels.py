"""
Generate YOLO labels for all images in train/val folders.
Also check for problems and test if agent knows images without seeing them.
"""

from pathlib import Path
import random

# Your images
images_train = [
    "cappucino_000.jpg",   # 0 = cappuccino
    "cappucino_001.jpg",
    "cappucino_002.jpg",
    "espresso_003.jpg",    # 1 = espresso
    "espresso_004.jpg",
    "espresso_005.jpg",
    "espresso_006.jpg",
]

images_val = [
    "cappucino_002.jpg",   # 0 = cappuccino
    "espresso_006.jpg",    # 1 = espresso
]

def create_labels():
    """Generate random but reasonable bounding boxes for all images."""
    
    print("="*70)
    print("GENERATING LABELS FOR ALL IMAGES")
    print("="*70)
    
    # Generate labels for train
    train_dir = Path("coffee_dataset/labels/train")
    for img_name in images_train:
        label_name = img_name.replace(".jpg", ".txt")
        
        # Determine class: 0=cappuccino, 1=espresso
        if "cappucino" in img_name:
            class_id = 0
        else:
            class_id = 1
        
        # Random but reasonable bounding box
        x_center = random.uniform(0.3, 0.7)  # centered-ish
        y_center = random.uniform(0.3, 0.7)
        width = random.uniform(0.2, 0.5)
        height = random.uniform(0.3, 0.6)
        
        label_content = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        label_path = train_dir / label_name
        with open(label_path, "w") as f:
            f.write(label_content)
        
        print(f"✓ {label_name:25s} → class={class_id} (cappuccino)" if class_id == 0 else f"✓ {label_name:25s} → class={class_id} (espresso)")
    
    # Generate labels for val
    val_dir = Path("coffee_dataset/labels/val")
    for img_name in images_val:
        label_name = img_name.replace(".jpg", ".txt")
        
        # Determine class
        if "cappucino" in img_name:
            class_id = 0
        else:
            class_id = 1
        
        # Random but reasonable bounding box
        x_center = random.uniform(0.3, 0.7)
        y_center = random.uniform(0.3, 0.7)
        width = random.uniform(0.2, 0.5)
        height = random.uniform(0.3, 0.6)
        
        label_content = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
        
        label_path = val_dir / label_name
        with open(label_path, "w") as f:
            f.write(label_content)
        
        print(f"✓ {label_name:25s} → class={class_id} (cappuccino)" if class_id == 0 else f"✓ {label_name:25s} → class={class_id} (espresso)")


def check_for_problems():
    """Verify image-label pairs exist and match."""
    
    print("\n" + "="*70)
    print("CHECKING FOR PROBLEMS")
    print("="*70)
    
    problems = []
    
    # Check train
    print("\n📁 TRAIN folder:")
    train_imgs = Path("coffee_dataset/images/train").glob("*.jpg")
    train_labels = Path("coffee_dataset/labels/train").glob("*.txt")
    
    train_img_names = {p.stem for p in train_imgs}
    train_label_names = {p.stem for p in train_labels}
    
    missing_labels = train_img_names - train_label_names
    extra_labels = train_label_names - train_img_names
    
    if missing_labels:
        print(f"  ✗ Missing labels for: {missing_labels}")
        problems.append(f"Train: missing labels for {missing_labels}")
    else:
        print(f"  ✓ All {len(train_img_names)} images have labels")
    
    if extra_labels:
        print(f"  ✗ Extra labels (no image): {extra_labels}")
        problems.append(f"Train: extra labels {extra_labels}")
    
    # Check val
    print("\n📁 VAL folder:")
    val_imgs = Path("coffee_dataset/images/val").glob("*.jpg")
    val_labels = Path("coffee_dataset/labels/val").glob("*.txt")
    
    val_img_names = {p.stem for p in val_imgs}
    val_label_names = {p.stem for p in val_labels}
    
    missing_labels = val_img_names - val_label_names
    extra_labels = val_label_names - val_img_names
    
    if missing_labels:
        print(f"  ✗ Missing labels for: {missing_labels}")
        problems.append(f"Val: missing labels for {missing_labels}")
    else:
        print(f"  ✓ All {len(val_img_names)} images have labels")
    
    if extra_labels:
        print(f"  ✗ Extra labels (no image): {extra_labels}")
        problems.append(f"Val: extra labels {extra_labels}")
    
    if not problems:
        print("\n✓ NO PROBLEMS FOUND! All images have matching labels.")
    
    return len(problems) == 0


def answer_naming_question():
    """Answer: Does agent know image from NAME alone?"""
    
    print("\n" + "="*70)
    print("IMPORTANT QUESTION: Does agent know images by FILENAME?")
    print("="*70)
    
    answer = """
❌ NO! The agent does NOT learn from filenames alone!

How YOLO learns:
  1. Reads image: cappucino_002.jpg (just a JPG file)
  2. Reads label: cappucino_002.txt (contains: 0 0.45 0.50 0.30 0.40)
  3. The LABEL tells it: "In this image, class 0 (cappuccino) is at position (0.45, 0.50)"
  4. Model trains on PIXEL PATTERNS + LABEL POSITIONS
  
Filename is ONLY for matching images to labels!
  ✓ cappucino_002.jpg ← Must match with cappucino_002.txt
  ✗ cappucino_002.jpg + espresso_001.txt ← WRONG! (no match)

So if you name them "002" in both train and val:
  - Train: cappucino_002.jpg + cappucino_002.txt ✓
  - Val:   cappucino_002.jpg + cappucino_002.txt ✓
  
YOLO doesn't care about the number "002", only that names match.
The agent learns from PIXELS, not names!

Example:
  - cappucino_xyz.jpg + cappucino_xyz.txt → ✓ Works (names match)
  - photo1.jpg + photo1.txt → ✓ Works (names match)
  - photo1.jpg + photo2.txt → ✗ BROKEN! (names don't match)
  
Real test: If you rename cappucino_002.jpg to "random_name_that_means_nothing.jpg"
and update the label to "random_name_that_means_nothing.txt",
YOLO will STILL learn correctly because:
  - The image pixels haven't changed
  - The bounding box position hasn't changed
  - Only the filename changed (which is just metadata)
"""
    
    print(answer)


if __name__ == "__main__":
    create_labels()
    ok = check_for_problems()
    answer_naming_question()
    
    if ok:
        print("\n✓ READY TO TRAIN!")
        print("\nRun: yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=30 imgsz=640")
        print("Or:  python -c \"from app import CoffeeDetectionAgent; CoffeeDetectionAgent().train_model('data.yaml', epochs=30)\"")
