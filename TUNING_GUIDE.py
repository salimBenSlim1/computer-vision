"""
🔧 Configuration Guide for Reducing Multiple Cup Detections

The issue: Model detects the same cup multiple times in consecutive frames.
Solution: Tune these parameters based on your café environment.
"""

from app import CoffeeDetectionAgent, DetectionConfig

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              🔧 TUNING COFFEE DETECTION TO REDUCE OVERCOUNTING              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 CURRENT PARAMETERS (after update):
═══════════════════════════════════════════════════════════════════════════════

confidence_threshold: 0.5
  → Detection threshold: only show boxes with ≥50% confidence
  → REDUCE if missing cups (0.3-0.4)
  → INCREASE if too many false positives (0.6-0.7)

min_confidence_to_count: 0.6
  → Counting threshold: only COUNT boxes with ≥60% confidence
  → INCREASE to be more strict (0.7-0.8)
  → DECREASE to count more (0.5)

cooldown_seconds: 3.0
  → After counting a cup, wait 3 seconds before counting same cup again
  → INCREASE if still getting duplicates (5.0-10.0)
  → DECREASE if cups stay too long (1.0-2.0)

max_distance_pixels: 150
  → If cup center moves <150px between frames, it's the same cup
  → INCREASE if cups close together (200-250)
  → DECREASE if one cup counted as different (100-120)

grid_cell_size: 100
  → NOT actively used, for future optimization
  → Leave at 100


═══════════════════════════════════════════════════════════════════════════════

🎯 RECOMMENDATIONS FOR YOUR CAFÉ
═══════════════════════════════════════════════════════════════════════════════

SCENARIO 1: "Detecting same cup 2-3 times per second"
──────────────────────────────────────────────────────
Increase cooldown significantly:

  config = DetectionConfig(
      confidence_threshold=0.5,
      min_confidence_to_count=0.65,
      cooldown_seconds=8.0,        # ← Increased from 3.0
      max_distance_pixels=150
  )
  agent = CoffeeDetectionAgent(config=config)
  agent.start_detection(0)


SCENARIO 2: "Missing some cups, but no overcounting"
─────────────────────────────────────────────────────
Relax confidence thresholds:

  config = DetectionConfig(
      confidence_threshold=0.4,    # ← Decreased from 0.5
      min_confidence_to_count=0.5, # ← Decreased from 0.6
      cooldown_seconds=3.0,
      max_distance_pixels=150
  )
  agent = CoffeeDetectionAgent(config=config)
  agent.start_detection(0)


SCENARIO 3: "Cups very close together, still overcounting"
─────────────────────────────────────────────────────────
Tighten distance and increase cooldown:

  config = DetectionConfig(
      confidence_threshold=0.5,
      min_confidence_to_count=0.65,
      cooldown_seconds=5.0,        # ← Increased
      max_distance_pixels=80       # ← Decreased (stricter)
  )
  agent = CoffeeDetectionAgent(config=config)
  agent.start_detection(0)


═══════════════════════════════════════════════════════════════════════════════

📈 TUNING WORKFLOW
═══════════════════════════════════════════════════════════════════════════════

1. Start with recommended config for your scenario above

2. Run live detection:
   python -c "from app import CoffeeDetectionAgent, DetectionConfig; \\
              config = DetectionConfig(cooldown_seconds=8.0, min_confidence_to_count=0.65); \\
              agent = CoffeeDetectionAgent(config=config); \\
              agent.start_detection(0)"

3. Test with real cups:
   - Pour a cup
   - Watch it get counted (should count ONCE)
   - Press 'c' in console to see tracking info
   - Check daily report

4. If still overcounting:
   - INCREASE cooldown_seconds (add 2-3 seconds)
   - INCREASE min_confidence_to_count (add 0.05)

5. If missing cups:
   - DECREASE confidence_threshold (subtract 0.05)
   - DECREASE min_confidence_to_count (subtract 0.05)


═══════════════════════════════════════════════════════════════════════════════

🔍 MONITORING (while running)
═══════════════════════════════════════════════════════════════════════════════

While detection is running:
  - Press 'c' to see console output with tracked objects
  - Output shows: "Current counts: {'espresso': 5, ...} | Tracked objects: 2"

Tracked objects = cups currently being "watched" (not yet expired from cooldown)

If you see:
  ✓ Tracked objects: 0-2 → Good, not overcounting
  ✗ Tracked objects: 10+ → Possible issue, may need longer cooldown


═══════════════════════════════════════════════════════════════════════════════

✅ RECOMMENDED STARTING POINT FOR YOU
═══════════════════════════════════════════════════════════════════════════════

Based on "detecting cups so many times", try this:

  from app import CoffeeDetectionAgent, DetectionConfig
  
  config = DetectionConfig(
      confidence_threshold=0.5,      # Standard
      min_confidence_to_count=0.65,  # Strict counting
      cooldown_seconds=6.0,          # Wait 6 sec before counting same cup
      max_distance_pixels=120        # Strict spatial matching
  )
  
  agent = CoffeeDetectionAgent(config=config)
  agent.start_detection(0)
  
  # Press 'c' to monitor tracking
  # Press 'q' to quit

Then test with real cups in your café!

═══════════════════════════════════════════════════════════════════════════════

💡 TIPS
═══════════════════════════════════════════════════════════════════════════════

• Cooldown_seconds = "amnesia time" for the model
  - If set to 3s: model forgets a cup exists after 3 seconds
  - If set to 10s: model remembers for 10 seconds

• min_confidence_to_count = quality filter
  - Higher = only count high-confidence detections
  - Lower = count more, but might get false positives

• Real problem = small dataset (9 images)
  - Model hasn't seen café lighting/angles
  - Collect 50+ real café images and retrain
  - This will improve accuracy more than tuning

• Retrain command (when you have more data):
  python -c "from app import CoffeeDetectionAgent; \\
             CoffeeDetectionAgent().train_model('data.yaml', epochs=50)"

═══════════════════════════════════════════════════════════════════════════════

Questions? Check README_FINAL.md for more details.
Good luck! ☕
""")
