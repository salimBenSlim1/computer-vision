#!/usr/bin/env python3
"""
Quick test runner for different detection configurations.
Choose a preset and it will run with recommended parameters.
"""

from app import CoffeeDetectionAgent, DetectionConfig

PRESETS = {
    "balanced": {
        "cooldown_seconds": 4.0,
        "min_confidence_to_count": 0.6,
        "max_distance_pixels": 150,
        "description": "Default balanced configuration"
    },
    "strict": {
        "cooldown_seconds": 8.0,
        "min_confidence_to_count": 0.7,
        "max_distance_pixels": 120,
        "description": "Strict - fewer false positives, might miss some"
    },
    "aggressive": {
        "cooldown_seconds": 3.0,
        "min_confidence_to_count": 0.5,
        "max_distance_pixels": 200,
        "description": "Aggressive - count more, higher chance of duplicates"
    },
    "sensitive": {
        "cooldown_seconds": 2.0,
        "min_confidence_to_count": 0.4,
        "max_distance_pixels": 200,
        "description": "Very sensitive - catches almost everything"
    },
}

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("☕ COFFEE DETECTION - PRESET CONFIGURATIONS")
    print("="*70)
    
    print("\nAvailable presets:\n")
    for name, config in PRESETS.items():
        print(f"  {name:15} - {config['description']}")
        print(f"                   cooldown={config['cooldown_seconds']}s, conf={config['min_confidence_to_count']}")
    
    print("\n" + "-"*70)
    
    if len(sys.argv) > 1:
        preset_name = sys.argv[1].lower()
        if preset_name not in PRESETS:
            print(f"\n❌ Unknown preset: {preset_name}")
            print(f"\nUsage: python test_config.py [balanced|strict|aggressive|sensitive]")
            sys.exit(1)
        
        preset = PRESETS[preset_name]
        print(f"\n✓ Using preset: {preset_name}")
        print(f"  Description: {preset['description']}")
        print(f"  Parameters:")
        print(f"    - cooldown_seconds: {preset['cooldown_seconds']}")
        print(f"    - min_confidence_to_count: {preset['min_confidence_to_count']}")
        print(f"    - max_distance_pixels: {preset['max_distance_pixels']}")
        print(f"\nStarting camera... (press 'q' to quit, 'c' for stats)\n")
        print("-"*70 + "\n")
        
        config = DetectionConfig(
            cooldown_seconds=preset['cooldown_seconds'],
            min_confidence_to_count=preset['min_confidence_to_count'],
            max_distance_pixels=preset['max_distance_pixels'],
        )
        agent = CoffeeDetectionAgent(config=config)
        agent.start_detection(0)
        
        print("\n" + "-"*70)
        print("✓ Detection stopped")
        print(f"Final counts: {agent.snapshot()}")
        print(f"Report saved: reports/coffee_report_*.json")
    else:
        print("\n📋 RECOMMENDATIONS:\n")
        print("  IF overcounting (same cup multiple times):")
        print("    → python test_config.py strict")
        print("")
        print("  IF missing cups (not detecting all):")
        print("    → python test_config.py sensitive")
        print("")
        print("  FOR production (balanced):")
        print("    → python test_config.py balanced")
        print("")
        print("  TO experiment (aggressive):")
        print("    → python test_config.py aggressive")
        print("\n" + "="*70 + "\n")
