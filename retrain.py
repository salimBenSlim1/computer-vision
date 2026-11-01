#!/usr/bin/env python3
"""
retrain.py

Simple wrapper to retrain the model after you've added more classes and labels.

Usage:
  python retrain.py --epochs 50 --imgsz 640

This calls the CoffeeDetectionAgent.train_model(data.yaml, ...)
"""
import argparse
from app import CoffeeDetectionAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    p.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    p.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint if available")
    p.add_argument("--device", default=None, help="Device to use for training (e.g. 'cpu' or '0')")
    args = p.parse_args()

    agent = CoffeeDetectionAgent()
    agent.train_model(args.data, epochs=args.epochs, imgsz=args.imgsz, resume=args.resume, device=args.device)

if __name__ == "__main__":
    main()
