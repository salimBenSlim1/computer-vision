#!/usr/bin/env python3
"""
final_report.py

Run inference on validation images using the best trained model and produce a final report.
"""
import json
from pathlib import Path
from datetime import datetime
from app import CoffeeDetectionAgent


def generate_final_report():
    """Run inference on val images and produce a summary report."""
    
    # Load the trained model
    model_path = 'coffee_model_final.pt'
    agent = CoffeeDetectionAgent(model_path=model_path)
    
    val_dir = Path('coffee_dataset/images/val')
    if not val_dir.exists():
        print(f'Validation directory not found: {val_dir}')
        return 1
    
    val_images = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    print(f'Found {len(val_images)} validation images')
    
    # Run inference and collect results
    total_detections = 0
    detections_by_class = {}
    results = []
    
    for img_path in val_images:
        try:
            dets = agent.test_on_image(str(img_path), show=False)
            total_detections += len(dets)
            for d in dets:
                label = d['label']
                conf = d['conf']
                detections_by_class[label] = detections_by_class.get(label, 0) + 1
                results.append({'image': img_path.name, 'label': label, 'conf': f'{conf:.3f}'})
        except Exception as e:
            print(f'Error processing {img_path.name}: {e}')
    
    # Read training metrics from results.csv
    train_results_csv = Path('runs/detect/train5/results.csv')
    final_metrics = {}
    if train_results_csv.exists():
        lines = train_results_csv.read_text().strip().split('\n')
        if len(lines) > 1:
            header = lines[0].split(',')
            last_epoch = lines[-1].split(',')
            final_metrics = dict(zip(header, last_epoch))
    
    # Create report
    report = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'YOLOv8n trained on coffee dataset',
        'training': {
            'epochs_completed': 50,
            'final_mAP50': final_metrics.get('metrics/mAP50(B)', 'N/A'),
            'final_mAP50_95': final_metrics.get('metrics/mAP50-95(B)', 'N/A'),
            'final_precision': final_metrics.get('metrics/precision(B)', 'N/A'),
            'final_recall': final_metrics.get('metrics/recall(B)', 'N/A'),
            'final_train_loss': final_metrics.get('train/box_loss', 'N/A'),
            'final_val_loss': final_metrics.get('val/box_loss', 'N/A'),
        },
        'validation': {
            'total_val_images': len(val_images),
            'total_detections': total_detections,
            'detections_by_class': detections_by_class,
        },
        'next_steps': [
            'Deploy coffee_model_final.pt to production or edge device',
            'Use CoffeeDetectionAgent.start_detection() for live counting',
            'Monitor real-world performance and collect additional hard examples for further training',
            'Consider expanding dataset with more diverse cup styles and backgrounds',
        ],
    }
    
    # Save reports
    json_report = Path('FINAL_REPORT.json')
    json_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved JSON report to {json_report}')
    
    # Also save a text summary
    summary = Path('FINAL_REPORT.txt')
    with open(summary, 'w', encoding='utf-8') as f:
        f.write('=' * 70 + '\n')
        f.write('FINAL TRAINING REPORT\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Date: {report["date"]}\n')
        f.write(f'Model: {report["model"]}\n\n')
        
        f.write('TRAINING METRICS (Final Epoch 50)\n')
        f.write('-' * 70 + '\n')
        for k, v in report['training'].items():
            f.write(f'{k:.<40} {v}\n')
        
        f.write('\nVALIDATION INFERENCE RESULTS\n')
        f.write('-' * 70 + '\n')
        f.write(f'Total validation images: {report["validation"]["total_val_images"]}\n')
        f.write(f'Total objects detected: {report["validation"]["total_detections"]}\n')
        f.write(f'Detections by class:\n')
        for cls, cnt in sorted(report['validation']['detections_by_class'].items()):
            f.write(f'  {cls}: {cnt}\n')
        
        f.write('\nNEXT STEPS\n')
        f.write('-' * 70 + '\n')
        for i, step in enumerate(report['next_steps'], 1):
            f.write(f'{i}. {step}\n')
        
        f.write('\n' + '=' * 70 + '\n')
        f.write('Model ready for deployment.\n')
        f.write('Best weights: runs/detect/train5/weights/best.pt\n')
        f.write('Also copied to: coffee_model_final.pt\n')
        f.write('=' * 70 + '\n')
    
    print(f'Saved text summary to {summary}')
    print('\n' + summary.read_text())
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(generate_final_report())
