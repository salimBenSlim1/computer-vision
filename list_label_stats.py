from pathlib import Path
from collections import Counter

def scan_labels(root='coffee_dataset/labels'):
    p = Path(root)
    if not p.exists():
        print(f'Labels folder not found: {root}')
        return 1

    counts = Counter()
    files_with_labels = 0
    total_files = 0
    samples = {}

    for subset in ('train', 'val'):
        folder = p / subset
        if not folder.exists():
            continue
        for f in sorted(folder.glob('*.txt')):
            total_files += 1
            lines = [ln.strip() for ln in f.read_text(encoding='utf-8').splitlines() if ln.strip()]
            if not lines:
                continue
            files_with_labels += 1
            for ln in lines:
                parts = ln.split()
                try:
                    cls = int(parts[0])
                except Exception:
                    cls = 'BAD'
                counts[cls] += 1
                if cls not in samples:
                    samples[cls] = f

    print(f'Total label files scanned: {total_files}')
    print(f'Files with at least one label: {files_with_labels}')
    print('Unique class ids and counts:')
    for cls, cnt in counts.most_common():
        print(f'  {cls}: {cnt} occurrences; example file: {samples.get(cls)}')

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(scan_labels())
