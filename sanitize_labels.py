from pathlib import Path
import sys

def sanitize_labels(labels_root='coffee_dataset/labels', data_yaml='data.yaml'):
    # Read num classes from data.yaml
    nc = None
    try:
        p = Path(data_yaml)
        lines = p.read_text(encoding='utf-8').splitlines()
        names_started = False
        max_idx = -1
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith('names:'):
                names_started = True
                continue
            if names_started:
                if ':' not in s:
                    break
                key, _ = s.split(':', 1)
                try:
                    idx = int(key.strip())
                    if idx > max_idx:
                        max_idx = idx
                except Exception:
                    continue
        if max_idx >= 0:
            nc = max_idx + 1
    except Exception:
        pass

    if nc is None:
        print('Unable to parse number of classes from', data_yaml)
        return 1

    print(f'Parsed num_classes={nc} from {data_yaml}. Scanning labels under {labels_root}...')

    root = Path(labels_root)
    changed = 0
    total_bad_lines = 0
    for subset in ('train', 'val'):
        folder = root / subset
        if not folder.exists():
            continue
        for f in folder.glob('*.txt'):
            lines = f.read_text(encoding='utf-8').splitlines()
            new_lines = []
            bad = 0
            for ln in lines:
                if not ln.strip():
                    continue
                parts = ln.split()
                try:
                    cls = int(parts[0])
                except Exception:
                    # non-numeric class, drop line
                    bad += 1
                    continue
                if cls < 0 or cls >= nc:
                    bad += 1
                else:
                    new_lines.append(ln)
            if bad > 0:
                total_bad_lines += bad
                changed += 1
                f.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
                print(f'Cleaned {f} — removed {bad} invalid lines')

    print(f'Done. Files modified: {changed}. Total invalid label lines removed: {total_bad_lines}')
    return 0


if __name__ == '__main__':
    sys.exit(sanitize_labels())
