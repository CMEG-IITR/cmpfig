import csv
import json
import re
from collections import defaultdict


def extract_img_name(path):
    """Extract image key normalised to integer string, e.g. img001 → '1'."""
    m = re.search(r'img(\d+)\.jpg', str(path))
    return str(int(m.group(1))) if m else None


def extract_labels(label_str):
    """Return sorted tuple of all rectanglelabels in the annotation."""
    try:
        boxes = json.loads(label_str)
        labels = []
        for box in boxes:
            labels.extend(box.get("rectanglelabels", []))
        return tuple(sorted(labels))
    except Exception:
        return ()


def read_csv(filepath):
    data = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = extract_img_name(row['image'])
            if img:
                data[img] = extract_labels(row['label'])
    return data


def cohen_kappa(y1, y2):
    """Compute Cohen's kappa given two lists of categorical labels."""
    assert len(y1) == len(y2), "Label lists must be same length"
    n = len(y1)
    categories = sorted(set(y1) | set(y2))

    # Observed agreement
    po = sum(a == b for a, b in zip(y1, y2)) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        p1 = y1.count(cat) / n
        p2 = y2.count(cat) / n
        pe += p1 * p2

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return kappa


def main():
    f1 = 'alloy_project_01.csv'
    f2 = 'alloy_project_02.csv'

    data1 = read_csv(f1)
    data2 = read_csv(f2)

    common = sorted(set(data1) & set(data2))
    print(f"File 1 images : {len(data1)}")
    print(f"File 2 images : {len(data2)}")
    print(f"Common images : {len(common)}")

    if not common:
        print("No common images found — cannot compute kappa.")
        return

    y1, y2 = [], []
    print("\n{'Image':<12} {'File1 labels':<35} {'File2 labels':<35} {'Match'}")
    print("-" * 90)
    for img in common:
        l1 = data1[img]
        l2 = data2[img]
        match = "YES" if l1 == l2 else "NO"
        print(f"{img:<12} {str(l1):<35} {str(l2):<35} {match}")
        y1.append(l1)
        y2.append(l2)

    kappa = cohen_kappa(y1, y2)
    agreed = sum(a == b for a, b in zip(y1, y2))

    print("\n" + "=" * 60)
    print(f"Total compared images : {len(common)}")
    print(f"Agreed                : {agreed}")
    print(f"Disagreed             : {len(common) - agreed}")
    print(f"Observed agreement Po : {agreed / len(common):.4f}")
    print(f"Cohen's Kappa         : {kappa:.4f}")

    # Interpretation
    if kappa < 0:
        interp = "Poor (worse than chance)"
    elif kappa < 0.20:
        interp = "Slight"
    elif kappa < 0.40:
        interp = "Fair"
    elif kappa < 0.60:
        interp = "Moderate"
    elif kappa < 0.80:
        interp = "Substantial"
    else:
        interp = "Almost Perfect"
    print(f"Interpretation        : {interp}")

    # Breakdown by label type
    print("\n--- Disagreements detail ---")
    for img, l1, l2 in zip(common, y1, y2):
        if l1 != l2:
            print(f"  {img}: File1={l1}  |  File2={l2}")


if __name__ == "__main__":
    main()
