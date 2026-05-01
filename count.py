import os
from pathlib import Path

dataset_path = Path("yolo_dataset")  # ← ajustez si nécessaire

for split in ["train", "val", "test"]:
    labels_path = dataset_path / "labels" / split
    counts = {0: 0, 1: 0, 2: 0}
    
    if not labels_path.exists():
        print(f"{split}: dossier introuvable")
        continue
        
    for txt_file in labels_path.glob("*.txt"):
        with open(txt_file) as f:
            for line in f:
                cls = int(line.strip().split()[0])
                if cls in counts:
                    counts[cls] += 1
    
    print(f"\n{split}:")
    print(f"  healthy           (0): {counts[0]} annotations")
    print(f"  aculus_olearius   (1): {counts[1]} annotations")
    print(f"  olive_peacock_spot(2): {counts[2]} annotations")