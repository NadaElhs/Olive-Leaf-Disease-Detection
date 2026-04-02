import os
import shutil
import random

# Vrais noms de classes
classes = {
    'healthy': 0,
    'aculus_olearius': 1,
    'olive_peacock_spot': 2
}

# Mapping noms réels dans les dossiers (gère la majuscule de Healthy)
folder_names = {
    'train': {
        'healthy': 'healthy',
        'aculus_olearius': 'aculus_olearius',
        'olive_peacock_spot': 'olive_peacock_spot'
    },
    'test': {
        'healthy': 'Healthy',  # majuscule dans test
        'aculus_olearius': 'aculus_olearius',
        'olive_peacock_spot': 'olive_peacock_spot'
    }
}

input_base = 'CNN_olive_dataset'
output_base = 'yolo_dataset'

def copy_images(img_dir, class_id, class_name, out_img_dir, out_lbl_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    if not os.path.exists(img_dir):
        print(f"  ✗ Non trouvé : {img_dir}")
        return 0

    images = [f for f in os.listdir(img_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_name in images:
        src = f"{img_dir}/{img_name}"
        new_name = f"{class_name}_{img_name}"
        shutil.copy(src, f"{out_img_dir}/{new_name}")

        label_name = new_name.rsplit('.', 1)[0] + '.txt'
        with open(f"{out_lbl_dir}/{label_name}", 'w') as f:
            f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

    return len(images)

# ── TRAIN + génération VAL (20% du train) ──────────────────────────
print("\nTraitement TRAIN → TRAIN + VAL...")
for class_name, class_id in classes.items():
    folder = folder_names['train'][class_name]
    img_dir = f"{input_base}/train/{folder}"

    if not os.path.exists(img_dir):
        print(f"  ✗ Non trouvé : {img_dir}")
        continue

    images = [f for f in os.listdir(img_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.seed(42)
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    # Train
    for img_name in train_imgs:
        src = f"{img_dir}/{img_name}"
        new_name = f"{class_name}_{img_name}"
        out_img = f"{output_base}/images/train"
        out_lbl = f"{output_base}/labels/train"
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        shutil.copy(src, f"{out_img}/{new_name}")
        with open(f"{out_lbl}/{new_name.rsplit('.', 1)[0]}.txt", 'w') as f:
            f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

    # Val
    for img_name in val_imgs:
        src = f"{img_dir}/{img_name}"
        new_name = f"{class_name}_{img_name}"
        out_img = f"{output_base}/images/val"
        out_lbl = f"{output_base}/labels/val"
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)
        shutil.copy(src, f"{out_img}/{new_name}")
        with open(f"{out_lbl}/{new_name.rsplit('.', 1)[0]}.txt", 'w') as f:
            f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

    print(f"  ✓ {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

# ── TEST ───────────────────────────────────────────────────────────
print("\nTraitement TEST...")
for class_name, class_id in classes.items():
    folder = folder_names['test'][class_name]
    img_dir = f"{input_base}/test/{folder}"
    count = copy_images(
        img_dir, class_id, class_name,
        f"{output_base}/images/test",
        f"{output_base}/labels/test"
    )
    print(f"  ✓ {class_name}: {count} images")

# ── data.yaml ─────────────────────────────────────────────────────
yaml_content = """path: ./yolo_dataset
train: images/train
val: images/val
test: images/test

nc: 3
names:
  - healthy
  - aculus_olearius
  - olive_peacock_spot
"""

with open('yolo_dataset/data.yaml', 'w') as f:
    f.write(yaml_content)

print("\n✓ data.yaml créé !")
print("✓ Conversion complète !")

# ── Résumé ─────────────────────────────────────────────────────────
print("\n── Résumé du dataset ──")
for split in ['train', 'val', 'test']:
    path = f"{output_base}/images/{split}"
    if os.path.exists(path):
        count = len(os.listdir(path))
        print(f"  {split}: {count} images")