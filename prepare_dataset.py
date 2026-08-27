from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


SOURCE = Path("archive")
OUTPUT = Path("construction_helmet_yolo")

IMAGE_DIR = SOURCE / "images"
ANNOTATION_DIR = SOURCE / "annotations"

# Original XML class -> YOLO class ID
CLASS_MAPPING = {
    "helmet": 0,
    "head": 1,
}

CLASS_NAMES = {
    0: "helmet",
    1: "no_helmet",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
RANDOM_SEED = 42


def voc_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """Convert Pascal VOC coordinates to normalized YOLO coordinates."""
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return x_center, y_center, box_width, box_height


def convert_annotation(xml_path):
    root = ET.parse(xml_path).getroot()

    width = float(root.findtext("size/width"))
    height = float(root.findtext("size/height"))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {xml_path}")

    yolo_labels = []

    for obj in root.findall("object"):
        original_class = obj.findtext("name", "").strip()

        # Skip the poorly represented 'person' class
        if original_class not in CLASS_MAPPING:
            continue

        class_id = CLASS_MAPPING[original_class]
        box = obj.find("bndbox")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid bounding box: {xml_path}")

        x, y, w, h = voc_to_yolo(
            xmin, ymin, xmax, ymax, width, height
        )

        yolo_labels.append(
            f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        )

    return yolo_labels


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Rename or remove it before rerunning."
        )

    image_files = sorted(IMAGE_DIR.glob("*.png"))

    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

    # Check every image has a matching XML file
    for image_path in image_files:
        xml_path = ANNOTATION_DIR / f"{image_path.stem}.xml"

        if not xml_path.exists():
            raise FileNotFoundError(
                f"Missing annotation for {image_path.name}"
            )

    # Reproducible random split
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(image_files)

    total = len(image_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:],
    }

    for split_name, split_images in splits.items():
        output_images = OUTPUT / "images" / split_name
        output_labels = OUTPUT / "labels" / split_name

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            xml_path = ANNOTATION_DIR / f"{image_path.stem}.xml"
            labels = convert_annotation(xml_path)

            shutil.copy2(
                image_path,
                output_images / image_path.name
            )

            label_path = output_labels / f"{image_path.stem}.txt"
            label_path.write_text(
                "\n".join(labels),
                encoding="utf-8"
            )

        print(f"{split_name}: {len(split_images)} images")

    yaml_content = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: no_helmet
"""

    (OUTPUT / "data.yaml").write_text(
        yaml_content,
        encoding="utf-8"
    )

    print("\nDataset preparation completed.")
    print(f"Output: {OUTPUT.resolve()}")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()