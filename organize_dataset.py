import os
import json
import shutil

dataset_path = "pec"
target_train = "food_dataset/train"
target_val = "food_dataset/validation"

# Load splits
with open(os.path.join(dataset_path, "meta", "train.json"), "r") as f:
    train_data = json.load(f)
with open(os.path.join(dataset_path, "meta", "test.json"), "r") as f:
    test_data = json.load(f)

def process_split(data, target_dir):
    for class_name, image_paths in data.items():
        # Extract image IDs from paths (e.g., "churros/1061830" → "1061830")
        for image_path in image_paths:
            # Split the path and get the image ID
            image_id = image_path.split("/")[-1]
            # Source path: pec/images/<class_name>/<image_id>.jpg
            src = os.path.join(dataset_path, "images", class_name, f"{image_id}.jpg")
            # Destination folder: food_dataset/train_or_validation/<class_name>
            dest_dir = os.path.join(target_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            if os.path.exists(src):
                shutil.copy(src, dest_dir)
            else:
                print(f"Missing: {src}")

# Organize training and validation data
process_split(train_data, target_train)
process_split(test_data, target_val)

print("Dataset organized!")