import os
import cv2

# Configuration
root_dir = r'C:\Users\avalo\repos\Object-Detection-in-Low-Light-Environments'  # Root directory path
labels_train_dir = os.path.join(root_dir, 'yolov5', 'data', 'labels', 'train')
labels_val_dir = os.path.join(root_dir, 'yolov5', 'data', 'labels', 'val')
images_dir = os.path.join(root_dir, 'ExDark_Dataset')

# Class mapping based on data.yaml
class_mapping = {
    'Bicycle': 1,     # bicycle
    'Boat': 9,        # boat
    'Bottle': 39,     # bottle
    'Bus': 5,         # bus
    'Car': 2,         # car
    'Cat': 15,        # cat
    'Chair': 56,      # chair
    'Dog': 16,        # dog
    'Motorbike': 3,   # motorcycle
    'People': 0,      # person
    'Table': 60,      # dining table
    # 'Others':  ignoring 
}

def convert_label_file(label_path, image_path):
    """
    Convert a label file to YOLO format.

    Args:
        label_path (str): Path to the original label file.
        image_path (str): Path to the corresponding image file.

    Returns:
        list: List of converted label lines.
    """
    converted_labels = []

    # Read image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return converted_labels
    height, width, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue  # Skip empty lines or lines starting with '%'

        # Ignore the first 16 characters
        if len(line) < 16:
            print(f"Skipping short line in {label_path}: {line}")
            continue
        relevant_part = line[16:].strip()

        parts = relevant_part.split()
        if len(parts) < 4:
            print(f"Skipping invalid line in {label_path}: {line}")
            continue

        class_name = parts[0]
        try:
            l = float(parts[1])
            t = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            print(f"Error: Non-numeric values in {label_path}: {line}")
            continue

        # Convert to YOLO format
        x_center = (l + w / 2) / width
        y_center = (t + h / 2) / height
        norm_width = w / width
        norm_height = h / height

        # Get class ID
        class_id = class_mapping.get(class_name)
        if class_id is None:
            print(f"Warning: Class '{class_name}' not found in class mapping.")
            continue

        # Ensure values are between 0 and 1
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= norm_width <= 1 and 0 <= norm_height <= 1):
            print(f"Warning: Normalized values out of range in {label_path}: {line}")
            continue

        converted_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
        converted_labels.append(converted_line)

    return converted_labels

def process_labels(labels_dir):
    """
    Process all label files in a directory.

    Args:
        labels_dir (str): Path to the labels directory.
    """
    for cls in os.listdir(labels_dir):
        cls_label_dir = os.path.join(labels_dir, cls)
        if not os.path.isdir(cls_label_dir):
            continue
        for lbl_file in os.listdir(cls_label_dir):
            if not lbl_file.endswith('.txt'):
                continue
            lbl_path = os.path.join(cls_label_dir, lbl_file)
            # Derive image path from label path
            image_name = os.path.splitext(lbl_file)[0] + '.jpg'  # Assuming images are .jpg
            image_path = os.path.join(images_dir, cls, image_name)
            if not os.path.exists(image_path):
                # Try other extensions
                for ext in ['.png', '.jpeg']:
                    image_path_alt = os.path.join(images_dir, cls, os.path.splitext(lbl_file)[0] + ext)
                    if os.path.exists(image_path_alt):
                        image_path = image_path_alt
                        break
            if not os.path.exists(image_path):
                print(f"Warning: Image for label {lbl_path} not found.")
                continue
            # Convert label
            converted = convert_label_file(lbl_path, image_path)
            if converted:
                # Overwrite the label file with converted labels
                with open(lbl_path, 'w') as f:
                    f.writelines(converted)

# Process training labels
process_labels(labels_train_dir)

# Process validation labels
process_labels(labels_val_dir)

print("Label conversion completed successfully.")
