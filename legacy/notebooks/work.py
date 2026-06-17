import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define dataset paths
DATASET_PATH_DIR = 'ExDark_Dataset'
GROUNDS_TRUTH_PATH_DIR = 'ground_truths'
OUTPUT_PATH_DIR = 'enhanced_images'
DETECTION_OUTPUT_DIR = 'detection_results'  # Directory to save images with detections

# Ensure output directories exist
os.makedirs(OUTPUT_PATH_DIR, exist_ok=True)
os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)

# ExDark classes mapping to COCO classes
coco_mapping = {
    'Bicycle': 1, 'Boat': 9, 'Bottle': 39, 'Bus': 5,
    'Car': 2, 'Cat': 15, 'Chair': 56, 'Dog': 16,
    'Motorbike': 3, 'People': 0, 'Table': 60
}

def initialize_model():
    """
    Initialize the YOLOv5 model.

    Returns:
        model: Loaded YOLOv5 model.
        device: Torch device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Using a medium-sized model for better accuracy
        model = torch.hub.load("ultralytics/yolov5", "yolov5n").to(device)
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # IoU threshold
        model.eval()
        torch.backends.cudnn.benchmark = True
        return model, device
    except Exception:
        return None, None

def parse_images_and_annotations(dataset_dir, annotations_dir, max_images=None):
    """
    Parse image and annotation file paths.

    Args:
        dataset_dir (str): Directory containing images organized by class.
        annotations_dir (str): Directory containing annotation files organized by class.
        max_images (int, optional): Maximum number of images to parse for testing.

    Returns:
        image_files (list): List of image file paths.
        annotation_files (list): List of corresponding annotation file paths.
    """
    image_files = []
    annotation_files = []
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        annotation_class_dir = os.path.join(annotations_dir, class_name)

        if not (os.path.isdir(class_dir) and os.path.isdir(annotation_class_dir)):
            continue

        for image_filename in os.listdir(class_dir):
            if max_images and len(image_files) >= max_images:
                break
            file_ext = os.path.splitext(image_filename)[1].lower()
            if file_ext not in img_extensions:
                continue
            img_path = os.path.join(class_dir, image_filename)
            annot_filename = os.path.splitext(image_filename)[0] + '.txt'
            annot_path = os.path.join(annotation_class_dir, annot_filename)
            if os.path.exists(annot_path):
                image_files.append(img_path)
                annotation_files.append(annot_path)

    return image_files, annotation_files

def ground_truth_annotation(annot_path):
    """
    Parse ground truth annotations from a file.

    Args:
        annot_path (str): Path to the annotation file.

    Returns:
        dict: Dictionary containing 'boxes' and 'labels'.
    """
    boxes = []
    labels = []

    try:
        with open(annot_path, 'r') as f:
            lines = f.readlines()
    except:
        return {'boxes': np.array([]), 'labels': np.array([])}

    start_line = 1 if lines and lines[0].startswith('%') else 0

    for line in lines[start_line:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        class_name, x, y, width, height = parts[:5]
        if class_name not in coco_mapping:
            continue
        label_idx = coco_mapping[class_name]
        try:
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
        except:
            continue
        boxes.append([x, y, x + width, y + height])
        labels.append(label_idx)

    return {'boxes': np.array(boxes), 'labels': np.array(labels)}

def histogram_equalization(image):
    """
    Apply Histogram Equalization to the V channel of the HSV image.

    Args:
        image (np.array): Input BGR image.

    Returns:
        np.array: Histogram equalized BGR image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

def gamma_correction(image, gamma=1.5):
    """
    Apply Gamma Correction to the image.

    Args:
        image (np.array): Input image.
        gamma (float): Gamma value.

    Returns:
        np.array: Gamma corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def clahe_enhancement(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel of LAB image.

    Args:
        image (np.array): Input BGR image.

    Returns:
        np.array: CLAHE enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def bilateral_filtering(image):
    """
    Apply bilateral filtering to reduce noise while preserving edges.

    Args:
        image (np.array): Input BGR image.

    Returns:
        np.array: Bilaterally filtered BGR image.
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def resize_image(image, size=(416, 416)):
    """
    Resize the image to the specified size.

    Args:
        image (np.array): Input image.
        size (tuple): Desired size.

    Returns:
        np.array: Resized image.
    """
    return cv2.resize(image, size)

def iou_computation(box, boxes):
    """
    Compute Intersection over Union (IoU) between a box and multiple boxes.

    Args:
        box (list): Single box [x1, y1, x2, y2].
        boxes (np.array): Multiple boxes [[x1, y1, x2, y2], ...].

    Returns:
        np.array: IoU values.
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection + 1e-6
    return intersection / union

def detection_results_evaluation(detections, ground_truths, threshold=0.5, confidence_threshold=0.3):
    """
    Evaluate detection results against ground truths.

    Args:
        detections (list): List of detection dictionaries.
        ground_truths (list): List of ground truth dictionaries.
        threshold (float): IoU threshold.
        confidence_threshold (float): Confidence score threshold.

    Returns:
        tuple: (precision, recall, average_precision, detections_made)
    """
    scores = []
    matches = []
    ground_truths_count = 0
    detections_made = False

    for det, gt in zip(detections, ground_truths):
        conf_mask = det['scores'] >= confidence_threshold
        boxes_det = det['boxes'][conf_mask]
        scores_det = det['scores'][conf_mask]
        labels_det = det['labels'][conf_mask]
        boxes_gt = gt['boxes']
        labels_gt = gt['labels']

        ground_truths_count += len(boxes_gt)
        pair_matches = np.zeros(len(boxes_det))

        if len(boxes_det) > 0:
            detections_made = True

        for i, box_det in enumerate(boxes_det):
            if len(boxes_gt) == 0:
                pair_matches[i] = 0
                continue
            ious = iou_computation(box_det, boxes_gt)
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            if max_iou >= threshold and labels_det[i] == labels_gt[max_iou_idx]:
                pair_matches[i] = 1
                boxes_gt = np.delete(boxes_gt, max_iou_idx, axis=0)
                labels_gt = np.delete(labels_gt, max_iou_idx, axis=0)
            else:
                pair_matches[i] = 0

        scores.extend(scores_det)
        matches.extend(pair_matches)

    scores = np.array(scores)
    matches = np.array(matches)

    if len(scores) == 0:
        return np.array([]), np.array([]), 0.0, False

    sorted_indices = np.argsort(-scores)
    scores = scores[sorted_indices]
    matches = matches[sorted_indices]

    cumulative_matches = np.cumsum(matches)
    cumulative_fp = np.cumsum(1 - matches)

    precision = cumulative_matches / (cumulative_matches + cumulative_fp + 1e-6)
    recall = cumulative_matches / (ground_truths_count + 1e-6)

    try:
        average_precision = average_precision_score(matches, scores)
    except:
        average_precision = 0.0

    return precision, recall, average_precision, detections_made

def evaluate_object_detections(results):
    """
    Convert YOLOv5 results to a list of detection dictionaries.

    Args:
        results: YOLOv5 detection results.

    Returns:
        list: List of detection dictionaries.
    """
    detections = []
    for res in results.xyxy:
        boxes = res[:, :4].cpu().numpy()
        scores = res[:, 4].cpu().numpy()
        labels = res[:, 5].cpu().numpy().astype(int)
        detections.append({
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        })
    return detections

def draw_detections(image, detections, ground_truth, class_names, output_path):
    """
    Draw bounding boxes for detections and ground truths on the image.

    Args:
        image (np.array): BGR image.
        detections (list): List of detection dictionaries.
        ground_truth (dict): Dictionary containing 'boxes' and 'labels'.
        class_names (dict): Mapping from class indices to class names.
        output_path (str): Path to save the annotated image.

    Returns:
        None
    """
    # Draw ground truth boxes in green
    for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, class_names.get(label, 'Unknown'), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw detection boxes in red
    for det in detections:
        for box, score, label in zip(det['boxes'], det['scores'], det['labels']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f"{class_names.get(label, 'Unknown')}:{score:.2f}"
            cv2.putText(image, text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Save the annotated image
    cv2.imwrite(output_path, image)

def object_detection_process(image_path, annot_path, model, device, class_names, enhancement=None, apply_bilateral=False):
    """
    Process a single image for object detection with specified enhancements.

    Args:
        image_path (str): Path to the image.
        annot_path (str): Path to the annotation file.
        model: YOLOv5 model.
        device: Torch device.
        class_names (dict): Mapping from class indices to class names.
        enhancement (function, optional): Enhancement function to apply.
        apply_bilateral (bool): Whether to apply bilateral filtering.

    Returns:
        dict: Detection metrics.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None

    original_h, original_w = image.shape[:2]

    # Apply bilateral filtering if specified
    if apply_bilateral:
        try:
            image = bilateral_filtering(image)
        except:
            return None

    # Apply enhancement if specified
    if enhancement:
        try:
            image = enhancement(image)
        except:
            return None

    # Resize image
    try:
        image = resize_image(image, size=(416, 416))
    except:
        return None

    # Compute scaling factors
    resized_h, resized_w = image.shape[:2]
    sx = resized_w / original_w
    sy = resized_h / original_h

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parse ground truth annotations
    ground_truth = ground_truth_annotation(annot_path)
    truths = [ground_truth]

    if len(truths[0]['boxes']) == 0:
        return None

    # Scale ground truth boxes if image was resized
    truths[0]['boxes'] = truths[0]['boxes'].astype(float)
    truths[0]['boxes'][:, [0, 2]] = truths[0]['boxes'][:, [0, 2]] * sx
    truths[0]['boxes'][:, [1, 3]] = truths[0]['boxes'][:, [1, 3]] * sy

    # Perform object detection
    try:
        with torch.no_grad():
            results = model(image_rgb)

        # Evaluate detections
        detections = evaluate_object_detections(results)
        precision, recall, ap, detections_made = detection_results_evaluation(detections, truths)

        # Draw detections and ground truths
        output_filename = os.path.basename(image_path)
        enhancement_suffix = enhancement.__name__ if enhancement else "original"
        bilateral_suffix = "_bilateral" if apply_bilateral else ""
        output_path = os.path.join(DETECTION_OUTPUT_DIR, f"detection_{enhancement_suffix}{bilateral_suffix}_{output_filename}")
        draw_detections(image.copy(), detections, ground_truth, class_names, output_path)

        return {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'detections_made': detections_made
        }
    except:
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def calculate_average_metrics(metrics):
    """
    Calculate average precision, recall, and AP from metrics.

    Args:
        metrics (list): List of metric dictionaries.

    Returns:
        tuple: (avg_precision, avg_recall, avg_ap)
    """
    if not metrics:
        return 0.0, 0.0, 0.0
    avg_precision = np.mean([m['precision'][-1] if len(m['precision']) > 0 else 0.0 for m in metrics])
    avg_recall = np.mean([m['recall'][-1] if len(m['recall']) > 0 else 0.0 for m in metrics])
    avg_ap = np.mean([m['ap'] for m in metrics])
    return avg_precision, avg_recall, avg_ap

def main():
    # Initialize the model
    model, device = initialize_model()
    if model is None:
        return

    # Parse images and annotations
    images, annotations = parse_images_and_annotations(DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR)
    if not images:
        return

    # Define enhancement techniques
    enhancement_methods = {
        'histogram_eq': histogram_equalization,
        'gamma_corr': lambda img: gamma_correction(img, gamma=1.5),
        'clahe': clahe_enhancement
        # Add more enhancement methods here if needed
    }

    # Initialize metrics storage
    metrics = {
        'original': [],
    }
    for name in enhancement_methods:
        metrics[name] = []
        metrics[f"{name}_bilateral"] = []

    # Initialize detection counts
    detection_counts = {
        'original': {'detections_made': 0, 'no_detections': 0},
    }
    for name in enhancement_methods:
        detection_counts[name] = {'detections_made': 0, 'no_detections': 0}
        detection_counts[f"{name}_bilateral"] = {'detections_made': 0, 'no_detections': 0}

    # Class names mapping (inverse of coco_mapping)
    class_names = {v: k for k, v in coco_mapping.items()}

    # Start processing images with tqdm progress bar
    for img, ann in tqdm(zip(images, annotations), total=len(images), desc="Processing Images"):
        # Process Original Image
        original_result = object_detection_process(
            image_path=img,
            annot_path=ann,
            model=model,
            device=device,
            class_names=class_names,
            enhancement=None,
            apply_bilateral=False
        )
        if original_result:
            metrics['original'].append(original_result)
            detection_counts['original']['detections_made'] += int(original_result['detections_made'])
            detection_counts['original']['no_detections'] += int(not original_result['detections_made'])

        # Process Enhanced Images
        for name, enhancement in enhancement_methods.items():
            # Without Bilateral Filtering
            enhanced_result = object_detection_process(
                image_path=img,
                annot_path=ann,
                model=model,
                device=device,
                class_names=class_names,
                enhancement=enhancement,
                apply_bilateral=False
            )
            if enhanced_result:
                metrics[name].append(enhanced_result)
                detection_counts[name]['detections_made'] += int(enhanced_result['detections_made'])
                detection_counts[name]['no_detections'] += int(not enhanced_result['detections_made'])

            # With Bilateral Filtering
            enhanced_bilateral_result = object_detection_process(
                image_path=img,
                annot_path=ann,
                model=model,
                device=device,
                class_names=class_names,
                enhancement=enhancement,
                apply_bilateral=True
            )
            if enhanced_bilateral_result:
                metrics[f"{name}_bilateral"].append(enhanced_bilateral_result)
                detection_counts[f"{name}_bilateral"]['detections_made'] += int(enhanced_bilateral_result['detections_made'])
                detection_counts[f"{name}_bilateral"]['no_detections'] += int(not enhanced_bilateral_result['detections_made'])

    # Compute and display average metrics
    print("\n=== Evaluation Results ===")
    for key in metrics:
        avg_prec, avg_recall, avg_ap = calculate_average_metrics(metrics[key])
        print(f"{key.replace('_', ' ').capitalize()} - Precision: {avg_prec:.4f}, Recall: {avg_recall:.4f}, AP: {avg_ap:.4f}")

    # Display detection counts
    print("\n=== Detection Counts ===")
    for key in detection_counts:
        total = detection_counts[key]['detections_made'] + detection_counts[key]['no_detections']
        print(f"{key.replace('_', ' ').capitalize()} Images:")
        print(f"  Detections Made: {detection_counts[key]['detections_made']} out of {total}")
        print(f"  No Detections: {detection_counts[key]['no_detections']} out of {total}")

if __name__ == "__main__":
    main()
