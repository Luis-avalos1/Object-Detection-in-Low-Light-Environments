import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define dataset paths
DATASET_PATH_DIR = 'ExDark_Dataset'
GROUNDS_TRUTH_PATH_DIR = 'ground_truths'
OUTPUT_PATH_DIR = 'enhanced_images'

# ExDark classes mapping to COCO classes
coco_mapping = {
    'People': 0, 'Bicycle': 1, 'Car': 2, 'Motorbike': 3, 'Bus': 5,
    'Boat': 9, 'Dog': 16, 'Cat': 15, 'Chair': 56, 'Table': 60, 'Bottle': 39
}

def initialize_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing model on {device}")
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True).to(device)
        model.conf = 0.25
        model.iou = 0.45
        model.eval()
        torch.backends.cudnn.benchmark = True
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def parse_images_and_annotations(dataset_dir, annotations_dir):
    image_files = []
    annotation_files = []
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        annotation_class_dir = os.path.join(annotations_dir, class_name)

        if not (os.path.isdir(class_dir) and os.path.isdir(annotation_class_dir)):
            continue

        for image_filename in os.listdir(class_dir):
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
            x = float(x)       # Keep as float
            y = float(y)
            width = float(width)
            height = float(height)
        except:
            continue
        boxes.append([x, y, x + width, y + height])
        labels.append(label_idx)

    return {'boxes': np.array(boxes), 'labels': np.array(labels)}

def histogram_equalization(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30)  # Increase saturation to enhance colors
    hist_eq = cv2.equalizeHist(v)
    hist_eq = cv2.normalize(hist_eq, None, 0, 255, cv2.NORM_MINMAX)
    hq_image = cv2.merge((h, s, hist_eq))
    return cv2.cvtColor(hq_image, cv2.COLOR_HSV2BGR)

def clahe_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_enhancement_with_gamma(image, enhancement_func, gamma=1.5):
    enhanced_image = enhancement_func(image)
    return gamma_correction(enhanced_image, gamma)

def resize_image(image, size=(416, 416)):
    return cv2.resize(image, size)

def iou_computation(box, boxes):
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

def object_detection_process(image_path, annot_path, model, device, enhancement_func=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return None

    resized_image = resize_image(image, size=(416, 416))
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    ground_truth = ground_truth_annotation(annot_path)
    truths = [ground_truth]

    if len(truths[0]['boxes']) == 0:
        return None

    if enhancement_func:
        enhanced_image = enhancement_func(resized_image)
        image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        # Save the enhanced image
        output_filename = os.path.basename(image_path)
        output_path = os.path.join(OUTPUT_PATH_DIR, f"enhanced_{output_filename}")
        cv2.imwrite(output_path, enhanced_image)
    
    try:
        with torch.no_grad():
            results = model(image_rgb)
            detections = evaluate_object_detections(results)
            precision, recall, ap, detections_made = detection_results_evaluation(detections, truths)
            return {
                'precision': precision,
                'recall': recall,
                'ap': ap,
                'detections_made': detections_made
            }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def calculate_average_metrics(metrics):
    if not metrics:
        return 0.0, 0.0, 0.0
    avg_precision = np.mean([m['precision'][-1] if len(m['precision']) > 0 else 0.0 for m in metrics])
    avg_recall = np.mean([m['recall'][-1] if len(m['recall']) > 0 else 0.0 for m in metrics])
    avg_ap = np.mean([m['ap'] for m in metrics])
    return avg_precision, avg_recall, avg_ap

def main():
    model, device = initialize_model()
    if model is None:
        print("Model initialization failed. Exiting.")
        return

    images, annotations = parse_images_and_annotations(DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR)

    metrics_original = []
    metrics_histogram = []
    metrics_clahe = []
    metrics_gamma = []
    metrics_hist_gamma = []
    metrics_clahe_gamma = []

    detection_counts = {
        'original': {'detections_made': 0, 'no_detections': 0},
        'histogram': {'detections_made': 0, 'no_detections': 0},
        'clahe': {'detections_made': 0, 'no_detections': 0},
        'gamma': {'detections_made': 0, 'no_detections': 0},
        'histogram_gamma': {'detections_made': 0, 'no_detections': 0},
        'clahe_gamma': {'detections_made': 0, 'no_detections': 0},
    }

    for img, ann in tqdm(zip(images, annotations), total=len(images), desc="Processing Images"):
        # Original image detection
        result = object_detection_process(img, ann, model, device)
        if result:
            metrics_original.append(result)
            detection_counts['original']['detections_made'] += int(result['detections_made'])
            detection_counts['original']['no_detections'] += int(not result['detections_made'])

        # Histogram equalization
        result = object_detection_process(img, ann, model, device, histogram_equalization)
        if result:
            metrics_histogram.append(result)
            detection_counts['histogram']['detections_made'] += int(result['detections_made'])
            detection_counts['histogram']['no_detections'] += int(not result['detections_made'])

        # CLAHE
        result = object_detection_process(img, ann, model, device, clahe_enhancement)
        if result:
            metrics_clahe.append(result)
            detection_counts['clahe']['detections_made'] += int(result['detections_made'])
            detection_counts['clahe']['no_detections'] += int(not result['detections_made'])

        # Gamma correction
        result = object_detection_process(img, ann, model, device, gamma_correction)
        if result:
            metrics_gamma.append(result)
            detection_counts['gamma']['detections_made'] += int(result['detections_made'])
            detection_counts['gamma']['no_detections'] += int(not result['detections_made'])

        # Histogram equalization + Gamma correction
        result = object_detection_process(img, ann, model, device, lambda img: apply_enhancement_with_gamma(img, histogram_equalization))
        if result:
            metrics_hist_gamma.append(result)
            detection_counts['histogram_gamma']['detections_made'] += int(result['detections_made'])
            detection_counts['histogram_gamma']['no_detections'] += int(not result['detections_made'])

        # CLAHE + Gamma correction
        result = object_detection_process(img, ann, model, device, lambda img: apply_enhancement_with_gamma(img, clahe_enhancement))
        if result:
            metrics_clahe_gamma.append(result)
            detection_counts['clahe_gamma']['detections_made'] += int(result['detections_made'])
            detection_counts['clahe_gamma']['no_detections'] += int(not result['detections_made'])

    # Compute average metrics
    og_avg_prec, og_avg_recall, og_avg_ap = calculate_average_metrics(metrics_original)
    hist_avg_prec, hist_avg_recall, hist_avg_ap = calculate_average_metrics(metrics_histogram)
    clahe_avg_prec, clahe_avg_recall, clahe_avg_ap = calculate_average_metrics(metrics_clahe)
    gamma_avg_prec, gamma_avg_recall, gamma_avg_ap = calculate_average_metrics(metrics_gamma)
    hist_gamma_avg_prec, hist_gamma_avg_recall, hist_gamma_avg_ap = calculate_average_metrics(metrics_hist_gamma)
    clahe_gamma_avg_prec, clahe_gamma_avg_recall, clahe_gamma_avg_ap = calculate_average_metrics(metrics_clahe_gamma)

    # Display results
    print("\nEvaluation Results:")
    print(f"Original Images - Precision: {og_avg_prec:.4f}, Recall: {og_avg_recall:.4f}, AP: {og_avg_ap:.4f}")
    print(f"Histogram Equalization - Precision: {hist_avg_prec:.4f}, Recall: {hist_avg_recall:.4f}, AP: {hist_avg_ap:.4f}")
    print(f"CLAHE - Precision: {clahe_avg_prec:.4f}, Recall: {clahe_avg_recall:.4f}, AP: {clahe_avg_ap:.4f}")
    print(f"Gamma Correction - Precision: {gamma_avg_prec:.4f}, Recall: {gamma_avg_recall:.4f}, AP: {gamma_avg_ap:.4f}")
    print(f"Histogram + Gamma - Precision: {hist_gamma_avg_prec:.4f}, Recall: {hist_gamma_avg_recall:.4f}, AP: {hist_gamma_avg_ap:.4f}")
    print(f"CLAHE + Gamma - Precision: {clahe_gamma_avg_prec:.4f}, Recall: {clahe_gamma_avg_recall:.4f}, AP: {clahe_gamma_avg_ap:.4f}")

    # Display detection counts
    print("\nDetection Counts:")
    for key, counts in detection_counts.items():
        total = counts['detections_made'] + counts['no_detections']
        print(f"{key.capitalize()} Images:")
        print(f"  Detections Made: {counts['detections_made']} out of {total}")
        print(f"  No Detections: {counts['no_detections']} out of {total}")

if __name__ == "__main__":
    main()
