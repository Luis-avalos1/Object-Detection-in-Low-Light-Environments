import os 
import ultralytics as yolo_model # pretrained model 
import cv2 # image processing techniques 
import numpy as np 
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib as plot 
from sklearn.metrics import precision_recall_curve, average_precision_score
import json 
import glob
import xml.etree.ElementTree as element_tree

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""
    Defining our Dataset Paths 
          
    we want to define a path for our dataset --> dataset should consist of low-light images with object
        :: --> should contains exdark classes in their own folders    
    define an output path for our results ---> enhanced images && detection results 
"""
DATASET_PATH_DIR = 'ExDark_Dataset'  
GROUNDS_TRUTH_PATH_DIR = 'ground_truths'
OUTPUT_PATH_DIR = 'enhanced_images' 


# we are using the ExDark data set with contains its own classes 
exdark_classes = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Dog', 'Motorbike', 'People', 'Table', 'Others'
]

# create a mapping from class names to index #s, for compatibility with COCO classes used by our pretrained model 
coco_mapping = {
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

def model_initialization():
    """
        Model Initialization
        
           --> define our object detection model --> using YOLO 
               we will use a small pretrained model from ultralytics
               
        currently running too slow on cpu --> lets make if available , use GPU 
    """
    configure_devie = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device {configure_devie}")
    
    object_detection_model = torch.hub.load("ultralytics/yolov5", "yolov5s").to(configure_devie)
     
    return object_detection_model

# rs 
# TODO: Current Issues :: unable to 'find' annotataion files despite them being there --> Issue in load_gt ()
    # TODO: potential fix - since we are loading in our data into these list, mabe we use the annotations from here 
def parse_images_and_annotations(DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR):
    # our dataset seperates their images by classes so we need to iterate all these folders and get their annotations
    # create a list of the images we want and their annotations 
     # Create lists to store image and annotation file paths
    image_files = []
    annotation_files = []

    # defining extension our imgs in our dataset can have --> there is ~7400 img, so if there is any that arent .png they will be handled
    img_extension = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    # Loop through each class directory in the dataset
    for current_class_name in os.listdir(DATASET_PATH_DIR):
        
        current_class_diretory = os.path.join(DATASET_PATH_DIR, current_class_name)
        current_annotation_dir = os.path.join(GROUNDS_TRUTH_PATH_DIR, current_class_name)

        print(f"\nProcessing class: {current_class_name}")
        print(f"Image directory: {current_class_diretory}")
        print(f"Annotation directory: {current_annotation_dir}")

        # Ensure both directories exist
        if os.path.isdir(current_class_diretory) and os.path.isdir(current_annotation_dir):
            # Process each image in the class directory
            for image_filename in os.listdir(current_class_diretory):
                file_ext = os.path.splitext(image_filename)[1].lower()

                # Check if file is an image
                if file_ext in img_extension:
                    img_file = os.path.join(current_class_diretory, image_filename)

                    # Construct corresponding annotation filename by replacing image extension with .txt
                    annotation_filename = os.path.splitext(image_filename)[0] + '.txt'
                    annot_file = os.path.join(current_annotation_dir, annotation_filename)

                    # Verify if the annotation file exists
                    if os.path.exists(annot_file):
                        image_files.append(img_file)
                        annotation_files.append(annot_file)
                    else:
                        print(f"Annotation file not found for image: {img_file}")
                else:
                    print(f"Skipped file with unsupported extension: {image_filename}")
        else:
            if not os.path.isdir(current_class_diretory):
                print(f"Class directory not found in dataset: {current_class_diretory}")
            if not os.path.isdir(current_annotation_dir):
                print(f"Class directory not found in ground truths: {current_annotation_dir}")

    print("\nTotal images collected:", len(image_files))
    print("Total annotations collected:", len(annotation_files))

    return image_files, annotation_files


# rs 
# TODO: annotations files are not being found despite them being there --> in dir 
def ground_truth_annotation(ground_truth_path):
    """
    Loads ground truth annotations from a .txt file in the specified format.

    Args:
        ground_truth_path (str): Path to the annotation .txt file.

    Returns:
        dict: Dictionary containing 'boxes' and 'labels'.
    """
    boxes = []
    labels = []

    try:
        with open(ground_truth_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Annotation file not found: {ground_truth_path}")
        return {'boxes': np.array([]), 'labels': np.array([])}
    except UnicodeDecodeError:
        print(f"Could not decode annotation file: {ground_truth_path}")
        return {'boxes': np.array([]), 'labels': np.array([])}

    # Skip the header line if it starts with '%'
    start_line = 0
    if lines[0].startswith('%'):
        start_line = 1

    # Process each annotation line
    for line in lines[start_line:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip invalid lines

        class_name = parts[0]
        if class_name not in coco_mapping:
            continue  # Skip classes not in the mapping

        label_idx = coco_mapping[class_name]

        # Parse bounding box coordinates
        x = int(float(parts[1]))
        y = int(float(parts[2]))
        width = int(float(parts[3]))
        height = int(float(parts[4]))

        # Convert width and height to x2, y2
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height

        boxes.append([x1, y1, x2, y2])
        labels.append(label_idx)

    return {'boxes': np.array(boxes), 'labels': np.array(labels)}


def histogram_equalization(low_light_image):
    """_summary_
    
    # method that enhances the image on the V channel of HSV color space 
    # Parameter: we want to intake an image in BGR color space 
    # returns enhaced image in BGR space 
    
    """
    
    # convert to hsv, then split into h s v channels
    hsv_conversion_image = cv2.cvtColor(low_light_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_conversion_image)
    # applyinh hq --> V channel 
    hist_eq = cv2.equalizeHist(v)
    hq_image = cv2.merge((h,s,hist_eq))
    # merge to bgr 
    operated_image = cv2.cvtColor(hq_image, cv2.COLOR_HSV2BGR)

    # return our enhanced Image 
    return operated_image

def single_scale_ret(low_light_image, sigma=15):
    # method that enhaces img using single scale retinex algo
    # parameter: img in bgr, standard deviation for Gaussion blur == sigma
    # returns enhanced img in bgr 
    
    # convert our img to float +1 --> to avoing log(0)
    flt = low_light_image.astype(np.float32) + 1.0
    # take the log of float 
    log = np.log(flt)
    # apply gaussian blur and take the log 
    gaus_log = np.log(cv2.GaussianBlur(flt, (0,0), sigma))
    # compute their diff 
    diff = log - gaus_log
    # normalize it 
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    # convert to uint8
    diff = diff.astype(np.uint8)
    
    return diff

def clahe_enhancement(low_light_image):
    # method that enhances an image using : Contrast Limited Adaptive Histogram Equalization 
    # Parameter: we want to intake an image in BGR space \
    # returns enhaced image 
    
    # convert and split h s v 
    hsv_conversion_image = cv2.cvtColor(low_light_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_conversion_image)
    # creating a CLAHE object 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # applyin enhancement to v channel 
    vc = clahe.apply(v)
    # merge 
    clahe_img = cv2.merge((h,s, vc))
    # convert back 
    final_img = cv2.cvtColor(clahe_img, cv2.COLOR_HSV2BGR)
    
    
    return final_img


def detection_results_evaluation(model_detections, ground_truths, threshold=0.5):
    """
        Method that evaluates the object detection model's results using precision, recall and average precision 

    Args:
        model_detections (list): a list of detection results from our YOLOv5 model 
        ground_truths (list): a list of ground truths 
        threshold (float, optional): IoU to consider a detection as a true positive, Defaults to 0.5.
        
    Returns: will return a tuple with precision array, recall array, and average precision score 
    """
    scores = []  # List to store confidence scores
    matches = []  # Store matches --> 1 == TP, 0 == FP
    ground_truths_count = 0

    # Flag to indicate if any detections were made
    detections_made = False

    # Iterate over detections and ground truth annotations
    for detection, gt in zip(model_detections, ground_truths):
        boxes_detection = detection['boxes']
        scores_detection = detection['scores']
        detection_labels = detection['labels']
        boxes_ground_truth = gt['boxes']
        labels_ground_truth = gt['labels']

        # Update ground truth counter
        ground_truths_count += len(boxes_ground_truth)
        pair_matches = np.zeros(len(boxes_detection))

        if len(boxes_detection) > 0:
            detections_made = True  # Detections were made in this image

        for i, box_detection in enumerate(boxes_detection):
            # No ground truths left to match
            if len(boxes_ground_truth) == 0:
                pair_matches[i] = 0
                continue

            # Compute IoU between detection and ground truth boxes
            calc_iou = iou_computation(box_detection, boxes_ground_truth)

            # Get the index of the ground truth box with highest IoU
            max_index = np.argmax(calc_iou)
            max_i = calc_iou[max_index]

            if max_i >= threshold and detection_labels[i] == labels_ground_truth[max_index]:
                # True Positive
                pair_matches[i] = 1
                # Remove matched ground truth to prevent duplicate matching
                boxes_ground_truth = np.delete(boxes_ground_truth, max_index, axis=0)
                labels_ground_truth = np.delete(labels_ground_truth, max_index, axis=0)
            else:
                # False Positive
                pair_matches[i] = 0

        # Append detection scores and matches to lists
        scores.extend(scores_detection)
        matches.extend(pair_matches)

    scores = np.array(scores)
    matches = np.array(matches)

    # No detection case
    if len(scores) == 0:
        # Do not print "No Detections Made"
        return np.array([]), np.array([]), 0.0, False

    # Sort detections in descending order
    sorted_index = np.argsort(-scores)
    scores = scores[sorted_index]
    matches = matches[sorted_index]

    # Compute cumulative true positives and false positives
    all_matches = np.cumsum(matches)
    all_fp = np.cumsum(1 - matches)

    # Calculate precision and recall for each detection
    precision = all_matches / (all_matches + all_fp + 1e-6)
    recall = all_matches / (ground_truths_count + 1e-6)

    # Calculate average precision
    average_precision = average_precision_score(matches, scores)

    return precision, recall, average_precision, detections_made
            
          
def iou_computation(box, boxes):
    
    """
        Method that calculates the intersection over union IOU beteween one bounding box and mulitple ground truth bounding boxes
            --> The iou basically quantifies our accuracy between our predicted boxes by measuring how much our predicted box overlaps 
                --> with our grounds truth box 
        
        Parameters: box - list or numpy.ndarray :: a single bounding box consists of coordinates [x1, y1, x2, y2] 
                    boxes - array of boxes 
        
        Returns: Array of iou values, in our cases we will return a numpy.ndarray 
        
        Procedure:
                (i)   : Calculate the overlapping coordinates 
                (ii)  : Calculate the area of intersection 
                (iii) : Calculate the iou   
        
    """
    
      # Calculate the coordinates of the intersection rectangle
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])  # Use np.minimum
    y2 = np.minimum(box[3], boxes[:, 3])  # Use np.minimum

    # Compute the area of intersection rectangle
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute the area of both the prediction and ground truth rectangles
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute the union area
    union = area_box + area_boxes - intersection + 1e-6  # Avoid division by zero

    # Compute the IoU
    iou = intersection / union

    return iou
    

def calculate_average_metrics(metrics):
    
    """
        Method the approximates the precision, recall, and average precision across our images to calculate the avg metrics for our entire dataset 
            --> By averaging our metrics we can gain an overview of the model's performance across our dataset 

        Args: metrics (list) : list of metrics dictionaries 
        
        Returns: returns a tuple --> Average precision , average recall and average ap 
    """
    
    # empty
    if len(metrics) == 0:
        return 0.0, 0.0, 0.0

    # compute the mean of the last precision, last recall value for each img 
    avg_p = np.mean([m['precision'][-1] for m in metrics if len(m['precision'])>0])
    avg_re = np.mean([m['recall'][-1] for m in metrics if len(m['recall'])>0])
    
    # calc mean of avergage precison scores
    avg_ap = np.mean([m['ap'] for m in metrics])
    
    
    return avg_p, avg_re, avg_ap
    
def evaluate_object_detections(results):
    """
    Parse YOLO detection results into a format :: boxes, scores, labels

    Args:
        results (YOLO Results): detection results from YOLO model
    """
    detections = []

    for res in results.xyxy:
        boxes = []
        scores = []
        labels = []
        for *box, score, cls in res.cpu().numpy():
            boxes.append(box)
            scores.append(score)
            labels.append(int(cls))
        detection = {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'labels': np.array(labels)
        }
        detections.append(detection)

    return detections
    

def object_detection(image_path, annot_path, obj_detection_model, og_metrics, histeq_metrics, clahe_metrics, retinex_metrics, detection_counts):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the ground truth for this image
    ground_truth = ground_truth_annotation(annot_path)
    truths = [ground_truth]
    
    # Skip images without annotations
    if len(truths[0]['boxes']) == 0:
        return
    
    # Perform object detection on the original image
    results_original = obj_detection_model(image_rgb)
    
    # Perform image enhancements
    histeq_image = histogram_equalization(image)
    clahe_image = clahe_enhancement(image)
    single_ret_image = single_scale_ret(image)
    
    # Convert enhanced images to RGB
    histeq_image_rgb = cv2.cvtColor(histeq_image, cv2.COLOR_BGR2RGB)
    clahe_image_rgb = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB)
    single_ret_image_rgb = cv2.cvtColor(single_ret_image, cv2.COLOR_BGR2RGB)
    
    # Perform object detection on enhanced images
    res_histeq = obj_detection_model(histeq_image_rgb)
    res_clahe = obj_detection_model(clahe_image_rgb)
    res_ret = obj_detection_model(single_ret_image_rgb)
    
    # Evaluate detections made from the model
    og_detections = evaluate_object_detections(results_original)
    hq_detections = evaluate_object_detections(res_histeq)
    cl_detections = evaluate_object_detections(res_clahe)
    ret_detections = evaluate_object_detections(res_ret)
    
    # Metrics: evaluate detections against ground truths
    og_precision, og_recall, og_ap, og_detections_made = detection_results_evaluation(og_detections, truths)
    histo_precision, histo_recall, histo_ap, histo_detections_made = detection_results_evaluation(hq_detections, truths)
    clahe_precision, clahe_recall, clahe_ap, clahe_detections_made = detection_results_evaluation(cl_detections, truths)
    ret_precision, ret_recall, ret_ap, ret_detections_made = detection_results_evaluation(ret_detections, truths)
    
    # Store the metrics in the list
    og_metrics.append({'precision': og_precision, 'recall': og_recall, 'ap': og_ap})
    histeq_metrics.append({'precision': histo_precision, 'recall': histo_recall, 'ap': histo_ap})
    clahe_metrics.append({'precision': clahe_precision, 'recall': clahe_recall, 'ap': clahe_ap})
    retinex_metrics.append({'precision': ret_precision, 'recall': ret_recall, 'ap': ret_ap})
    
    # Update detection counts
    detection_counts['original']['detections_made'] += int(og_detections_made)
    detection_counts['original']['no_detections'] += int(not og_detections_made)
    detection_counts['histeq']['detections_made'] += int(histo_detections_made)
    detection_counts['histeq']['no_detections'] += int(not histo_detections_made)
    detection_counts['clahe']['detections_made'] += int(clahe_detections_made)
    detection_counts['clahe']['no_detections'] += int(not clahe_detections_made)
    detection_counts['retinex']['detections_made'] += int(ret_detections_made)
    detection_counts['retinex']['no_detections'] += int(not ret_detections_made)
    
    # Save enhanced images
    filename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_PATH_DIR, f"{filename}_histEQ.jpg"), histeq_image)
    cv2.imwrite(os.path.join(OUTPUT_PATH_DIR, f"{filename}_clahe.jpg"), clahe_image)
    cv2.imwrite(os.path.join(OUTPUT_PATH_DIR, f"{filename}_retinex.jpg"), single_ret_image)

    

def main():
    # Initialize the YOLOv5 model
    yolov5 = model_initialization()
    
    # Collect images and their annotation paths
    images, annotations = parse_images_and_annotations(DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR)
    
    # Create lists to store metrics for each image enhancement technique
    og_metrics = []
    hist_metrics = []
    clahe_metrics = []
    ret_metrics = []
    
    # Initialize detection counts
    detection_counts = {
        'original': {'detections_made': 0, 'no_detections': 0},
        'histeq': {'detections_made': 0, 'no_detections': 0},
        'clahe': {'detections_made': 0, 'no_detections': 0},
        'retinex': {'detections_made': 0, 'no_detections': 0},
    }
    
    # For loop calling object_detection()
    for image_path, annot_path in tqdm(zip(images, annotations), total=len(images), desc="YOLOv5 Processing Images"):
        object_detection(image_path, annot_path, yolov5, og_metrics, hist_metrics, clahe_metrics, ret_metrics, detection_counts)
    
    # Compute average metrics
    og_avg_prec, og_avg_recall, og_avg_ap = calculate_average_metrics(og_metrics)
    hq_avg_prec, hq_avg_recall, hq_avg_ap = calculate_average_metrics(hist_metrics)
    cl_avg_prec, cl_avg_recall, cl_avg_ap = calculate_average_metrics(clahe_metrics)
    re_avg_prec, re_avg_recall, re_avg_ap = calculate_average_metrics(ret_metrics)
    
    # Display results
    print("\nEvaluation Results")
    print(f"\nOriginal Images \nPrecision: {og_avg_prec:.4f}, \nRecall: {og_avg_recall:.4f}, \nAP: {og_avg_ap:.4f}")
    print(f"\nImages Enhanced with Histogram Equalization \nPrecision: {hq_avg_prec:.4f}, \nRecall: {hq_avg_recall:.4f}, \nAP: {hq_avg_ap:.4f}")
    print(f"\nImages Enhanced with CLAHE \nPrecision: {cl_avg_prec:.4f}, \nRecall: {cl_avg_recall:.4f}, \nAP: {cl_avg_ap:.4f}")
    print(f"\nImages Enhanced with Retinex \nPrecision: {re_avg_prec:.4f}, \nRecall: {re_avg_recall:.4f}, \nAP: {re_avg_ap:.4f}")
    
    # Display detection counts
    print("\nDetection Counts:")
    for key in detection_counts:
        total_images = detection_counts[key]['detections_made'] + detection_counts[key]['no_detections']
        print(f"{key.capitalize()} Images:")
        print(f"  Detections Made: {detection_counts[key]['detections_made']} out of {total_images}")
        print(f"  No Detections: {detection_counts[key]['no_detections']} out of {total_images}")

    
if __name__ == "__main__":
    main()
    