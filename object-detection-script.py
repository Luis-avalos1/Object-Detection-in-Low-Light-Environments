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
    """
    object_detection_model = torch.hub.load("ultralytics/yolov5", "yolov5s")
     
    return object_detection_model

def parse_images_and_annotations(DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR):
    # our dataset seperates their images by classes so we need to iterate all these folders and get their annotations
    # create a list of the images we want and their annotations 
    images = []
    annotations = []

    # looping over class directories 
    for class_name in os.listdir(DATASET_PATH_DIR):
        class_dir = os.path.join(DATASET_PATH_DIR, class_name)
        # collect images in class dir 
        if os.path.isdir(class_dir):
            img_file = glob.glob(os.path.join(class_dir, '*.jpg'))
            annot = [os.path.join(GROUNDS_TRUTH_PATH_DIR, f"{os.path.basename(f).replace('.jpg', '.xml')}") for f in img_file]
            
            # append 
            images.extend(img_file)
            annotations.extend(annot)


    print("size of images: ",  len(images))
    print("size of annot: ",  len(annotations))


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
    flt = low_light_image.astype(np.floa32) + 1.0
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


def ground_truth_annotation(ground_truth_path):
    """_summary_

    loads our grounds truth from our dataset 
    
    Args:
        ground_truth_path (str): path to annotations 

    Returns: dictionary containing boxes and labels 
    """
    
    
    tree = element_tree.parse(ground_truth_path)
    root = tree.getroot()
    boxes = []
    labels = []
    
    for object in root.findall('object'):
        lbl = object.find('name').text
        
        if lbl not in coco_mapping:
            continue
            
        idx = coco_mapping[lbl]
        bbox = object.find('bndbox')
        x1 = int(float(bbox.find('xmin').text))
        y1 = int(float(bbox.find('ymin').text))
        x2 = int(float(bbox.find('xmax').text))
        y2 = int(float(bbox.find('ymax').text))
        boxes.append([x1, y1, x2, y2])
        labels.append(idx)
        
    return {'boxes': np.array(boxes), 'labels': np.array(labels)}


def detection_results_evaluation(model_detections, ground_truths, threshold=0.5):
    """
        Method that evaluates the object detection model's results using precision, recall and average precision 

    Args:
        model_detections (list): a list of detection results from our YOLOv5 model 
        ground_truths (list): a list of ground truths 
        threshold (float, optional): IoU to consider a detection as a true positive, Defaults to 0.5.
        
    Returns: will return a tuple with precision array, recall array, and average precision score 
    """
    scores =[] # list to store our confindence score 
    matches = [] # store our matches --> 1 == TP :: 0 == FP 
    ground_truths_count = 0;
     
    #  iterate over our detectios and their ground truth annotatiosn
    for detection, gt in zip(model_detections, ground_truths):
        boxes_detection = detection['boxes']
        scores_detection = detection['scores']
        detection_labels = detection['labels']
        boxes_ground_truth = gt['boxes']
        labels_ground_truth = gt['labels']
        
        # update grounds truth counter 
        ground_truths_count += len(boxes_ground_truth)
        pair_matches = np.zeros(len(boxes_detection))
        
        for i, boxes_detection in enumerate(boxes_detection):
            # no grounds truth left case
            if len(boxes_ground_truth) == 0:
                pair_matches[i] = 0
                continue
        
            # compute the IoU between 'box' and gt boxes 
            calc_iou = iou_computation(boxes_detection, boxes_ground_truth)
            
            # get the index of gt box with highes iou 
            max_index = np.argmax(calc_iou)
            max_i = calc_iou[max_index]
            
            if max_i >= threshold and detection_labels[i] == labels_ground_truth[max_index]:
                # TP 
                pair_matches[i] == 1
                # remove the matches gt --> prevent duplicae matching
                boxes_ground_truth = np.delete(boxes_ground_truth, max_index, axis=0)
                labels_ground_truth = np.delete(labels_ground_truth, max_index, axis=0)
            
            # FP 
            else:
                pair_matches[i] == 0
        
        # append our detection scores and matches to list 
        scores.extend(scores_detection)
        matches.extend(pair_matches)

    scores = np.array(scores)
    matches = np.array(matches)
    
    # no detection case 
    if len(scores) == 0:
        # no detection 
        print("No Detections Made")
        return np.array([]), np.array([]), 0.0
    
    # sort our detections in ascending order 
    sorted_index = np.argsort(-scores)
    scores = scores[sorted_index]
    matches = matches[sorted_index]
    
    # compute cumulative tp and fp s 
    all_matches = np.cumsum(all_matches)
    all_fp = np.cumsum(1-all_matches)
    
    # calc precision and recall for each detection 
    precision = all_matches / (all_matches + all_fp + 1e-6)
    recall = all_matches / (ground_truths_count + 1e-6)
    
    # calv average precision 
    average_precision = average_precision_score(matches, scores)
    
    return precision, recall, average_precision 
            
          
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
    
    # calc the coordingate of the intersection 
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.maximum(box[2], boxes[:, 2])
    y2 = np.maximum(box[3], boxes[:, 3])
    
    # compute the area 
    intersection = np.maximum(0, x2-x1) * np.maximum(0, y1, y2)
    area_box = (box[2] - box[0]) *(box[3] - box[1])
    area_boxes = (boxes[:,2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # union 
    U = area_box + area_boxes - intersection
    
    # iou --> avoid divide by 0
    iou = intersection / (U + 1e-6) 
    
    
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
    detect = []
    
    for res in results.xyxy:
        for *box, score, cls in res.cpu().numpy():
            detect.append({
                'boxes': np.array([[box[0], box[1], box[2], box[3]]]), 
                'scores': np.array([score]), 
                'labels': np.array([int(cls)])
            })
    
    return detect
    

def object_detection( DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR, obj_detection_model, og_metrics, histeq_metrics, clahe_metrics, retinex_metrics ): 
    """
    Methods the performs object detection on the original image, then on our enhacned images
        --> then it evaluates detections, and stores their metrics for comparison

    Args:
        DATASET_PATH_DIR (string): path to our dataset images 
        GROUNDS_TRUTH_PATH_DIR (string): path to our annotations for our images in dataset
        obj_detection_model (torch.hub.loader): our obj detection model --> loaded YOLOv5 model
        og_metrics (list): list to store metrics of og results with no enhancements
        histeq_metrics (list): metrics of hist eq rseults
        clahe_metrics (list): metrics of clahe results
        retinex_metrics (list): metrics of retinex results
    
    """
    # load our single img 
    image = cv2.imread(DATASET_PATH_DIR)
    
    if image is None:
        print(f"Failed to load {DATASET_PATH_DIR}")
        return
    
    # load the ground truth for this image 
    ground_truth = ground_truth_annotation(GROUNDS_TRUTH_PATH_DIR)
    truths = [ground_truth]
    
    #skip images without annotation
    if len(truths[0]['boxes']) == 0:
        return
    
    # perform object detection on the original image with no image enhancement
    results_original = obj_detection_model(image)
    
    # perform object detetion on image enhacements 
    histeq_image = histogram_equalization(image)
    clahe_image = clahe_enhancement(image)
    single_ret = single_scale_ret(image)
    
    res_histeq = obj_detection_model(histeq_image)
    res_clahe = obj_detection_model(clahe_image)
    res_ret = obj_detection_model(single_ret)
    
    # evaluate detections made from model
    og_detections = evaluate_object_detections(results_original)
    hq_detection = evaluate_object_detections(res_histeq)
    cl_detections = evaluate_object_detections(res_clahe)
    ret_detection = evaluate_object_detections(res_ret)
    
    # ... more to do 


def main():
    # begin with initializing our YOLOv5 model 
    yolov5 = model_initialization
    
    # collect our images and their annotations paths/ground truths 
    images, annotations = parse_images_and_annotations(DATASET_PATH_DIR, GROUNDS_TRUTH_PATH_DIR)
    


if __name__ == "__main__":
    main()
    