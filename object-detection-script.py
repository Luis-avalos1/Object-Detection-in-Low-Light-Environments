import os 
import ultralytics as YOLO # pretrained model 
import cv2 # image processing techniques 
import numpy as np 
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib as plot 
from sklearn.metrics import precision_recall_curve, average_precision_score
import json 
import glob

# we want to define a path for our dataset --> dataset should consist of low-light images with object 
# TODO: obtain dataset, with grount truth annotations  --> an option rn is ExDark Dataset 
dataset_path = 'ExDark_Dataset'
# define an output path for our results ---> enhanced images && detection results 
output_path = 'enhanced_images'

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

# define our object detection model --> using YOLO 
    # we will use a small pretrained model from ultralytics 
object_detection_model = YOLO('yolov5s.pt')


# our dataset seperates their images by classes so we need to iterate all these folders and get their annotations
# create a list of the images we want and their annotations 
images = []
annotations = []

# looping over class directories 
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    # collect images in class dir 
    if os.path.is_dir(class_dir):
        img_file = glob(os.path.join(class_dir, '*.jpg'))
        annot = [f.replace('.jpg', '.xml') for f in img_file]
        
        # append 
        images.extend(img_file)
        annotations.extend(annot)

# additionally, we want to verify that the image and annotations for those images actually exist and are correct 
valid_images = []
valid_annotations = []

for img_p , ann_p in zip(images, annotations):
    if os.path.exists(img_p) and os.path.exists(ann_p):
        valid_annotations.append(ann_p)
        valid_images.append(img_p)
    else:
        print(f"Missing file: {img_p} or {ann_p}")
        
# if everything is correct we can use valid lis t
images = valid_images
annotations = valid_annotations

def histogram_equalization(low_light_image):
    # method that enhances the image on the V channel of HSV color space 
    # Parameter: we want to intake an image in BGR color space 
    # returns enhaced image in BGR space 
    
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


def perform_object_detection(image):
    # method that performs object detection on an image usingn our pretrain YOLO model 
    # parameter: img in bgr 
    # returns detection results from our model 
    
    res = object_detection_model(image)
    return res

def ground_truth_annotation():
    return