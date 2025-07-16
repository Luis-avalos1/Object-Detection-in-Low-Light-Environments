#!/usr/bin/env python3
"""
Enhanced Labeling System for ExDark Dataset

This module provides enhanced labeling capabilities for the ExDark low-light object detection dataset,
including confidence scoring, quality validation, lighting condition assessment, and metadata enrichment.

Author: Enhanced by Claude Code
"""

import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Enhanced bounding box with additional metadata."""
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    difficulty: str = 'normal'  # easy, normal, hard
    visibility: str = 'clear'   # clear, partial, occluded
    lighting_condition: str = 'unknown'  # bright, dim, dark, very_dark

@dataclass
class EnhancedAnnotation:
    """Enhanced annotation structure with rich metadata."""
    image_path: str
    image_width: int
    image_height: int
    class_name: str
    class_id: int
    bbox: BoundingBox
    created_date: str
    enhanced: bool = True
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    blur_score: float = 0.0
    noise_level: float = 0.0

class ExDarkLabelEnhancer:
    """Enhanced labeling system for ExDark dataset."""
    
    def __init__(self, dataset_path: str, ground_truth_path: str):
        self.dataset_path = Path(dataset_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.class_mapping = {
            'Bicycle': 1, 'Boat': 9, 'Bottle': 39, 'Bus': 5,
            'Car': 2, 'Cat': 15, 'Chair': 56, 'Dog': 16,
            'Motorbike': 3, 'People': 0, 'Table': 60
        }
        self.enhanced_labels = []
        
    def assess_lighting_condition(self, image: np.ndarray) -> str:
        """
        Assess lighting condition of the image.
        
        Args:
            image: Input image array
            
        Returns:
            Lighting condition string
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness > 180:
            return 'bright'
        elif mean_brightness > 120:
            return 'dim'
        elif mean_brightness > 60:
            return 'dark'
        else:
            return 'very_dark'
    
    def calculate_image_quality_metrics(self, image: np.ndarray) -> Dict[str, float]:
        """
        Calculate image quality metrics for enhanced labeling.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary of quality metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness score (normalized mean intensity)
        brightness = np.mean(gray) / 255.0
        
        # Contrast score (standard deviation of intensity)
        contrast = np.std(gray) / 255.0
        
        # Blur score (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        # Noise level estimation (using high-frequency content)
        noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0)) / 255.0
        
        return {
            'brightness_score': brightness,
            'contrast_score': contrast,
            'blur_score': blur_score,
            'noise_level': noise
        }
    
    def assess_object_difficulty(self, bbox: Dict, image: np.ndarray) -> str:
        """
        Assess object detection difficulty based on size, position, and image quality.
        
        Args:
            bbox: Bounding box dictionary
            image: Input image array
            
        Returns:
            Difficulty level string
        """
        h, w = image.shape[:2]
        
        # Calculate relative size
        relative_area = (bbox['width'] * bbox['height']) / (w * h)
        
        # Check if object is near image borders
        border_distance = min(bbox['x'], bbox['y'], 
                             w - (bbox['x'] + bbox['width']), 
                             h - (bbox['y'] + bbox['height']))
        near_border = border_distance < min(w, h) * 0.05
        
        # Assess based on size and position
        if relative_area < 0.01 or near_border:
            return 'hard'
        elif relative_area < 0.05:
            return 'normal'
        else:
            return 'easy'
    
    def assess_visibility(self, bbox: Dict, image: np.ndarray) -> str:
        """
        Assess object visibility within bounding box.
        
        Args:
            bbox: Bounding box dictionary
            image: Input image array
            
        Returns:
            Visibility level string
        """
        # Extract ROI
        x, y, w, h = int(bbox['x']), int(bbox['y']), int(bbox['width']), int(bbox['height'])
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 'occluded'
        
        # Calculate edge density as visibility indicator
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.1:
            return 'clear'
        elif edge_density > 0.05:
            return 'partial'
        else:
            return 'occluded'
    
    def parse_original_annotation(self, annotation_path: str) -> List[Dict]:
        """
        Parse original ExDark annotation file.
        
        Args:
            annotation_path: Path to annotation file
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Failed to read annotation file {annotation_path}: {e}")
            return annotations
        
        # Skip header line if present
        start_line = 1 if lines and lines[0].startswith('%') else 0
        
        for line in lines[start_line:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            class_name = parts[0]
            if class_name not in self.class_mapping:
                continue
                
            try:
                x = float(parts[1])
                y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                annotations.append({
                    'class_name': class_name,
                    'class_id': self.class_mapping[class_name],
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })
            except ValueError as e:
                logger.warning(f"Invalid annotation in {annotation_path}: {line.strip()}")
                continue
        
        return annotations
    
    def enhance_single_annotation(self, image_path: str, annotation_path: str) -> List[EnhancedAnnotation]:
        """
        Enhance annotations for a single image.
        
        Args:
            image_path: Path to image file
            annotation_path: Path to annotation file
            
        Returns:
            List of enhanced annotations
        """
        enhanced_annotations = []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return enhanced_annotations
        
        h, w = image.shape[:2]
        
        # Calculate image quality metrics
        quality_metrics = self.calculate_image_quality_metrics(image)
        lighting_condition = self.assess_lighting_condition(image)
        
        # Parse original annotations
        original_annotations = self.parse_original_annotation(annotation_path)
        
        for ann in original_annotations:
            # Assess object-specific properties
            difficulty = self.assess_object_difficulty(ann, image)
            visibility = self.assess_visibility(ann, image)
            
            # Create enhanced bounding box
            bbox = BoundingBox(
                x=ann['x'],
                y=ann['y'],
                width=ann['width'],
                height=ann['height'],
                confidence=1.0,  # Original annotations assumed to be ground truth
                difficulty=difficulty,
                visibility=visibility,
                lighting_condition=lighting_condition
            )
            
            # Create enhanced annotation
            enhanced_ann = EnhancedAnnotation(
                image_path=image_path,
                image_width=w,
                image_height=h,
                class_name=ann['class_name'],
                class_id=ann['class_id'],
                bbox=bbox,
                created_date=datetime.now().isoformat(),
                enhanced=True,
                brightness_score=quality_metrics['brightness_score'],
                contrast_score=quality_metrics['contrast_score'],
                blur_score=quality_metrics['blur_score'],
                noise_level=quality_metrics['noise_level']
            )
            
            enhanced_annotations.append(enhanced_ann)
        
        return enhanced_annotations
    
    def validate_annotation_quality(self, annotation: EnhancedAnnotation) -> Dict[str, Any]:
        """
        Validate annotation quality and provide quality metrics.
        
        Args:
            annotation: Enhanced annotation to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 1.0
        }
        
        bbox = annotation.bbox
        
        # Check bounding box validity
        if bbox.x < 0 or bbox.y < 0:
            validation_results['errors'].append("Negative coordinates")
            validation_results['valid'] = False
        
        if bbox.width <= 0 or bbox.height <= 0:
            validation_results['errors'].append("Invalid dimensions")
            validation_results['valid'] = False
        
        if bbox.x + bbox.width > annotation.image_width:
            validation_results['errors'].append("Bounding box exceeds image width")
            validation_results['valid'] = False
        
        if bbox.y + bbox.height > annotation.image_height:
            validation_results['errors'].append("Bounding box exceeds image height")
            validation_results['valid'] = False
        
        # Quality assessment
        quality_factors = []
        
        # Size quality
        relative_size = (bbox.width * bbox.height) / (annotation.image_width * annotation.image_height)
        if relative_size < 0.001:
            validation_results['warnings'].append("Very small object")
            quality_factors.append(0.5)
        elif relative_size > 0.8:
            validation_results['warnings'].append("Very large object")
            quality_factors.append(0.7)
        else:
            quality_factors.append(1.0)
        
        # Brightness quality
        if annotation.brightness_score < 0.2:
            validation_results['warnings'].append("Very dark image")
            quality_factors.append(0.6)
        else:
            quality_factors.append(min(annotation.brightness_score * 2, 1.0))
        
        # Visibility quality
        visibility_scores = {'clear': 1.0, 'partial': 0.7, 'occluded': 0.4}
        quality_factors.append(visibility_scores.get(bbox.visibility, 0.5))
        
        # Calculate overall quality score
        validation_results['quality_score'] = np.mean(quality_factors)
        
        return validation_results
    
    def export_enhanced_labels_yolo(self, output_dir: str) -> None:
        """
        Export enhanced labels in YOLO format with additional metadata.
        
        Args:
            output_dir: Output directory for enhanced labels
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create separate directories for different formats
        yolo_dir = output_path / 'yolo_format'
        metadata_dir = output_path / 'metadata'
        yolo_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)
        
        for annotation in self.enhanced_labels:
            # Generate YOLO format file
            image_name = Path(annotation.image_path).stem
            yolo_file = yolo_dir / f"{image_name}.txt"
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (annotation.bbox.x + annotation.bbox.width / 2) / annotation.image_width
            y_center = (annotation.bbox.y + annotation.bbox.height / 2) / annotation.image_height
            norm_width = annotation.bbox.width / annotation.image_width
            norm_height = annotation.bbox.height / annotation.image_height
            
            yolo_line = f"{annotation.class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
            
            with open(yolo_file, 'a') as f:
                f.write(yolo_line)
            
            # Export metadata
            metadata_file = metadata_dir / f"{image_name}.json"
            metadata = asdict(annotation)
            
            # Load existing metadata if file exists
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    existing_data = json.load(f)
                if 'annotations' not in existing_data:
                    existing_data = {'annotations': [existing_data]}
                existing_data['annotations'].append(metadata)
            else:
                existing_data = {'annotations': [metadata]}
            
            with open(metadata_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
    
    def export_quality_report(self, output_path: str) -> None:
        """
        Export quality assessment report for all enhanced annotations.
        
        Args:
            output_path: Path to save quality report
        """
        report = {
            'summary': {
                'total_annotations': len(self.enhanced_labels),
                'avg_brightness': np.mean([ann.brightness_score for ann in self.enhanced_labels]),
                'avg_contrast': np.mean([ann.contrast_score for ann in self.enhanced_labels]),
                'avg_blur_score': np.mean([ann.blur_score for ann in self.enhanced_labels]),
                'lighting_distribution': {},
                'difficulty_distribution': {},
                'visibility_distribution': {}
            },
            'annotations': []
        }
        
        # Calculate distributions
        lighting_counts = {}
        difficulty_counts = {}
        visibility_counts = {}
        
        for ann in self.enhanced_labels:
            # Lighting distribution
            lighting = ann.bbox.lighting_condition
            lighting_counts[lighting] = lighting_counts.get(lighting, 0) + 1
            
            # Difficulty distribution
            difficulty = ann.bbox.difficulty
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            
            # Visibility distribution
            visibility = ann.bbox.visibility
            visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1
            
            # Validate annotation
            validation = self.validate_annotation_quality(ann)
            
            report['annotations'].append({
                'image_path': ann.image_path,
                'class_name': ann.class_name,
                'lighting_condition': lighting,
                'difficulty': difficulty,
                'visibility': visibility,
                'quality_score': validation['quality_score'],
                'warnings': validation['warnings'],
                'errors': validation['errors']
            })
        
        report['summary']['lighting_distribution'] = lighting_counts
        report['summary']['difficulty_distribution'] = difficulty_counts
        report['summary']['visibility_distribution'] = visibility_counts
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {output_path}")
    
    def process_dataset(self, max_images: Optional[int] = None) -> None:
        """
        Process the entire dataset and enhance all annotations.
        
        Args:
            max_images: Maximum number of images to process (for testing)
        """
        logger.info("Starting dataset enhancement process...")
        
        processed_count = 0
        
        for class_dir in self.dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            annotation_dir = self.ground_truth_path / class_name
            
            if not annotation_dir.exists():
                logger.warning(f"No annotation directory found for class: {class_name}")
                continue
            
            for image_file in class_dir.iterdir():
                if max_images and processed_count >= max_images:
                    break
                
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                annotation_file = annotation_dir / f"{image_file.stem}.txt"
                
                if not annotation_file.exists():
                    logger.warning(f"No annotation file found for image: {image_file}")
                    continue
                
                # Enhance annotations for this image
                enhanced_anns = self.enhance_single_annotation(str(image_file), str(annotation_file))
                self.enhanced_labels.extend(enhanced_anns)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} images...")
        
        logger.info(f"Enhanced labeling completed. Processed {processed_count} images with {len(self.enhanced_labels)} annotations.")

def main():
    """Main function to demonstrate enhanced labeling functionality."""
    
    # Initialize enhancer
    enhancer = ExDarkLabelEnhancer(
        dataset_path='ExDark_Dataset',
        ground_truth_path='ground_truths'
    )
    
    # Process a subset for demonstration (remove max_images for full dataset)
    enhancer.process_dataset(max_images=50)
    
    # Export enhanced labels
    enhancer.export_enhanced_labels_yolo('enhanced_labels')
    
    # Export quality report
    enhancer.export_quality_report('quality_report.json')
    
    print(f"Enhanced {len(enhancer.enhanced_labels)} annotations")
    print("Enhanced labels exported to 'enhanced_labels' directory")
    print("Quality report saved to 'quality_report.json'")

if __name__ == "__main__":
    main()