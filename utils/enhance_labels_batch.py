#!/usr/bin/env python3
"""
Batch Label Enhancement Utility for ExDark Dataset

Simple command-line utility for enhancing ExDark dataset labels with advanced metadata.

Usage:
    python enhance_labels_batch.py --dataset_path ExDark_Dataset --ground_truth_path ground_truths
"""

import argparse
import sys
from pathlib import Path
from enhanced_labeling import ExDarkLabelEnhancer

def main():
    parser = argparse.ArgumentParser(description='Enhance ExDark dataset labels with advanced metadata')
    
    parser.add_argument('--dataset_path', type=str, default='ExDark_Dataset',
                       help='Path to ExDark dataset directory')
    parser.add_argument('--ground_truth_path', type=str, default='ground_truths',
                       help='Path to ground truth annotations directory')
    parser.add_argument('--output_dir', type=str, default='enhanced_labels',
                       help='Output directory for enhanced labels')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--quality_report', type=str, default='quality_report.json',
                       help='Path for quality assessment report')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        sys.exit(1)
    
    if not Path(args.ground_truth_path).exists():
        print(f"Error: Ground truth path '{args.ground_truth_path}' does not exist")
        sys.exit(1)
    
    print("=== ExDark Dataset Label Enhancement ===")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Ground truth path: {args.ground_truth_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max images: {args.max_images if args.max_images else 'All'}")
    print()
    
    # Initialize enhancer
    enhancer = ExDarkLabelEnhancer(
        dataset_path=args.dataset_path,
        ground_truth_path=args.ground_truth_path
    )
    
    # Process dataset
    enhancer.process_dataset(max_images=args.max_images)
    
    # Export enhanced labels
    enhancer.export_enhanced_labels_yolo(args.output_dir)
    
    # Export quality report
    enhancer.export_quality_report(args.quality_report)
    
    print("\n=== Enhancement Complete ===")
    print(f"Enhanced {len(enhancer.enhanced_labels)} annotations")
    print(f"Enhanced labels saved to: {args.output_dir}")
    print(f"Quality report saved to: {args.quality_report}")

if __name__ == "__main__":
    main()