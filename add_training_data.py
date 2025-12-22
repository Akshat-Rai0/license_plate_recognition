#!/usr/bin/env python3
"""
Script to add labeled license plate images to the training dataset.
Usage: python add_training_data.py --image <image_path> --text <plate_text>
"""

import argparse
import sys
import os
import cv2
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.io_utils import load_image
from src.preprocess import resize_image, convert_to_grayscale, enhance_contrast, reduce_noise
from src.plate_detection import (
    get_contours,
    filter_contours,
    get_plate_corners,
    four_point_transform,
    normalize_orientation,
)
from src.character_segmentation import segment_characters_hybrid

DATASET_DIR = "data/raw/nigerian_plates/training_data/train20X20"
PARAMS_FILE = "best_params.json"

def load_best_params():
    """Load best parameters from JSON file, or use defaults"""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            return json.load(f)
    return {
        'blur_kernel': [5, 5],
        'canny_threshold1': 50,
        'canny_threshold2': 150,
        'min_area': 500,
        'ar_min': 1.5,
        'ar_max': 6.0,
        'binarize_block_size': 11,
        'binarize_C': 12,
        'morph_kernel_size': [3, 3],
        'char_min_area': 50,
        'height_ratio_min': 0.3,
        'height_ratio_max': 0.95,
        'aspect_ratio_min': 0.15,
        'aspect_ratio_max': 1.2
    }

def return_edges_tunable(image, blur_kernel, canny_threshold1, canny_threshold2):
    if isinstance(blur_kernel, list):
        blur_kernel = tuple(blur_kernel)
    blurred = cv2.GaussianBlur(image, blur_kernel, sigmaX=0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    return edges

def binarize_plate_tunable(plate_gray, block_size, C):
    binary_plate = cv2.adaptiveThreshold(
        plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, C
    )
    return binary_plate

def clean_binary_plate_tunable(binary_plate, kernel_size):
    if isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)
    cleaned_plate = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned_plate

def detect_plate(image, params):
    gray = convert_to_grayscale(image)
    edges = return_edges_tunable(
        gray,
        params['blur_kernel'],
        params['canny_threshold1'],
        params['canny_threshold2']
    )
    contours = get_contours(edges)
    contours = filter_contours(contours, min_area=params['min_area'])

    candidates = []
    for contour in contours:
        corners = get_plate_corners(contour)
        if corners is None:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ar = w / float(h) if h > 0 else 0
        if params['ar_min'] <= ar <= params['ar_max']:
            score = abs(ar - 3.5)
            candidates.append((score, corners))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_corners = candidates[0][1]
        plate = four_point_transform(image, best_corners)
        norm_plate = normalize_orientation(plate)
        return norm_plate

    raise RuntimeError("License plate not detected in image")

def add_training_data(image_path: str, plate_text: str, dry_run: bool = False):
    """Extract characters from image and add to training dataset."""
    
    print(f"\n{'='*60}")
    print(f"Adding training data")
    print(f"{'='*60}")
    print(f"Image: {image_path}")
    print(f"Plate text: {plate_text}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*60}\n")
    
    # Validate plate text
    plate_text = plate_text.upper().strip()
    if not plate_text:
        raise ValueError("Plate text cannot be empty")
    
    # Load parameters
    params = load_best_params()
    
    # Step 1: Load and preprocess
    print("Step 1: Loading and preprocessing image...")
    image = load_image(image_path)
    resized = resize_image(image)
    gray_pre = convert_to_grayscale(resized)
    enhanced = enhance_contrast(gray_pre, alpha=1.3, beta=10)
    denoised = reduce_noise(enhanced, method='bilateral')
    resized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    print("  ✓ Preprocessed")
    
    # Step 2: Detect plate
    print("\nStep 2: Detecting plate...")
    plate = detect_plate(resized, params)
    print(f"  ✓ Plate detected: {plate.shape}")
    
    # Step 3: Binarize
    print("\nStep 3: Binarizing plate...")
    plate_gray = convert_to_grayscale(plate)
    binary_plate = binarize_plate_tunable(
        plate_gray,
        params['binarize_block_size'],
        params['binarize_C']
    )
    print("  ✓ Binarized")
    
    # Step 4: Clean
    print("\nStep 4: Cleaning binary plate...")
    cleaned_plate = clean_binary_plate_tunable(
        binary_plate,
        params['morph_kernel_size']
    )
    print("  ✓ Cleaned")
    
    # Step 5: Segment characters
    print("\nStep 5: Segmenting characters...")
    char_boxes = segment_characters_hybrid(cleaned_plate, cleaned_plate.shape, params)
    print(f"  ✓ Found {len(char_boxes)} character boxes")
    
    # Validate number of characters
    if len(char_boxes) != len(plate_text):
        print(f"\n⚠️  WARNING: Character count mismatch!")
        print(f"  Expected: {len(plate_text)} characters")
        print(f"  Found: {len(char_boxes)} characters")
        print(f"\n  This may indicate segmentation issues.")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
    
    # Step 6: Save characters
    print("\nStep 6: Saving characters to dataset...")
    
    if dry_run:
        print("  [DRY RUN - not saving files]")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, (x, y, w, h) in enumerate(char_boxes):
        if i >= len(plate_text):
            print(f"  Skipping extra character {i}")
            continue
            
        char_label = plate_text[i]
        char_img = cleaned_plate[y:y+h, x:x+w]
        
        # Resize to 20x20 to match dataset
        char_img = cv2.resize(char_img, (20, 20))
        
        # Create class directory if needed
        class_dir = Path(DATASET_DIR) / char_label
        
        if not dry_run:
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Save image
            filename = f"{timestamp}_{i}.png"
            output_path = class_dir / filename
            cv2.imwrite(str(output_path), char_img)
            print(f"  ✓ Saved '{char_label}' to {output_path}")
        else:
            print(f"  [DRY RUN] Would save '{char_label}' (20x20)")
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}")
    
    if not dry_run:
        print(f"\nCharacters added to: {DATASET_DIR}")
        print(f"\nNext step: Retrain the model:")
        print(f"  ./venv/bin/python train_improved.py {DATASET_DIR}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add labeled license plate image to training dataset"
    )
    parser.add_argument('--image', type=str, required=True,
                       help="Path to license plate image")
    parser.add_argument('--text', type=str, required=True,
                       help="Correct plate text (e.g., 'KL01CA2555')")
    parser.add_argument('--dry-run', action='store_true',
                       help="Preview without saving files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        success = add_training_data(args.image, args.text, dry_run=args.dry_run)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
