#!/usr/bin/env python3
"""
Debug script to visualize each step of the license plate recognition pipeline.
Usage: python debug_recognition.py <image_path>
"""
import sys
import os
import shutil

import json
import cv2
import numpy as np
from pathlib import Path

from src.io_utils import load_image
from src.preprocess import resize_image, convert_to_grayscale, enhance_contrast, reduce_noise
from src.plate_detection import (
    get_contours,
    filter_contours,
    get_plate_corners,
    four_point_transform,
    normalize_orientation,
)
from src.character_segmentation import (
    find_character_contours,
    extract_and_resize_characters,
    segment_characters_hybrid,
)
from src.features_improved import extract_features_from_characters
from src.model_improved import load_model

MODEL_PATH = "models/char_svm.pkl"
PARAMS_FILE = "best_params.json"
OUTPUT_DIR = "debug_output"

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

def debug_recognition(image_path: str):
    """Debug the recognition pipeline with visual output"""
    
    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Debugging: {image_path}")
    print(f"{'='*60}\n")
    
    # Load parameters and model
    params = load_best_params()
    print(f"Parameters loaded:")
    print(f"  - char_min_area: {params['char_min_area']}")
    print(f"  - height_ratio: {params['height_ratio_min']} - {params['height_ratio_max']}")
    print(f"  - aspect_ratio: {params['aspect_ratio_min']} - {params['aspect_ratio_max']}")
    print(f"  - binarize_block_size: {params['binarize_block_size']}")
    print(f"  - binarize_C: {params['binarize_C']}\n")
    
    model = load_model(MODEL_PATH)
    
    # Step 1: Load and preprocess image
    print("Step 1: Loading image...")
    image = load_image(image_path)
    resized = resize_image(image)
    cv2.imwrite(f"{OUTPUT_DIR}/1_resized.png", resized)
    print(f"  ✓ Image loaded: {resized.shape}")
    
    # Step 2: Enhance
    print("\nStep 2: Enhancing...")
    gray_pre = convert_to_grayscale(resized)
    enhanced = enhance_contrast(gray_pre, alpha=1.3, beta=10)
    denoised = reduce_noise(enhanced, method='bilateral')
    resized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{OUTPUT_DIR}/2_enhanced.png", resized)
    print("  ✓ Enhanced")
    
    # Step 3: Detect plate
    print("\nStep 3: Detecting plate...")
    try:
        plate = detect_plate(resized, params)
        cv2.imwrite(f"{OUTPUT_DIR}/3_plate_detected.png", plate)
        print(f"  ✓ Plate detected: {plate.shape}")
    except RuntimeError as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Step 4: Binarize
    print("\nStep 4: Binarizing plate...")
    plate_gray = convert_to_grayscale(plate)
    cv2.imwrite(f"{OUTPUT_DIR}/4a_plate_gray.png", plate_gray)
    
    binary_plate = binarize_plate_tunable(
        plate_gray,
        params['binarize_block_size'],
        params['binarize_C']
    )
    cv2.imwrite(f"{OUTPUT_DIR}/4b_binary.png", binary_plate)
    print(f"  ✓ Binarized")
    
    # Step 5: Clean
    print("\nStep 5: Cleaning binary plate...")
    cleaned_plate = clean_binary_plate_tunable(
        binary_plate,
        params['morph_kernel_size']
    )
    cv2.imwrite(f"{OUTPUT_DIR}/5_cleaned.png", cleaned_plate)
    print("  ✓ Cleaned")
    
    # Step 6: Segment characters
    print("\nStep 6: Segmenting characters...")
    char_boxes = segment_characters_hybrid(cleaned_plate, cleaned_plate.shape, params)
    print(f"  ✓ Found {len(char_boxes)} character boxes")
    
    if len(char_boxes) == 0:
        print("\n⚠️  WARNING: No characters detected!")
        print("\nTrying different parameters...")
        
        # Try with more lenient parameters
        lenient_params = params.copy()
        lenient_params['char_min_area'] = 20
        lenient_params['height_ratio_min'] = 0.2
        lenient_params['height_ratio_max'] = 0.95
        lenient_params['aspect_ratio_min'] = 0.1
        lenient_params['aspect_ratio_max'] = 1.5
        
        char_boxes = segment_characters_hybrid(cleaned_plate, cleaned_plate.shape, lenient_params)
        print(f"  With lenient params: Found {len(char_boxes)} character boxes")
    
    # Draw character boxes
    plate_with_boxes = cv2.cvtColor(cleaned_plate, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(char_boxes):
        cv2.rectangle(plate_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(plate_with_boxes, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(f"{OUTPUT_DIR}/6_character_boxes.png", plate_with_boxes)
    
    for i, (x, y, w, h) in enumerate(char_boxes):
        print(f"    Char {i}: x={x}, y={y}, w={w}, h={h}, area={w*h}")
    
    if len(char_boxes) == 0:
        print("\n⚠️  Cannot proceed without characters")
        return
    
    # Step 7: Extract and predict
    print("\nStep 7: Extracting and recognizing characters...")
    characters = extract_and_resize_characters(
        cleaned_plate, char_boxes, target_size=(28, 28)
    )
    
    # Save individual characters
    for i, char in enumerate(characters):
        cv2.imwrite(f"{OUTPUT_DIR}/7_char_{i}.png", char)
    
    features = extract_features_from_characters(characters, use_combined=True)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)
    
    predictions = model.predict(features)
    
    if hasattr(model, 'label_encoder_'):
        predictions = model.label_encoder_.inverse_transform(predictions)
    
    plate_text = "".join(predictions)
    
    print(f"\n{'='*60}")
    print(f"RESULT: {plate_text}")
    print(f"{'='*60}\n")
    print(f"Debug images saved to: {OUTPUT_DIR}/")
    
    return plate_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_recognition.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)
    
    try:
        debug_recognition(image_path)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
