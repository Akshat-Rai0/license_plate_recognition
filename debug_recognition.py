
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
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image

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
    segment_characters_hybrid,
    extract_and_resize_characters,
)
# Use pipeline imports or direct definitions?
# We need get_model_and_labels from pipeline or re-implement
from src.pipeline import get_model_and_labels, load_best_params
from src.pipeline import (
    binarize_plate_tunable, 
    clean_binary_plate_tunable, 
    detect_plate,
    PARAMS_FILE,
    MODEL_PATH,
    LABEL_ENCODER_PATH
)


OUTPUT_DIR = "debug_output"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def debug_recognition(image_path: str):
    """Debug the recognition pipeline with visual output"""
    
    # Create output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Debugging: {image_path}")
    print(f"{'='*60}\n")
    
    # Load parameters
    params = load_best_params()
    print(f"Parameters loaded:")
    print(f"  - char_min_area: {params['char_min_area']}")
    print(f"  - binarize_block_size: {params['binarize_block_size']}")
    
    # Load model
    print("Loading PyTorch model...")
    try:
        model, label_map = get_model_and_labels()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Step 1: Load and preprocess image
    print("Step 1: Loading image...")
    image = load_image(image_path)
    if image is None:
        print("Error: Could not load image")
        return
        
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
    
    # Draw character boxes
    plate_with_boxes = cv2.cvtColor(cleaned_plate, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(char_boxes):
        cv2.rectangle(plate_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(plate_with_boxes, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(f"{OUTPUT_DIR}/6_character_boxes.png", plate_with_boxes)
    
    # Step 7: Extract and predict
    if len(char_boxes) > 0:
        print("\nStep 7: Recognizing characters...")
        characters = extract_and_resize_characters(
            cleaned_plate, char_boxes, target_size=(28, 28)
        )
        
        # Save individual characters
        for i, char in enumerate(characters):
            cv2.imwrite(f"{OUTPUT_DIR}/7_char_{i}.png", char)
        
        preds = []
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        with torch.no_grad():
            for i, char_img in enumerate(characters):
                pil_img = Image.fromarray(char_img)
                input_tensor = transform_pipeline(pil_img).unsqueeze(0).to(device)
                
                output = model(input_tensor)
                _, predicted = torch.max(output.data, 1)
                
                idx = predicted.item()
                char_str = label_map.get(idx, '?')
                preds.append(char_str)
                print(f"  Char {i}: Predicted '{char_str}' (Class {idx})")
        
        plate_text = "".join(preds)
        print(f"\nResult: {plate_text}")
    else:
        print("\nSkipping recognition (no characters)")

    print(f"\nDebug output saved to {OUTPUT_DIR}")


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

