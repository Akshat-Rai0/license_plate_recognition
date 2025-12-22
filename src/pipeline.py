import os
import json
import cv2
import numpy as np

from src.io_utils import load_image
from src.preprocess import resize_image, convert_to_grayscale, enhance_contrast, reduce_noise
from src.plate_detection import (
    get_contours,
    filter_contours,
    get_plate_corners,
    four_point_transform,
    normalize_orientation,
    detect_plate_vertical_projection,
)
from src.character_segmentation import (
    find_character_contours,
    sort_characters_left_to_right,
    extract_and_resize_characters,
    segment_characters_hybrid,
    filter_character_contours_tunable,
)
from src.features import extract_features_from_characters
from src.model import load_model
from src.post_process import validate_and_correct_plate, filter_valid_characters

MODEL_PATH = "models/char_svm.pkl"
PARAMS_FILE = "best_params.json"

def load_best_params():
    """Load best parameters from JSON file, or use defaults"""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            params = json.load(f)
            # Validate parameters are reasonable
            return params
    # Default parameters (more general)
    return {
        'blur_kernel': [5, 5],
        'canny_threshold1': 50,
        'canny_threshold2': 150,
        'min_area': 500,
        'ar_min': 1.5,  # More lenient
        'ar_max': 6.0,  # More lenient
        'binarize_block_size': 11,
        'binarize_C': 12,
        'morph_kernel_size': [3, 3],
        'char_min_area': 50,  # Lower threshold
        'height_ratio_min': 0.3,  # More lenient
        'height_ratio_max': 0.95,
        'aspect_ratio_min': 0.15,  # More lenient
        'aspect_ratio_max': 1.2  # More lenient
    }

def return_edges_tunable(image, blur_kernel, canny_threshold1, canny_threshold2):
    """Edge detection with parameters"""
    if isinstance(blur_kernel, list):
        blur_kernel = tuple(blur_kernel)
    blurred = cv2.GaussianBlur(image, blur_kernel, sigmaX=0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    return edges

def binarize_plate_tunable(plate_gray, block_size, C):
    """Binarization with parameters"""
    binary_plate = cv2.adaptiveThreshold(
        plate_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block_size, C
    )
    return binary_plate

def clean_binary_plate_tunable(binary_plate, kernel_size):
    """Morphological cleaning with parameters"""
    if isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)
    cleaned_plate = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned_plate

# filter_character_contours_tunable moved to character_segmentation.py to avoid circular imports

def detect_plate(image: np.ndarray, params, return_coords: bool = False):
    """Detect plate with optimized parameters"""
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    
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
        if return_coords:
            return norm_plate, best_corners
        return norm_plate

    raise RuntimeError("License plate not detected in image")

def recognize_plate(image_path: str) -> str:
    """Recognize license plate with optimized parameters"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Please train the model first using: python train.py"
        )
    
    params = load_best_params()
    model = load_model(MODEL_PATH)

    image = load_image(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Invalid image loaded from {image_path}")
    
    # Preprocessing: resize and enhance (from Nigerian repo approach)
    resized = resize_image(image)
    
    # Enhance contrast and reduce noise for better detection
    gray_pre = convert_to_grayscale(resized)
    enhanced = enhance_contrast(gray_pre, alpha=1.3, beta=10)
    denoised = reduce_noise(enhanced, method='bilateral')
    resized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for compatibility
    # Use hybrid detection (contour first, projection fallback)
    try:
        # Try contour-based detection first
        plate = detect_plate(resized, params)
    except RuntimeError:
        # Fallback to vertical projection method
        try:
            plate = detect_plate_vertical_projection(resized)
            plate = normalize_orientation(plate)
        except RuntimeError:
            raise RuntimeError("License plate not detected using any method")

    plate_gray = convert_to_grayscale(plate)
    binary_plate = binarize_plate_tunable(
        plate_gray,
        params['binarize_block_size'],
        params['binarize_C']
    )
    cleaned_plate = clean_binary_plate_tunable(
        binary_plate,
        params['morph_kernel_size']
    )

    # Use hybrid character segmentation (contour first, projection fallback)
    char_boxes = segment_characters_hybrid(cleaned_plate, cleaned_plate.shape, params)

    characters = extract_and_resize_characters(
        cleaned_plate, char_boxes, target_size=(28, 28)
    )
    
    if len(characters) == 0:
        raise RuntimeError("No characters detected on license plate")

    features = extract_features_from_characters(characters)
    
    # Validate features before prediction
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)
    
    predictions = model.predict(features)
    
    # Decode predictions if label encoder exists
    if hasattr(model, 'label_encoder_'):
        predictions = model.label_encoder_.inverse_transform(predictions)
    
    plate_text = "".join(predictions)
    
    # Post-processing: validate and correct common errors
    plate_text = filter_valid_characters(plate_text)
    plate_text = validate_and_correct_plate(plate_text)
    
    return plate_text