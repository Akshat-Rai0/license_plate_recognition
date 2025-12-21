import os
import cv2
import numpy as np

from src.io_utils import load_image
from src.preprocess import resize_image, convert_to_grayscale
from src.plate_detection import (
    get_contours,
    filter_contours,
    get_plate_corners,
    four_point_transform,
    normalize_orientation,
)
from src.character_segmentation import (
    find_character_contours,
    sort_characters_left_to_right,
    extract_and_resize_characters,
)
from src.features import extract_features_from_characters
from src.model import load_model

MODEL_PATH = "models/char_svm.pkl"

def return_edges_tunable(image, blur_kernel=(5, 5), canny_threshold1=50, canny_threshold2=150):
    """Tunable edge detection"""
    if isinstance(blur_kernel, list):
        blur_kernel = tuple(blur_kernel)
    blurred = cv2.GaussianBlur(image, blur_kernel, sigmaX=0)
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    return edges

def binarize_plate_tunable(plate_gray, block_size=11, C=12):
    """Tunable binarization"""
    binary_plate = cv2.adaptiveThreshold(
        plate_gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C
    )
    return binary_plate

def clean_binary_plate_tunable(binary_plate, kernel_size=(3, 3)):
    """Tunable morphological cleaning"""
    if isinstance(kernel_size, list):
        kernel_size = tuple(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)
    cleaned_plate = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return cleaned_plate

def filter_character_contours_tunable(contours, plate_shape, 
                                     min_area=100, 
                                     height_ratio_min=0.4, 
                                     height_ratio_max=0.95,
                                     aspect_ratio_min=0.2, 
                                     aspect_ratio_max=1.0):
    """Tunable character contour filtering"""
    plate_height, plate_width = plate_shape[:2]
    valid_character_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        height_ratio = h / plate_height
        aspect_ratio = w / h if h != 0 else 0

        if (
            area > min_area and
            height_ratio_min < height_ratio < height_ratio_max and
            aspect_ratio_min < aspect_ratio < aspect_ratio_max
        ):
            valid_character_boxes.append((x, y, w, h))
    
    return valid_character_boxes

def detect_plate_tunable(image, 
                        blur_kernel=(5, 5),
                        canny_threshold1=50,
                        canny_threshold2=150,
                        min_area=500,
                        ar_min=2.0,
                        ar_max=5.0):
    """Tunable plate detection"""
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    
    gray = convert_to_grayscale(image)
    edges = return_edges_tunable(gray, blur_kernel, canny_threshold1, canny_threshold2)
    contours = get_contours(edges)
    contours = filter_contours(contours, min_area=min_area)

    candidates = []
    for contour in contours:
        corners = get_plate_corners(contour)
        if corners is None:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        ar = w / float(h) if h > 0 else 0
        if ar_min <= ar <= ar_max:
            score = abs(ar - 3.5)
            candidates.append((score, corners))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_corners = candidates[0][1]
        plate = four_point_transform(image, best_corners)
        norm_plate = normalize_orientation(plate)
        return norm_plate

    raise RuntimeError("License plate not detected in image")

def recognize_plate_tunable(image_path, params):
    """
    Full pipeline with tunable parameters
    
    Args:
        image_path: Path to input image
        params: Dictionary with parameter values:
            - blur_kernel: tuple or list [x, y]
            - canny_threshold1: int
            - canny_threshold2: int
            - min_area: int
            - ar_min: float
            - ar_max: float
            - binarize_block_size: int (must be odd)
            - binarize_C: int
            - morph_kernel_size: tuple or list [x, y]
            - char_min_area: int
            - height_ratio_min: float
            - height_ratio_max: float
            - aspect_ratio_min: float
            - aspect_ratio_max: float
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Please train the model first using: python train.py"
        )
    
    model = load_model(MODEL_PATH)
    
    image = load_image(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Invalid image loaded from {image_path}")
    
    resized = resize_image(image)
    
    # Plate detection with parameters
    plate = detect_plate_tunable(
        resized,
        blur_kernel=params['blur_kernel'],
        canny_threshold1=params['canny_threshold1'],
        canny_threshold2=params['canny_threshold2'],
        min_area=params['min_area'],
        ar_min=params['ar_min'],
        ar_max=params['ar_max']
    )
    
    # Character segmentation with parameters
    plate_gray = convert_to_grayscale(plate)
    binary_plate = binarize_plate_tunable(
        plate_gray,
        block_size=params['binarize_block_size'],
        C=params['binarize_C']
    )
    cleaned_plate = clean_binary_plate_tunable(
        binary_plate,
        kernel_size=params['morph_kernel_size']
    )
    
    contours = find_character_contours(cleaned_plate)
    char_boxes = filter_character_contours_tunable(
        contours,
        cleaned_plate.shape,
        min_area=params['char_min_area'],
        height_ratio_min=params['height_ratio_min'],
        height_ratio_max=params['height_ratio_max'],
        aspect_ratio_min=params['aspect_ratio_min'],
        aspect_ratio_max=params['aspect_ratio_max']
    )
    char_boxes = sort_characters_left_to_right(char_boxes)
    
    characters = extract_and_resize_characters(
        cleaned_plate, char_boxes, target_size=(28, 28)
    )
    
    if len(characters) == 0:
        raise RuntimeError("No characters detected")
    
    features = extract_features_from_characters(characters)
    
    # Validate features before prediction
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.clip(features, -10.0, 10.0)
    
    predictions = model.predict(features)
    plate_text = "".join(predictions)
    
    return plate_text