import os
import json
import cv2
import numpy as np
import torch
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
    detect_plate_vertical_projection,
)
from src.character_segmentation import (
    find_character_contours,
    sort_characters_left_to_right,
    extract_and_resize_characters,
    segment_characters_hybrid,
    filter_character_contours_tunable,
)
from src.pytorch_model import SimpleCNN
from src.post_process import validate_and_correct_plate, filter_valid_characters


MODEL_PATH = "models/char_cnn.pth"
LABEL_ENCODER_PATH = "models/label_encoder.json"
PARAMS_FILE = "best_params.json"

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_best_params():
    """Load best parameters from JSON file, or use defaults"""
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE, 'r') as f:
            params = json.load(f)
            return params
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

def detect_plate(image: np.ndarray, params, return_coords: bool = False):
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

# Cache model and labels
_model = None
_label_map = None

def get_model_and_labels():
    global _model, _label_map
    if _model is None:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
             raise FileNotFoundError("Model files not found. Run train_pytorch.py first.")
        
        # Load Labels
        with open(LABEL_ENCODER_PATH, 'r') as f:
            _label_map = json.load(f)
            # Keys in json are strings, convert to int if needed? 
            # Actually json keys are always strings. We map int index -> string label.
            # So we need to ensure we can look up by int index.
            # The loaded dict will be {"0": "0", "1": "1", ...}
            # We need int keys.
            _label_map = {int(k): v for k, v in _label_map.items()}

        num_classes = len(_label_map)
        _model = SimpleCNN(num_classes=num_classes)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _model.to(device)
        _model.eval()
    
    return _model, _label_map

def recognize_plate(image_path: str) -> str:
    params = load_best_params()
    model, label_map = get_model_and_labels()

    image = load_image(image_path)
    if image is None or image.size == 0:
        raise ValueError(f"Invalid image loaded from {image_path}")
    
    # Preprocessing
    resized = resize_image(image)
    gray_pre = convert_to_grayscale(resized)
    enhanced = enhance_contrast(gray_pre, alpha=1.3, beta=10)
    denoised = reduce_noise(enhanced, method='bilateral')
    resized = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    try:
        plate = detect_plate(resized, params)
    except RuntimeError:
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

    char_boxes = segment_characters_hybrid(cleaned_plate, cleaned_plate.shape, params)
    characters = extract_and_resize_characters(
        cleaned_plate, char_boxes, target_size=(28, 28)
    )
    
    if len(characters) == 0:
        raise RuntimeError("No characters detected on license plate")

    # PyTorch Inference pipeline
    preds = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    with torch.no_grad():
        for char_img in characters:
            # char_img is equivalent to a 28x28 numpy array (uint8)
            # Create PIL image
            pil_img = Image.fromarray(char_img)
            # Transform
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)
            
            idx = predicted.item()
            if idx in label_map:
                preds.append(label_map[idx])
            else:
                preds.append('?')

    plate_text = "".join(preds)
    
    # Post-processing
    plate_text = filter_valid_characters(plate_text)
    plate_text = validate_and_correct_plate(plate_text)
    
    return plate_text