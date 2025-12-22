"""Multi-scale plate detection with confidence scoring."""

import cv2
import numpy as np

def calculate_plate_confidence(plate: np.ndarray) -> float:
    """Calculate confidence score for detected plate."""
    if plate is None or plate.size == 0:
        return 0.0
    
    h, w = plate.shape[:2]
    
    # Aspect ratio score
    aspect_ratio = w / h if h > 0 else 0
    ar_score = 1.0 - min(abs(aspect_ratio - 3.5) / 3.5, 1.0)
    
    # Size score
    area = w * h
    size_score = min(area / 15000, 1.0)
    
    # Edge density
    if len(plate.shape) == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (w * h)
    edge_score = min(edge_density * 10, 1.0)
    
    confidence = ar_score * 0.4 + size_score * 0.3 + edge_score * 0.3
    return confidence


def detect_plate_multiscale(image: np.ndarray, 
                           detect_func,
                           params: dict,
                           scales: list = None) -> tuple:
    """Try detection at multiple scales."""
    if scales is None:
        scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    
    best_plate = None
    best_confidence = 0.0
    best_scale = 1.0
    
    h, w = image.shape[:2]
    
    for scale in scales:
        try:
            new_w, new_h = int(w * scale), int(h * scale)
            scaled = cv2.resize(image, (new_w, new_h))
            
            plate = detect_func(scaled, params)
            
            if plate is not None and plate.size > 0:
                confidence = calculate_plate_confidence(plate)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_plate = plate
                    best_scale = scale
        except Exception:
            continue
    
    if best_plate is None:
        raise RuntimeError("No plate detected at any scale")
    
    return best_plate, best_confidence, best_scale


def detect_plate_robust(image: np.ndarray,
                       detect_func,
                       params: dict,
                       use_multiscale: bool = True,
                       use_preprocessing: bool = True) -> tuple:
    """Robust detection combining strategies."""
    results = []
    
    # Strategy 1: Direct detection
    try:
        plate = detect_func(image, params)
        confidence = calculate_plate_confidence(plate)
        results.append((plate, confidence, {'strategy': 'direct', 'scale': 1.0}))
    except Exception:
        pass
    
    # Strategy 2: Multi-scale
    if use_multiscale:
        try:
            plate, confidence, scale = detect_plate_multiscale(image, detect_func, params)
            results.append((plate, confidence, {'strategy': 'multiscale', 'scale': scale}))
        except Exception:
            pass
    
    if not results:
        raise RuntimeError("All detection strategies failed")
    
    best = max(results, key=lambda x: x[1])
    return best
