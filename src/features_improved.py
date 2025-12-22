import numpy as np
from skimage.feature import hog, local_binary_pattern
from scipy.ndimage import gaussian_filter

def extract_hog_features(character_img: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a single character image.
    """
    # Ensure image is in correct format
    if character_img.dtype != np.uint8:
        character_img = character_img.astype(np.uint8)
    
    # Ensure image is 2D
    if len(character_img.shape) > 2:
        character_img = character_img[:, :, 0] if len(character_img.shape) == 3 else character_img
    
    # Ensure minimum size
    if character_img.size == 0:
        return np.zeros(1764)
    
    try:
        hog_vector = hog(
            character_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
        
        # Replace any NaN or Inf values with 0
        hog_vector = np.nan_to_num(hog_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize to prevent extreme values (L2 normalization)
        norm = np.linalg.norm(hog_vector)
        if norm > 0:
            hog_vector = hog_vector / norm
        
        # Clip to safe range
        hog_vector = np.clip(hog_vector, -1.0, 1.0)
        
        return hog_vector
    except Exception as e:
        print(f"Warning: HOG extraction failed: {e}")
        # Return zero vector of expected size
        return np.zeros(1764)


def extract_lbp_features(character_img: np.ndarray) -> np.ndarray:
    """
    Extract Local Binary Pattern (texture) features.
    """
    try:
        # Ensure correct format
        if character_img.dtype != np.uint8:
            character_img = character_img.astype(np.uint8)
        
        if len(character_img.shape) > 2:
            character_img = character_img[:, :, 0]
        
        # Compute LBP
        lbp = local_binary_pattern(character_img, P=8, R=1, method='uniform')
        
        # Compute histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
        
        # Normalize
        lbp_hist = lbp_hist.astype(float)
        total = np.sum(lbp_hist)
        if total > 0:
            lbp_hist = lbp_hist / total
        
        return lbp_hist
    except Exception as e:
        print(f"Warning: LBP extraction failed: {e}")
        return np.zeros(59)


def extract_zoning_features(character_img: np.ndarray, zones=(4, 4)) -> np.ndarray:
    """
    Extract zoning features by dividing image into zones and computing statistics.
    """
    try:
        if character_img.dtype != np.uint8:
            character_img = character_img.astype(np.uint8)
        
        if len(character_img.shape) > 2:
            character_img = character_img[:, :, 0]
        
        h, w = character_img.shape
        zone_h, zone_w = h // zones[0], w // zones[1]
        
        features = []
        for i in range(zones[0]):
            for j in range(zones[1]):
                y_start = i * zone_h
                y_end = (i + 1) * zone_h if i < zones[0] - 1 else h
                x_start = j * zone_w
                x_end = (j + 1) * zone_w if j < zones[1] - 1 else w
                
                zone = character_img[y_start:y_end, x_start:x_end]
                
                # Multiple statistics per zone
                features.append(np.mean(zone))
                features.append(np.std(zone))
                features.append(np.sum(zone < 128) / zone.size)  # Density of dark pixels
        
        return np.array(features)
    except Exception as e:
        print(f"Warning: Zoning extraction failed: {e}")
        return np.zeros(zones[0] * zones[1] * 3)


def extract_projection_features(character_img: np.ndarray) -> np.ndarray:
    """
    Extract horizontal and vertical projection features.
    """
    try:
        if character_img.dtype != np.uint8:
            character_img = character_img.astype(np.uint8)
        
        if len(character_img.shape) > 2:
            character_img = character_img[:, :, 0]
        
        h, w = character_img.shape
        
        # Horizontal projection (sum along rows)
        h_proj = np.sum(character_img, axis=1) / w
        
        # Vertical projection (sum along columns)
        v_proj = np.sum(character_img, axis=0) / h
        
        # Normalize
        h_proj = h_proj / 255.0
        v_proj = v_proj / 255.0
        
        # Combine
        projection_features = np.concatenate([h_proj, v_proj])
        
        return projection_features
    except Exception as e:
        print(f"Warning: Projection extraction failed: {e}")
        return np.zeros(character_img.shape[0] + character_img.shape[1])


def extract_contour_features(character_img: np.ndarray) -> np.ndarray:
    """
    Extract shape-based features from character contours.
    """
    import cv2
    
    try:
        if character_img.dtype != np.uint8:
            character_img = character_img.astype(np.uint8)
        
        if len(character_img.shape) > 2:
            character_img = character_img[:, :, 0]
        
        # Threshold to binary
        _, binary = cv2.threshold(character_img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return np.zeros(7)
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Extract features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        
        # Hu moments (shape descriptors)
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        else:
            hu_moments = np.zeros(7)
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        features = np.array([
            area / 784.0,  # Normalize by max area (28*28)
            perimeter / 112.0,  # Normalize by max perimeter (4*28)
            aspect_ratio,
            extent,
            circularity
        ])
        
        # Combine with first 2 Hu moments (most discriminative)
        features = np.concatenate([features, hu_moments[:2]])
        
        return features
    except Exception as e:
        print(f"Warning: Contour extraction failed: {e}")
        return np.zeros(7)


def extract_combined_features(character_img: np.ndarray) -> np.ndarray:
    """
    Extract multiple feature types and combine them for better accuracy.
    Combines HOG, LBP, Zoning, Projection, and Contour features.
    """
    # 1. HOG features (main features)
    hog_features = extract_hog_features(character_img)
    
    # 2. LBP features (texture)
    lbp_features = extract_lbp_features(character_img)
    
    # 3. Zoning features (spatial layout)
    zoning_features = extract_zoning_features(character_img, zones=(4, 4))
    
    # 4. Projection features
    projection_features = extract_projection_features(character_img)
    
    # 5. Contour features (shape)
    contour_features = extract_contour_features(character_img)
    
    # Combine all features
    combined = np.concatenate([
        hog_features,        # ~1764 features
        lbp_features,        # 59 features
        zoning_features,     # 48 features (4x4 zones, 3 stats each)
        projection_features, # 56 features (28+28)
        contour_features     # 7 features
    ])
    
    # Final safety check
    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    combined = np.clip(combined, -10.0, 10.0)
    
    return combined


def extract_features_from_characters(character_images: list, use_combined: bool = True) -> np.ndarray:
    """
    Convert list of character images to feature matrix.
    
    Args:
        character_images: List of character images
        use_combined: If True, use combined features (more accurate but slower)
                     If False, use only HOG features (faster)
    """
    features = []

    for char_img in character_images:
        if use_combined:
            feature_vector = extract_combined_features(char_img)
        else:
            feature_vector = extract_hog_features(char_img)
        features.append(feature_vector)

    feature_matrix = np.array(features)

    # Handle case where feature_matrix might be empty
    if feature_matrix.size == 0:
        return np.array([])
    
    # Final check for NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure no zero-variance features (add small epsilon to prevent divide by zero)
    feature_std = np.std(feature_matrix, axis=0)
    feature_std = np.where(feature_std == 0, 1e-8, feature_std)  # Prevent divide by zero
    
    # Normalize features
    feature_mean = np.mean(feature_matrix, axis=0)
    feature_matrix = (feature_matrix - feature_mean) / feature_std
    
    # Final clip to safe range
    feature_matrix = np.clip(feature_matrix, -10.0, 10.0)
    
    return feature_matrix
