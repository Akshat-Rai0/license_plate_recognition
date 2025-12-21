import numpy as np
from skimage.feature import hog

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


def extract_features_from_characters(character_images: list) -> np.ndarray:
    """
    Convert list of character images to feature matrix.
    """
    features = []

    for char_img in character_images:
        hog_vector = extract_hog_features(char_img)
        features.append(hog_vector)

    feature_matrix = np.array(features)
    
    # Final check for NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure no zero-variance features (add small epsilon to prevent divide by zero)
    feature_std = np.std(feature_matrix, axis=0)
    feature_std[feature_std == 0] = 1e-8  # Prevent divide by zero
    
    # Normalize features
    feature_mean = np.mean(feature_matrix, axis=0)
    feature_matrix = (feature_matrix - feature_mean) / feature_std
    
    # Final clip to safe range
    feature_matrix = np.clip(feature_matrix, -10.0, 10.0)
    
    return feature_matrix