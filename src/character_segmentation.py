# getting tha characters from the plate which we get from preprocessing

import cv2 
import numpy as np



def binarize_plate(plate_gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale plate to binary image using adaptive thresholding.
    """

    # apply adaptive thresholding
    # invert result so characters become white on black background
    binary_plate = cv2.adaptiveThreshold(
        plate_gray,                          # Source image (must be grayscale)
        255,                                 # Maximum value (white)
        cv2.ADAPTIVE_THRESH_MEAN_C,         # Adaptive method
        cv2.THRESH_BINARY_INV,              # Threshold type
        11,                                  # Block size (must be odd: 3, 5, 7, 9, 11, etc.)
        12                                   # Constant C (subtracted from mean)
    )

    return binary_plate


def binarize_plate_otsu(plate_gray: np.ndarray) -> np.ndarray:
    """
    Binarize plate using Otsu's method (alternative approach).
    """
    _, binary_plate = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_plate


def binarize_plate_hybrid(plate_gray: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """
    Hybrid binarization: tries adaptive first, falls back to Otsu if needed.
    """
    if method == 'adaptive':
        return binarize_plate(plate_gray)
    elif method == 'otsu':
        return binarize_plate_otsu(plate_gray)
    else:
        # Try adaptive first
        try:
            binary = binarize_plate(plate_gray)
            # Check if result is reasonable (has some white pixels)
            if np.sum(binary == 255) > 100:
                return binary
        except:
            pass
        
        # Fallback to Otsu
        return binarize_plate_otsu(plate_gray)








def clean_binary_plate(binary_plate: np.ndarray, kernel_size: tuple = (3, 3)) -> np.ndarray:
    """
    Remove noise and enhance character shapes.
    """

    # create rectangular kernel
    # apply morphological close
    # apply morphological open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    closed = cv2.morphologyEx(binary_plate, cv2.MORPH_CLOSE, kernel)
    cleaned_plate = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return cleaned_plate






def find_character_contours(cleaned_plate: np.ndarray) -> list:
    """
    Detect contours that may correspond to characters.
    """

    # find external contours in binary image
    contours, hierarchy = cv2.findContours(
        cleaned_plate,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    return contours





def filter_character_contours(contours: list,
                              plate_shape: tuple) -> list:
    """
    Keep only contours that resemble characters.
    """

    plate_height, plate_width = plate_shape[:2]
    valid_character_boxes = []

    # for each contour:
    #   compute bounding box
    #   compute height ratio (h / plate_height)
    #   compute aspect ratio (w / h)
    #   compute contour area
    #   apply threshold rules
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        height_ratio = h / plate_height
        aspect_ratio = w / h if h != 0 else 0

        if (
        area > 100 and                    # Must be large enough (not noise)
        0.4 < height_ratio < 0.95 and    # Must be 40-95% of plate height
        0.2 < aspect_ratio < 1.0          # Must have character like proportions
            ):
            valid_character_boxes.append((x, y, w, h))
        # If all conditions pass, this is likely a character
    return valid_character_boxes




def sort_characters_left_to_right(char_boxes: list) -> list:
    """
    Sort bounding boxes based on x-position.
    """

    # sort boxes by x coordinate
    sorted_boxes = sorted(char_boxes, key=lambda box: box[0])
    return sorted_boxes




def extract_and_resize_characters(binary_plate: np.ndarray,
                                  char_boxes: list,
                                  target_size: tuple = (28, 28)) -> list:
    """
    Crop each character and resize to fixed size.
    """

    character_images = []

    # crop each bounding box from binary image
    # resize to target_size
    # store in list
    for (x, y, w, h) in char_boxes:
        # Loop through each bounding box character location
        char = binary_plate[y:y+h, x:x+w]
    # Crop/extract the character from the plate image
        resized_char = cv2.resize(char, target_size)
    # Resize the character to a standard size
        character_images.append(resized_char)


    return character_images


def segment_characters_horizontal_projection(binary_plate: np.ndarray) -> list:
    """
    Segment characters using horizontal projection method (from Nigerian repo approach).
    This method sums pixel values row-wise to find character boundaries.
    
    Args:
        binary_plate: Binary plate image (white text on black background)
        
    Returns:
        List of character bounding boxes [(x, y, w, h), ...]
    """
    plate_height, plate_width = binary_plate.shape[:2]
    
    # Horizontal projection: sum pixel values along each row
    horizontal_projection = np.sum(binary_plate == 255, axis=1)  # Count white pixels (text)
    
    # Find threshold for text rows
    mean_projection = np.mean(horizontal_projection)
    threshold = mean_projection * 0.3  # Adjust threshold
    
    # Find rows with text
    text_rows = horizontal_projection > threshold
    
    if np.sum(text_rows) == 0:
        return []
    
    # Find top and bottom of text region
    text_row_indices = np.where(text_rows)[0]
    y_start = max(0, text_row_indices[0] - 2)
    y_end = min(plate_height, text_row_indices[-1] + 2)
    
    # Extract text region
    text_region = binary_plate[y_start:y_end, :]
    
    # Now do vertical projection on text region to find individual characters
    vertical_projection = np.sum(text_region == 255, axis=0)
    
    # Find character boundaries (valleys in projection)
    char_threshold = np.mean(vertical_projection) * 0.2
    char_regions = vertical_projection > char_threshold
    
    # Find connected character regions
    char_boxes = []
    in_char = False
    char_start = 0
    
    for i, is_char in enumerate(char_regions):
        if is_char and not in_char:
            # Start of character
            char_start = i
            in_char = True
        elif not is_char and in_char:
            # End of character
            char_end = i
            char_width = char_end - char_start
            
            # Filter out very narrow regions (noise)
            if char_width > 5:  # Minimum character width
                char_boxes.append((char_start, y_start, char_width, y_end - y_start))
            in_char = False
    
    # Handle case where last character extends to edge
    if in_char:
        char_end = len(char_regions)
        char_width = char_end - char_start
        if char_width > 5:
            char_boxes.append((char_start, y_start, char_width, y_end - y_start))
    
    return char_boxes


def filter_character_contours_tunable(contours, plate_shape, params):
    """Character filtering with parameters - moved here to avoid circular imports"""
    plate_height, plate_width = plate_shape[:2]
    valid_character_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        height_ratio = h / plate_height
        aspect_ratio = w / h if h != 0 else 0

        if (
            area > params.get('char_min_area', 100) and
            params.get('height_ratio_min', 0.3) < height_ratio < params.get('height_ratio_max', 0.95) and
            params.get('aspect_ratio_min', 0.15) < aspect_ratio < params.get('aspect_ratio_max', 1.2)
        ):
            valid_character_boxes.append((x, y, w, h))
    
    return valid_character_boxes


def segment_characters_hybrid(binary_plate: np.ndarray, plate_shape: tuple, params) -> list:
    """
    Hybrid character segmentation: tries contour method first, falls back to horizontal projection.
    Combines the best of both approaches from Nigerian repo and contour-based methods.
    """
    # Try contour-based segmentation first
    contours = find_character_contours(binary_plate)
    char_boxes = filter_character_contours_tunable(contours, plate_shape, params)
    
    if len(char_boxes) >= 6:  # If we found reasonable number of characters
        return sort_characters_left_to_right(char_boxes)
    
    # Fallback to horizontal projection method (Nigerian repo approach)
    try:
        char_boxes_proj = segment_characters_horizontal_projection(binary_plate)
        if len(char_boxes_proj) > 0:
            return sort_characters_left_to_right(char_boxes_proj)
    except Exception:
        pass
    
    # If both methods fail, return what we have
    return sort_characters_left_to_right(char_boxes)
