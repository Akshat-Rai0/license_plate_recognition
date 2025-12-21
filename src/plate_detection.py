import cv2
import numpy as np

def get_contours(edges: np.ndarray) -> list:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
"""
. mode (required) - Contour Retrieval Mode

cv2.RETR_EXTERNAL - Retrieves only the outer contours  # THIS WAS USED
cv2.RETR_LIST - Retrieves all contours without hierarchy
cv2.RETR_CCOMP - Retrieves all contours, organized into 2-level hierarchy
cv2.RETR_TREE - Retrieves all contours with full hierarchy (parent-child relationships)

. method (required) - Contour Approximation Method

cv2.CHAIN_APPROX_NONE - Stores all contour points (more memory)
cv2.CHAIN_APPROX_SIMPLE - Compresses horizontal, vertical, and diagonal segments (saves memory)  # THIS WAS USED
cv2.CHAIN_APPROX_TC89_L1 - Applies Teh-Chin chain approximation algorithm
cv2.CHAIN_APPROX_TC89_KCOS - Applies Teh-Chin chain approximation algorithm

"""




def draw_contours(image: np.ndarray, contours: list)-> np.ndarray:
    """
    Draw contours on a copy of the original image
    """
    output = image.copy() # creates a new image so original doesnt get modified
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)  # Green contours, thickness=2
    return output




def filter_contours(contours: list, min_area: float = 500) -> list:
    """
    Keep only contours with area greater than min_area.
    """
    filtered = [
        c for c in contours 
        if min_area < cv2.contourArea(c) < 50000
    ]
    filtered.sort(key=cv2.contourArea, reverse=True)
    return filtered



def get_bounding_boxes(contours: list) -> list: # Gets rectangular bounding boxes around each contour.
    """
    Returns a list of bounding rectangles for each contour.
    Each box is (x, y, w, h)
    """
    return [cv2.boundingRect(c) for c in contours]



def clean_plate(binary: np.ndarray,
                kernel_size: tuple = (5, 5)) -> np.ndarray:
    """
    Cleans binary plate image using morphological operations.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
#MORPH_CLOSE: Fills small holes and gaps (connects broken edges)
#MORPH_OPEN: Removes small noise/dots (smooths the image)
    return opened




def get_plate_corners(contour: np.ndarray):
    """
    Approximates contour to a 4-point polygon.
    Includes aspect ratio checks and fallbacks.
    """
    # 1. Aspect Ratio Check
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    
    # Typical plates are wide. Allow range 2.0 to 10.0 (roughly)
    # Some plates might be stacked or have holders making them taller (ratio ~1.5)
    # We'll be lenient: 1.5 to 15
    if aspect_ratio < 1.5 or aspect_ratio > 15:
        return None

    # 2. Try Approx Poly
    peri = cv2.arcLength(contour, True) 
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True) 

    if len(approx) == 4: 
        return approx
    
    # 3. Fallback: Min Area Rect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    return box








def order_points(pts: np.ndarray) -> np.ndarray:#Sorts 4 corner points into a consistent order: top-left, top-right, bottom-right, bottom-left.
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect

    """
    Top-left: Point with smallest sum (x+y is smallest)
    Bottom-right: Point with largest sum (x+y is largest)
    Top-right: Point with smallest difference (x-y is smallest)
    Bottom-left: Point with largest difference (x-y is largest)

    """






def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.reshape(4, 2))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
#This function removes perspective distortion and returns a front-facing, rectangular view of the object




def normalize_orientation(plate: np.ndarray) -> np.ndarray:
    """
    Ensures width > height (landscape orientation).
    """
    h, w = plate.shape[:2]
    if h > w:
        plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
    return plate
#If height > width (portrait), rotate it 90° clockwise
#Otherwise, leave it as is


def detect_plate_vertical_projection(image: np.ndarray) -> np.ndarray:
    """
    Detect license plate using vertical projection method (from Nigerian repo approach).
    This method sums pixel values column-wise to find text regions.
    
    Args:
        image: Input grayscale or BGR image
        
    Returns:
        Detected plate region as numpy array
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Vertical projection: sum pixel values along each column
    vertical_projection = np.sum(binary == 0, axis=0)  # Count black pixels (text)
    
    # Find threshold for text regions
    mean_projection = np.mean(vertical_projection)
    threshold = mean_projection * 0.5  # Adjust threshold as needed
    
    # Find regions with high projection (text regions)
    text_regions = vertical_projection > threshold
    
    # Find start and end of plate region
    plate_columns = np.where(text_regions)[0]
    
    if len(plate_columns) == 0:
        raise RuntimeError("No plate region found using vertical projection")
    
    x_start = max(0, plate_columns[0] - 10)  # Add padding
    x_end = min(image.shape[1], plate_columns[-1] + 10)
    
    # Extract plate region (full height)
    plate = image[:, x_start:x_end]
    
    return plate


def detect_plate_hybrid(image: np.ndarray, params, return_coords: bool = False):
    """
    Hybrid plate detection: tries contour method first, falls back to vertical projection.
    This function needs to be called from pipeline.py where detect_plate is defined.
    """
    # This will be implemented in pipeline.py to avoid circular imports
    pass



