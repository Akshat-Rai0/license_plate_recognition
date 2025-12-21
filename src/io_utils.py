# in this file we handle the input output of the image
import cv2# computer vision 
import numpy as np
from pathlib import Path

def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk.
    Returns BGR image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path)) # converts the image into numpy array of height , widht , RGB or number of channels 
    """
    Opens the file at the given path
    Decodes the image bytes (JPEG/PNG/etc.)
    Converts it into a NumPy array
    Stores pixel values as integers (0–255)
    """
    if image is None:
        raise ValueError("Failed to load image")

    return image


# one can also use  image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) to read the given image directly into grayscale 
