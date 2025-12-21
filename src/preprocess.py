import cv2
import numpy as np
from typing import List 
def resize_image(image: np.ndarray, target_width: int = 600) -> np.ndarray:
    """
    Resize image while preserving aspect ratio.
    """
    h, w = image.shape[:2] # image.shape = (600, 800, 3) this is how we may get an image so 600 will be the height , 800 will be width and 3 is RGB 
    scale = target_width / w # in this we calculate what should be the height by using target width which is set to be 600 
    new_height = int(h * scale) # now we can have a new height that should fit the scale 

    resized = cv2.resize(image, (target_width, new_height)) #the new image after resize would be represented by this it converts the image into new size
    return resized



def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to grayscale.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # the image size is(600, 800, 3)  this function converts into whatever the resized image return (600,800) as grayscale only have height and width 
    # the color information is reduced to the shades of grey



# one can also use  image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) to read the given image directly into grayscale while loading the image 



"""
OpenCV uses a weighted sum of the B, G, R channels to calculate the grayscale intensity for each pixel. The formula is:

y=0.299⋅𝑅+0.58⋅𝐺+0.114⋅𝐵
Y=0.299⋅R+0.587⋅G+0.114⋅B

R, G, B are the values of the red, green, and blue channels of a pixel (0- 255).

Y is the resulting grayscale value (0 - 255).

Why these weights?

Human eyes are more sensitive to green light, moderately sensitive to red, and least sensitive to blue.

The weights reflect this sensitivity, so the grayscale image looks natural.

Example:

Suppose a pixel has values:

B = 50, G = 100, R = 200


Grayscale value:

y=0.114*50+0.587*100+0.299*200≈0+58.7+59.8≈118
Y=0.114*50+0.587*100+0.299*200≈0+58.7+59.8≈118

So the pixel in the grayscale image will have value 118.

Resulting image:

Original shape: (600, 800, 3)

Grayscale shape: (600, 800)

Only intensity information is kept; no color.

"""
def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    Enhance image contrast (from Nigerian repo approach).
    alpha: contrast control (1.0-3.0)
    beta: brightness control (0-100)
    """
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced


def reduce_noise(image: np.ndarray, method: str = 'gaussian') -> np.ndarray:
    """
    Reduce noise in image (from Nigerian repo approach).
    Methods: 'gaussian', 'median', 'bilateral'
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 5)
    elif method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    return image


def return_edges(image: np.ndarray, 
                  blur_kernel: tuple = (3,3), 
                  canny_threshold1: int = 30, 
                  canny_threshold2: int = 200) -> np.ndarray:

    blurred = cv2.GaussianBlur(image, blur_kernel, sigmaX=0) # Syntax: cv2.GaussianBlur(src, ksize, sigmaX) src is the image , ksize is the kernel size larger value more blur
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    return edges

    """
    The algorithm looks at each pixel and calculates how much the brightness changes around it (the "gradient"):

    Above upper threshold: "Definitely an edge!" ✅
    Below lower threshold: "Definitely NOT an edge" ❌
    Between the two thresholds: "Maybe an edge... only if it's connected to a definite edge" 🤔
    """


