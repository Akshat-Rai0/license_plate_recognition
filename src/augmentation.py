import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import random

def elastic_transform(image, alpha=5, sigma=3, random_state=None):
    """Elastic deformation of images."""
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    return distorted.astype(np.uint8)


def augment_character_advanced(image):
    """Advanced augmentation with comprehensive transformations."""
    augmented = image.copy()
    
    # Rotation
    if random.random() > 0.3:
        angle = random.uniform(-15, 15)
        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=255)
    
    # Scaling
    if random.random() > 0.5:
        scale = random.uniform(0.85, 1.15)
        h, w = augmented.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(augmented, (new_w, new_h))
        augmented = cv2.resize(scaled, (w, h))
    
    # Brightness and contrast
    if random.random() > 0.4:
        alpha = random.uniform(0.7, 1.3)
        beta = random.randint(-30, 30)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)
    
    # Elastic deformation
    if random.random() > 0.6:
        augmented = elastic_transform(augmented, alpha=5, sigma=3)
    
    # Gaussian noise
    if random.random() > 0.5:
        noise = np.random.normal(0, random.uniform(5, 15), augmented.shape)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return augmented


def augment_dataset(images, labels, augmentations_per_image=3, mode='advanced'):
    """Augment entire dataset."""
    print(f"Augmenting dataset with {augmentations_per_image} variations per image...")
    
    augmented_images = list(images)
    augmented_labels = list(labels)
    
    total = len(images) * augmentations_per_image
    count = 0
    
    for img, label in zip(images, labels):
        for _ in range(augmentations_per_image):
            aug_img = augment_character_advanced(img.copy())
            augmented_images.append(aug_img)
            augmented_labels.append(label)
            
            count += 1
            if count % 500 == 0:
                print(f"Progress: {count}/{total} augmentations created")
    
    print(f"Dataset size: {len(images)} -> {len(augmented_images)}")
    if len(images) > 0:
        print(f"Augmentation factor: {len(augmented_images) / len(images):.1f}x")
    else:
        print("Augmentation factor: N/A (no original images to augment)")
    
    return augmented_images, augmented_labels
