# Add this function to train.py
def augment_image(img):
    """Apply data augmentation to character image"""
    import random
    
    # Random rotation
    angle = random.uniform(-5, 5)
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=255)
    
    # Random noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random scaling
    scale = random.uniform(0.9, 1.1)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    img = cv2.resize(img, (w, h))
    
    return img

# Then in load_dataset(), augment each image:
for file in image_files:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    images.append(img)
    labels.append(label)
    
    # Add augmented versions
    for _ in range(2):  # Add 2 augmented versions per image
        aug_img = augment_image(img.copy())
        images.append(aug_img)
        labels.append(label)