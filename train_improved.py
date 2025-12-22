#!/usr/bin/env python3
"""Improved training script with all enhancements."""

import os
import cv2
import numpy as np
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.features_improved import extract_features_from_characters
from src.model_improved import train_ensemble_model, save_model, cross_validate_model, train_simple_model
from src.augmentation import augment_dataset


def load_character_dataset(dataset_dir: str, target_size=(28, 28)):
    """Load character images from directory structure."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    images = []
    labels = []
    
    print(f"Loading dataset from: {dataset_dir}")
    
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"No character subdirectories found in {dataset_dir}")
    
    print(f"Found {len(class_dirs)} character classes")
    
    for class_dir in sorted(class_dirs):
        label = class_dir.name
        
        image_files = list(class_dir.glob("*.png")) + \
                     list(class_dir.glob("*.jpg")) + \
                     list(class_dir.glob("*.jpeg"))
        
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        print(f"Loading class '{label}': {len(image_files)} images")
        
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                continue
            
            img = cv2.resize(img, target_size)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            images.append(img)
            labels.append(label)
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Unique characters: {len(set(labels))}")
    
    return images, labels


def train_improved_model(dataset_dir: str,
                        output_model_path: str = "models/char_model_improved.pkl",
                        use_augmentation: bool = True,
                        augmentations_per_image: int = 3,
                        use_ensemble: bool = True,
                        use_combined_features: bool = True,
                        validation_split: float = 0.2):
    """Train improved character recognition model."""
    print("=" * 80)
    print("IMPROVED CHARACTER RECOGNITION MODEL TRAINING")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Augmentation: {use_augmentation}")
    print(f"  - Augmentations per image: {augmentations_per_image}")
    print(f"  - Ensemble model: {use_ensemble}")
    print(f"  - Combined features: {use_combined_features}")
    print(f"  - Validation split: {validation_split}")
    print("=" * 80)
    print()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    
    print("Step 1: Loading dataset...")
    images, labels = load_character_dataset(dataset_dir)
    
    if use_augmentation:
        print("\nStep 2: Augmenting dataset...")
        images, labels = augment_dataset(images, labels, augmentations_per_image, mode='advanced')
    else:
        print("\nStep 2: Skipping augmentation")
    
    print("\nStep 3: Extracting features...")
    features = extract_features_from_characters(images, use_combined=use_combined_features)
    print(f"Feature shape: {features.shape}")
    
    labels = np.array(labels)
    
    print("\nStep 4: Training model...")
    if use_ensemble:
        model = train_ensemble_model(features, labels, validation_split=validation_split)
    else:
        print("Training simple SVM model...")
        model = train_simple_model(features, labels)
    
    print("\nStep 5: Cross-validation...")
    cv_scores = cross_validate_model(model, features, labels, cv=5)
    
    print("\nStep 6: Saving model...")
    save_model(model, output_model_path)
    
    # Save configuration
    config = {
        'use_augmentation': use_augmentation,
        'augmentations_per_image': augmentations_per_image,
        'use_ensemble': use_ensemble,
        'use_combined_features': use_combined_features,
        'num_samples': len(images),
        'num_classes': len(set(labels)),
        'feature_dim': features.shape[1],
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std())
    }
    
    config_path = output_model_path.replace('.pkl', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model saved to: {output_model_path}")
    print(f"Final CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("=" * 80)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train improved character recognition model")
    parser.add_argument('dataset_dir', type=str, help="Path to character dataset directory")
    parser.add_argument('--output', type=str, default="models/char_model_improved.pkl",
                       help="Output model path")
    parser.add_argument('--no-augmentation', action='store_true',
                       help="Disable data augmentation")
    parser.add_argument('--augmentations', type=int, default=3,
                       help="Number of augmentations per image")
    parser.add_argument('--simple', action='store_true',
                       help="Use simple SVM instead of ensemble")
    parser.add_argument('--basic-features', action='store_true',
                       help="Use only HOG features (faster)")
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help="Validation split fraction")
    
    args = parser.parse_args()
    
    train_improved_model(
        dataset_dir=args.dataset_dir,
        output_model_path=args.output,
        use_augmentation=not args.no_augmentation,
        augmentations_per_image=args.augmentations,
        use_ensemble=not args.simple,
        use_combined_features=not args.basic_features,
        validation_split=args.validation_split
    )
