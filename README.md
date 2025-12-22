# License Plate Recognition System

An automated license plate recognition (ALPR) system that detects and reads license plates from images using computer vision and machine learning.

## Features

- **Automatic Plate Detection**: Detects license plates from vehicle images
- **Character Segmentation**: Segments individual characters from detected plates
- **OCR Recognition**: Recognizes alphanumeric characters using trained SVM models
- **Debug Tools**: Visual debugging tools to inspect each processing step
- **Retraining Pipeline**: Easy workflow to add new training data and improve accuracy

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd license_plate_recognition
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Recognize a License Plate

Debug mode (shows all processing steps):
```bash
./venv/bin/python debug_recognition.py data/raw/test_1.webp
```

Output files will be saved to `debug_output/` showing each processing step.

### Add Training Data

To improve model accuracy with your own labeled images:

```bash
./venv/bin/python add_training_data.py --image path/to/image.jpg --text "ABC123XY"
```

Options:
- `--image`: Path to the license plate image
- `--text`: Correct plate text (letters and numbers)
- `--dry-run`: Preview without saving files

### Retrain the Model

After adding training data:

```bash
./venv/bin/python train_improved.py data/raw/nigerian_plates/training_data/train20X20 --output models/char_svm.pkl --simple
```

Training options:
- `--simple`: Use simple SVM (faster, recommended)
- `--no-augmentation`: Skip data augmentation
- `--augmentations N`: Number of augmentations per image (default: 3)
- `--validation-split`: Validation fraction (default: 0.2)

### Diagnose Issues

To debug contour detection problems:

```bash
./venv/bin/python diagnose_contours.py data/raw/test_1.webp
```

## Project Structure

```
license_plate_recognition/
├── src/                          # Source code modules
│   ├── io_utils.py              # Image loading utilities
│   ├── preprocess.py            # Image preprocessing
│   ├── plate_detection.py       # Plate detection logic
│   ├── character_segmentation.py # Character segmentation
│   ├── features.py              # Basic feature extraction
│   ├── features_improved.py     # Advanced feature extraction
│   ├── model.py                 # Basic model training
│   ├── model_improved.py        # Ensemble model training
│   └── augmentation.py          # Data augmentation
├── data/                        # Dataset directory
│   └── raw/
│       └── nigerian_plates/
│           └── training_data/   # Training character images
├── models/                      # Trained models
│   └── char_svm.pkl            # Character recognition model
├── debug_output/               # Debug visualization output
├── debug_recognition.py        # Main debug script
├── add_training_data.py        # Add labeled data script
├── train_improved.py           # Model training script
├── diagnose_contours.py        # Contour diagnosis tool
└── best_params.json            # Optimized parameters

```

## How It Works

1. **Preprocessing**: Resize, grayscale conversion, contrast enhancement, noise reduction
2. **Plate Detection**: Edge detection → contour finding → aspect ratio filtering
3. **Binarization**: Adaptive thresholding and morphological operations
4. **Character Segmentation**: Contour-based character detection with filtering
5. **Feature Extraction**: Combined HOG, LBP, zoning, projection, and contour features (314 dimensions)
6. **Recognition**: Ensemble SVM classifier predicts each character

## Parameters

Key parameters in `best_params.json`:
- `blur_kernel`: Gaussian blur kernel size
- `canny_threshold1/2`: Canny edge detection thresholds
- `min_area`: Minimum contour area for plate candidates
- `ar_min/ar_max`: Aspect ratio range for plates
- `binarize_block_size/C`: Adaptive threshold parameters
- `char_min_area`: Minimum character area
- `height_ratio_min/max`: Character height ratio filters
- `aspect_ratio_min/max`: Character aspect ratio filters

## Troubleshooting

### No characters detected
- Run `diagnose_contours.py` to visualize contours
- Adjust parameters in `best_params.json`
- Try more lenient character filtering parameters

### Feature mismatch error
- Ensure you're using `features_improved.py` with `use_combined=True`
- The model expects 314-dimensional feature vectors

### Plate not detected
- Check if aspect ratio is within range (1.5-6.0 by default)
- Adjust `min_area` and canny thresholds
- Ensure good image quality and lighting

## Model Performance

- **Feature Dimensions**: 314 (combined features)
- **Training Data**: Nigerian plate dataset + custom additions
- **Model Type**: Ensemble (SVM + Random Forest + Neural Network)
- **Target Size**: 20x20 pixels for character images

## Contributing

To add new training data:
1. Use `add_training_data.py` to add labeled images
2. Retrain the model with `train_improved.py`
3. Test improvements with `debug_recognition.py`

## License

[Specify your license here]

## Acknowledgments

- Nigerian Plates Dataset for initial training data
- scikit-learn for machine learning models
- OpenCV for computer vision operations
