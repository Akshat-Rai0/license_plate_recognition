import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

def train_model(X, y):
    """
    Train SVM classifier on HOG features with regularization.
    """
    # Pre-process features to prevent scaler issues
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle zero variance features
    feature_std = np.std(X, axis=0)
    feature_std[feature_std == 0] = 1.0  # Set zero std to 1 to avoid division
    
    # Encode labels if they are strings
    label_encoder = None
    if y.dtype == object or y.dtype.kind in ('U', 'S'):  # String type
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Use StandardScaler with better settings
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", LinearSVC(C=0.1, max_iter=10000, dual=False, tol=1e-4))
    ])

    model.fit(X, y)
    
    # Store label encoder with the model for decoding predictions
    if label_encoder is not None:
        model.label_encoder_ = label_encoder
    
    return model


def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)