import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

def train_ensemble_model(X, y, validation_split=0.2, use_grid_search=False):
    """
    Train an ensemble of models for better accuracy.
    Combines SVM, Random Forest, and Neural Network.
    """
    print("=" * 80)
    print("Training Ensemble Model")
    print("=" * 80)
    
    # Pre-process features
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode string labels to integers for MLPClassifier compatibility
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=validation_split, stratify=y_encoded, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of classes: {len(np.unique(y_encoded))}")
    print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    print()
    
    # Create base models
    scaler = StandardScaler(with_mean=True, with_std=True)
    
    # 1. SVM with RBF kernel
    svm_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=42))
    ])
    
    # 2. Random Forest
    rf_model = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            n_estimators=200, 
            max_depth=30, 
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # 3. Neural Network
    mlp_model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])
    
    # Train individual models
    print("Training individual models...")
    
    print("1. Training SVM...")
    svm_model.fit(X_train, y_train)
    svm_train_acc = svm_model.score(X_train, y_train)
    svm_val_acc = svm_model.score(X_val, y_val)
    print(f"   Train accuracy: {svm_train_acc:.4f}")
    print(f"   Validation accuracy: {svm_val_acc:.4f}")
    
    print("2. Training Random Forest...")
    rf_model.fit(X_train, y_train)
    rf_train_acc = rf_model.score(X_train, y_train)
    rf_val_acc = rf_model.score(X_val, y_val)
    print(f"   Train accuracy: {rf_train_acc:.4f}")
    print(f"   Validation accuracy: {rf_val_acc:.4f}")
    
    print("3. Training Neural Network...")
    mlp_model.fit(X_train, y_train)
    mlp_train_acc = mlp_model.score(X_train, y_train)
    mlp_val_acc = mlp_model.score(X_val, y_val)
    print(f"   Train accuracy: {mlp_train_acc:.4f}")
    print(f"   Validation accuracy: {mlp_val_acc:.4f}")
    print()
    
    # Create ensemble with weighted voting based on validation accuracy
    # Normalize weights to sum to number of models (for proper VotingClassifier weighting)
    total_acc = svm_val_acc + rf_val_acc + mlp_val_acc
    if total_acc > 0:
        weights = [svm_val_acc / total_acc * 3, rf_val_acc / total_acc * 3, mlp_val_acc / total_acc * 3]
    else:
        weights = [1.0, 1.0, 1.0]  # Equal weights if all accuracies are 0
    
    print(f"Ensemble weights (SVM, RF, MLP): {[f'{w:.2f}' for w in weights]}")
    
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('rf', rf_model),
            ('mlp', mlp_model)
        ],
        voting='soft',
        weights=weights
    )
    
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    ensemble_train_acc = ensemble.score(X_train, y_train)
    ensemble_val_acc = ensemble.score(X_val, y_val)
    
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Ensemble train accuracy: {ensemble_train_acc:.4f}")
    print(f"Ensemble validation accuracy: {ensemble_val_acc:.4f}")
    print(f"Improvement over best individual: {ensemble_val_acc - max(svm_val_acc, rf_val_acc, mlp_val_acc):.4f}")
    print("=" * 80)
    
    # Store label encoder with the ensemble for decoding predictions
    ensemble.label_encoder_ = label_encoder
    
    return ensemble


def train_simple_model(X, y):
    """Train a simple but effective SVM classifier."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Encode labels if they are strings
    label_encoder = None
    if y.dtype == object or y.dtype.kind in ('U', 'S'):  # String type
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svm", SVC(C=10.0, kernel='rbf', gamma='scale', probability=True, random_state=42))
    ])

    model.fit(X, y)
    
    # Store label encoder with the model for decoding predictions
    if label_encoder is not None:
        model.label_encoder_ = label_encoder
    
    return model


def save_model(model, path: str):
    """Save trained model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def load_model(path: str):
    """Load trained model from disk."""
    return joblib.load(path)


def cross_validate_model(model, X, y, cv=5):
    """Perform cross-validation."""
    # Encode labels if they are strings and model has label encoder
    if hasattr(model, 'label_encoder_'):
        y = model.label_encoder_.transform(y)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    return scores
