import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import joblib
import json
from src.pytorch_model import SimpleCNN

# Configuration
DATA_DIR = "data/raw/nigerian_plates/training_data/train20X20"
MODEL_SAVE_PATH = "models/char_cnn.pth"
LABEL_ENCODER_PATH = "models/label_encoder.json"
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 28 # Pipeline expects 28x28

def train():
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Transforms
    # Images in dataset might be grayscale or RGB. We need grayscale 1 channel.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
    ])

    # Load Data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        return

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")

    # Save class mapping
    class_mapping = {i: cls_name for i, cls_name in enumerate(classes)}
    with open(LABEL_ENCODER_PATH, 'w') as f:
        json.dump(class_mapping, f)
    print(f"Saved class mapping to {LABEL_ENCODER_PATH}")

    # Split Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Model
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {running_loss/len(train_loader):.4f} Acc: {epoch_acc:.2f}%")

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {val_acc:.2f}%")

    # Save Model
    # Save state dict for portability
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
