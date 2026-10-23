# Define your import statements...

import numpy as np
import os
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# Your constants...


# Your classes and functions here...

def load_pgm_image(filepath: str) -> np.ndarray:
    """
    Loads a PGM image from the BioID Face Database.
    
    The PGM format in the BioID database has a 4-line header.
    This function reads the header to get image dimensions and then
    reads the binary data into a NumPy array.
    Args:
        filepath (str): The full path to the .pgm file.
    Returns:
        np.ndarray: A 2D NumPy array representing the grayscale image.
    """
    with open(filepath, 'rb') as f:
        # 1. Read the "magic number" (e.g., P5)
        format = f.readline().strip()
        if format != b'P5':
            raise ValueError(f"File {filepath} is not a binary PGM file (P5).")
        # 2. Read width and height
        width = int(f.readline().strip())
        height = int(f.readline().strip())
        # 3. Read max gray value
        max_val = int(f.readline().strip())
        if max_val != 255:
            print(f"Warning: Max gray value is {max_val}, not 255.")
        # 4. Read the image data
        image_data = f.read(width * height)
        # Convert the binary data to a NumPy array and reshape it
        image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width))
    return image

def load_eye_positions(filepath: str) -> Tuple[int, int, int, int]:
    """
    Loads the eye positions from a .eye file.
    The file format is a header line followed by a line with four
    space-separated integers: Lx Ly Rx Ry.
    Args:
        filepath (str): The full path to the .eye file.
    Returns:
        A tuple containing four integers: (left_eye_x, left_eye_y, right_eye_x, right_eye_y).
    """
    with open(filepath, 'r') as f:
        # Skip the header line
        f.readline()
        # Read the line with coordinates
        coords_line = f.readline().strip()
        # Split the line by spaces and convert to integers
        coords = list(map(int, coords_line.split()))
    lx, ly, rx, ry = coords
    return (lx, ly, rx, ry)

def load_landmarks(filepath: str) -> np.ndarray:
    """
    Loads 20 facial landmark points from a .pts file.
    The file format is:
    version: 1
    n_points: 20
    {
    x0 y0
    ...
    x19 y19
    }
    Args:
        filepath (str): The full path to the .pts file.
    Returns:
        np.ndarray: A NumPy array of shape (20, 2) containing the (x, y) coordinates.
    """
    points = []
    with open(filepath, 'r') as f:
        # Skip the first 3 header lines
        for _ in range(3):
            f.readline()
        # Read the 20 landmark points
        for _ in range(20):
            line = f.readline().strip()
            x, y = map(float, line.split())
            points.append([x, y])
    return np.array(points)

def visualize_image_with_landmarks(data_dir: str, sample_id: str) -> None:
    image_path = os.path.join(data_dir, f"BioID_{sample_id}.pgm")
    eye_path = os.path.join(data_dir, f"BioID_{sample_id}.eye")
    pts_path = os.path.join(data_dir, f"bioid_{sample_id}.pts")
    try:
        image = load_pgm_image(image_path)
        lx, ly, rx, ry = load_eye_positions(eye_path)
        landmarks = load_landmarks(pts_path)
        print(f"Image shape: {image.shape}")
        print(f"Eye positions: Left=({lx}, {ly}), Right=({rx}, {ry})")
        print(f"Landmarks shape: {landmarks.shape}")
        # --- Visualize the data ---
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(image, cmap='gray')
        # Plot eye positions from the .eye file (as large red crosses)
        ax.scatter([lx, rx], [ly, ry], color='red', marker='x', s=100, label='Eye Positions (.eye)')
        # Plot all 20 landmarks from the .pts file (as small cyan dots)
        ax.scatter(landmarks[:, 0], landmarks[:, 1], color='cyan', marker='.', s=50, label='Landmarks (.pts)')
        # Annotate each landmark with its index number for reference
        for i, (x, y) in enumerate(landmarks):
            ax.text(x + 2, y + 2, str(i), color='yellow', fontsize=8)
        ax.set_title(f"BioID Image {sample_id} with Annotations")
        ax.legend()
        plt.savefig(f"BioID_{sample_id}_landmarks.png")
    except FileNotFoundError as e:
        print(f"Error: A file was not found. Check your directory paths.\n{e}")

def load_full_dataset(data_dir: str):
    """
    Loads all images, eye positions, and landmarks from the BioID dataset.
    Args:
        data_dir (str): Path to the directory with .pgm, .eye, and .pts files.
    Returns:
        A tuple (X, y_eyes, y_landmarks):
        - X (np.ndarray): Shape (num_samples, height, width, 1).
        - y_eyes (np.ndarray): Shape (num_samples, 4).
        - y_landmarks (np.ndarray): Shape (num_samples, 20, 2).
    """
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pgm')])
    images = []
    eye_positions = []
    facial_landmarks = []
    print(f"Found {len(image_files)} images. Loading dataset...")
    for img_file in tqdm(image_files):
        base_id = img_file.split('_')[1].split('.')[0]
        image_path = os.path.join(data_dir, f"BioID_{base_id}.pgm")
        eye_path = os.path.join(data_dir, f"BioID_{base_id}.eye")
        pts_path = os.path.join(data_dir, f"bioid_{base_id}.pts")
        if os.path.exists(eye_path) and os.path.exists(pts_path):
            images.append(load_pgm_image(image_path))
            eye_positions.append(load_eye_positions(eye_path))
            facial_landmarks.append(load_landmarks(pts_path))
    # Convert lists to NumPy arrays
    X = np.array(images)[..., np.newaxis] 
    y_eyes = np.array(eye_positions)
    y_landmarks = np.array(facial_landmarks)
    return X, y_eyes, y_landmarks

class BioIDFaceDataset(Dataset):
    """Custom PyTorch Dataset for the BioID Face Database."""
    def __init__(self, images, eye_labels, landmark_labels, transform=None):
        """
        Args:
            images (np.ndarray): The image data (N, H, W, C).
            eye_labels (np.ndarray): The eye coordinates (N, 4).
            landmark_labels (np.ndarray): The landmark coordinates (N, 20, 2).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Convert numpy arrays to PyTorch tensors
        self.images = torch.from_numpy(images).float()
        self.eye_labels = torch.from_numpy(eye_labels).float()
        self.landmark_labels = torch.from_numpy(landmark_labels).float()
        # --- IMPORTANT: Reshape images for PyTorch ---
        # PyTorch CNNs expect images in (N, C, H, W) format.
        # Our current format is (N, H, W, C), so we permute the dimensions.
        self.images = self.images.permute(0, 3, 1, 2)
        # Normalize images to the [0, 1] range
        self.images = self.images / 255.0
        self.transform = transform
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)
    def __getitem__(self, idx):
        """
        Retrieves the image and its labels at the given index.
        Returns:
            A tuple (image, labels_dict)
        """
        image = self.images[idx]
        labels = {
            'eyes': self.eye_labels[idx],
            'landmarks': self.landmark_labels[idx]
        }
        if self.transform:
            image = self.transform(image)
        return image, labels
   

class EyeDetectorCNN(nn.Module):
    def __init__(self):
        super(EyeDetectorCNN, self).__init__()
        # Input shape: (Batch, 1, 286, 384) -> (Channels, Height, Width)
        # We will define our network as a sequence of layers
        self.feature_extractor = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 143x192
            # Convolutional Block 2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 71x96 
            # Convolutional Block 3
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 35x48
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            # The flattened size is 16 (channels) * 35 (height) * 48 (width)
            nn.Linear(in_features=16 * 35 * 48, out_features=512),
            nn.ReLU(),
            # The final layer outputs 4 values for the eye coordinates
            nn.Linear(in_features=512, out_features=4) 
        )
    def forward(self, x):
        # Pass input through the feature extractor
        features = self.feature_extractor(x)
        # Pass the extracted features to the regressor to get coordinates
        coordinates = self.regressor(features)
        return coordinates


class LandmarkDetectorCNN(nn.Module):
    def __init__(self):
        super(LandmarkDetectorCNN, self).__init__()
        # Input shape: (Batch, 1, 286, 384) -> (Channels, Height, Width)
        # We will define our network as a sequence of layers
        self.feature_extractor = nn.Sequential(
            # Convolutional Block 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 143x192
            # Convolutional Block 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 71x96
            # Convolutional Block 3
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 35x48
            # Convolutional Block 4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output size: 17x24
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            # The flattened size is 64 (channels) * 17 (height) * 24 (width)
            nn.Linear(in_features=64 * 17 * 24, out_features=512),
            nn.ReLU(),
            # The final layer outputs 40 values for the 20 landmarks (x,y) pairs
            nn.Linear(in_features=512, out_features=40) 
        )
    def forward(self, x):
        # Pass input through the feature extractor
        features = self.feature_extractor(x)
        # Pass the extracted features to the regressor to get coordinates
        coordinates = self.regressor(features)
        return coordinates


def visualize_predictions(dataset, model, num_samples=5):
    """
    Plots model predictions against ground truth for a few samples.
    """
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    # Get a few random samples from the validation set
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for i, idx in enumerate(indices):
        image, labels = dataset[idx]
        true_eyes = labels['eyes']
        # Prepare the image for the model
        image_tensor = image.unsqueeze(0).to(device) # Add a batch dimension
        # Get model prediction
        with torch.no_grad():
            predicted_eyes = model(image_tensor).squeeze(0).cpu().numpy()
        # Reshape image for plotting (C, H, W) -> (H, W)
        image_to_plot = image.squeeze(0).numpy()
        ax = axes[i]
        ax.imshow(image_to_plot, cmap='gray')
        # Plot ground truth (Green circles)
        ax.scatter([true_eyes[2], true_eyes[0]], [true_eyes[3], true_eyes[1]],
                   c='lime', marker='o', s=80, label='Ground Truth')
        # Plot prediction (Red crosses)
        ax.scatter([predicted_eyes[2], predicted_eyes[0]], [predicted_eyes[3], predicted_eyes[1]],
                   c='red', marker='x', s=80, label='Prediction')
        ax.set_title(f"Sample {idx}")
        ax.axis('off')
    plt.legend()
    plt.savefig("predictions_vs_ground_truth.png")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Load and visualize BioID dataset.")
    parser.add_argument('--data_dir', type=str, default="./BioID_FaceDatabase", help="Path to the BioID dataset directory.")
    parser.add_argument('--sample_id', type=str, default="0000", help="Sample ID to visualize.")
    parser.add_argument('--visualize', action='store_true', help="Whether to visualize a sample image with landmarks.")
    parser.add_argument('--train', choices=["eyes", "face"], default="eyes", help="Whether to train a model to predict eye positions or facial landmarks.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--val', type=float, default=0.2, help="Fraction of data to use for validation.")
    args = parser.parse_args()

    data_dir = args.data_dir
    sample_id = args.sample_id
    visualize = args.visualize
    train_mode = args.train
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    val_fraction = args.val

    # Visualize a sample image with landmarks
    if visualize:
        visualize_image_with_landmarks(data_dir, sample_id)

    # Create appropriate model based on training mode
    if train_mode == "eyes":
        model = EyeDetectorCNN()
        print("Initialized EyeDetectorCNN model.")
    elif train_mode == "face":
        model = LandmarkDetectorCNN()
        print("Initialized LandmarkDetectorCNN model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Move model to the appropriate device
    model.to(device)

    # Load the full dataset
    X_data, y_eyes_data, y_landmarks_data = load_full_dataset(data_dir)
    full_dataset = BioIDFaceDataset(images=X_data, eye_labels=y_eyes_data, landmark_labels=y_landmarks_data)

    # Split into training and validation sets
    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for images, labels in bar:
            images = images.to(device)
            if train_mode == "eyes":
                labels = labels['eyes'].to(device)
            elif train_mode == "face":
                labels = labels['landmarks'].view(labels['landmarks'].size(0), -1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            bar.set_postfix({'Batch Loss': loss.item()})
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                images = images.to(device)
                if train_mode == "eyes":
                    labels = labels['eyes'].to(device)
                elif train_mode == "face":
                    labels = labels['landmarks'].view(labels['landmarks'].size(0), -1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Validation Loss: {epoch_val_loss:.4f}")

    print("\nTraining complete!")

    # Plot training and validation loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig("training_validation_loss.png")

    if visualize:
        visualize_predictions(val_dataset, model, num_samples=5)
