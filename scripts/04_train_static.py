"""
Training script for the static hand gesture classification model.
Uses a Deep Neural Network (TensorFlow/Keras) with custom feature engineering
(pairwise distances) to improve accuracy on visually similar signs.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Configuration paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'static_features.csv')
MODEL_NAME = os.path.join(BASE_DIR, 'models', 'model_static.keras')

def plot_history(history):
    """Plots and saves the training accuracy and loss history."""
    plt.figure(figsize=(10, 4))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    log_path = os.path.join(BASE_DIR, 'logs', 'static_training_history.png')
    plt.savefig(log_path)
    print(f"[INFO] Training history graph saved as '{log_path}'")

def normalize_landmarks(features):
    """
    Coordinate Normalization and Feature Engineering (Pairwise Distances).
    1. Subtracts the wrist coordinates (origin) from all other hand landmarks.
    2. Computes the Euclidean distance between every pair of landmarks (210 pairwise distances).
    This significantly boosts classification accuracy for highly similar gestures (e.g., U vs R).
    """
    import math
    augmented_features = []
    
    for i in range(len(features)):
        row = features[i]
        # Wrist coordinates (first 3 elements)
        wx, wy, wz = row[0], row[1], row[2]
        
        # 1. Normalize Coordinates (63 features)
        norm_coords = []
        points = [] # Store points for distance calculation
        for j in range(0, 63, 3):
            nx = row[j] - wx
            ny = row[j+1] - wy
            nz = row[j+2] - wz
            norm_coords.extend([nx, ny, nz])
            points.append((nx, ny, nz))
            
        # 2. Pairwise Distances (21 * 20 / 2 = 210 features)
        distances = []
        for p1 in range(21):
            for p2 in range(p1 + 1, 21):
                dx = points[p1][0] - points[p2][0]
                dy = points[p1][1] - points[p2][1]
                dz = points[p1][2] - points[p2][2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                distances.append(dist)
                
        # 3. Scale Invariance
        # Divide all coordinates and distances by the maximum absolute coordinate value in this frame
        # This ensures the features are invariant to the distance of the hand from the camera
        max_val = max([abs(x) for x in norm_coords])
        if max_val > 0:
            norm_coords = [x / max_val for x in norm_coords]
            distances = [d / max_val for d in distances]
                
        augmented_features.append(norm_coords + distances)
        
    return np.array(augmented_features, dtype=np.float32)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset {DATA_PATH} not found. Please run 03_extract_features.py first.")
        return
        
    print("[INFO] Loading dataset...")
    df = pd.DataFrame(pd.read_csv(DATA_PATH))
    
    # Separate features and labels
    # Column 0: label (A, B, C...)
    # Column 1: hand_type (Right, Left)
    # Column 2-64: x, y, z coordinates
    y_raw = df['label'].values
    hand_types = df['hand_type'].values
    X_raw = df.iloc[:, 2:].values
    
    # Apply normalization and pairwise feature engineering
    X = normalize_landmarks(X_raw)
    
    # Encode categorical labels to integers (A=0, B=1, etc.)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    print(f"[INFO] Found {num_classes} classes: {encoder.classes_}")
    
    # Save the label encoding map for real-time inference
    class_path = os.path.join(BASE_DIR, 'models', 'static_classes.npy')
    np.save(class_path, encoder.classes_)
    
    # Split the dataset (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_test)}")
    
    # Build the Dense Neural Network Model
    # Since we use 273 features (63 coordinates + 210 distances), wider layers are preferred.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3), # Prevent overfitting
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(num_classes, activation='softmax') # Output layer
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # EarlyStopping prevents overfitting by halting training when validation loss stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Train the model
    print("[INFO] Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    
    # Evaluate performance on validation set
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"\n[INFO] Final Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save the trained model
    model.save(MODEL_NAME)
    print(f"[INFO] Model successfully saved to '{MODEL_NAME}'")
    
    # Save visualization graph
    plot_history(history)

if __name__ == '__main__':
    main()
