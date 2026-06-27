"""
Training script for sequential/dynamic sign language gestures (e.g., J & Z).
Uses Long Short-Term Memory (LSTM) layers in TensorFlow/Keras to capture temporal dependencies.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration paths
DATA_DIR = '../data/raw' # Note: Assumes dynamic sequence files (.npy) are stored under their respective class folders
MODEL_NAME = '../models/model_dynamic.keras'

def plot_history(history):
    """Plots and saves the LSTM training accuracy and loss history."""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('LSTM Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('../logs/dynamic_training_history.png')
    print("[INFO] Training history graph saved as '../logs/dynamic_training_history.png'")

def normalize_sequence(seq):
    """
    Normalizes coordinates in a sequence relative to the wrist's position in each frame.
    Input sequence shape: (30 frames, 21 landmarks, 3 coordinates)
    Output shape: (30 frames, 63 flattened features) - suitable for LSTM input.
    """
    seq_normalized = np.zeros((seq.shape[0], 63))
    for frame_idx in range(seq.shape[0]):
        frame_landmarks = seq[frame_idx] # Shape: (21, 3)
        # The wrist is represented by the 0-th landmark index
        wx, wy, wz = frame_landmarks[0]
        
        flat_landmarks = []
        for i in range(21):
            lx = frame_landmarks[i, 0] - wx
            ly = frame_landmarks[i, 1] - wy
            lz = frame_landmarks[i, 2] - wz
            flat_landmarks.extend([lx, ly, lz])
            
        seq_normalized[frame_idx] = np.array(flat_landmarks)
        
    return seq_normalized

def main():
    print(f"[INFO] Searching for dynamic sequence datasets in '{DATA_DIR}'...")
    base_path = Path(DATA_DIR)
    
    if not base_path.exists():
        print(f"[ERROR] Directory {DATA_DIR} not found!")
        return

    X = []
    y_raw = []
    
    # Auto-detect which letters possess dynamic sequence files (.npy)
    letters = []
    for d in base_path.iterdir():
        if d.is_dir() and len(list(d.glob("seq_*.npy"))) > 0:
            letters.append(d.name)
            
    print(f"[INFO] Dynamic gesture classes detected: {letters}")
    
    if len(letters) == 0:
        print("[ERROR] No dynamic sequence .npy files found!")
        return

    # Load all sequence data into memory
    for letter in letters:
        npy_files = list((base_path / letter).glob("seq_*.npy"))
        for npy_file in npy_files:
            sequence = np.load(npy_file) # Expected shape: (30, 21, 3)
            # Normalize and flatten -> (30, 63)
            sequence_norm = normalize_sequence(sequence)
            
            X.append(sequence_norm)
            y_raw.append(letter)

    X = np.array(X)
    print(f"[INFO] Input Dataset Shape (Samples, TimeSteps, Features): {X.shape}")
    
    # Encode class string labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    # Save the label encoding map for real-time inference mapping
    np.save('../models/dynamic_classes.npy', encoder.classes_)

    # Split the dataset (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build the LSTM Neural Network
    model = tf.keras.models.Sequential([
        # LSTM input shape requirement: (TimeSteps=30, Features=63)
        tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(128, return_sequences=False, activation='tanh'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        # Uses softmax for potential multi-class scalability, rather than single-unit sigmoid
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # Train the model
    # LSTM networks typically converge quickly on small datasets; 50-70 epochs is standard.
    print("[INFO] Initiating LSTM model training...")
    history = model.fit(
        X_train, y_train,
        epochs=70,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # Evaluate model performance on validation set
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"\n[INFO] Final LSTM Validation Accuracy: {val_acc*100:.2f}%")
    
    # Save the trained model
    model.save(MODEL_NAME)
    print(f"[INFO] Dynamic model successfully saved to '{MODEL_NAME}'")
    
    # Generate visualization graph
    plot_history(history)

if __name__ == '__main__':
    main()
