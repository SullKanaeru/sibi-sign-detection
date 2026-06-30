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
    Coordinate Normalization and Feature Engineering.
    Extracts a rich feature set to maximize discriminability between visually similar gestures:

    1. Normalized Coordinates (63 features):
       Subtracts wrist position (landmark 0) so all coordinates are wrist-relative.

    2. Pairwise Euclidean Distances (210 features):
       Captures the overall shape and proportional geometry of the hand.

    3. Scale Invariance:
       Divides all features by the max absolute coordinate so the model is
       fully invariant to how close or far the hand is from the camera.

    4. Signed Finger Crossing Features (4 features):
       Computes the SIGNED (directional) difference in X and Z coordinates
       between the index fingertip (landmark 8) and middle fingertip (landmark 12).
       This directly encodes whether the fingers are CROSSED (R) or PARALLEL (U),
       which pure Euclidean distance cannot capture reliably.

    5. T vs X Discriminating Features (6 features):
       In SIBI, T tucks the thumb between index and middle fingers, while X hooks
       the index finger. These features capture:
       - Thumb tip position relative to index and middle finger bases
       - Index finger curl angle (hook detection for X)
       - Thumb-to-index-middle midpoint distance (tuck detection for T)
    """
    import math
    augmented_features = []
    
    # MediaPipe landmark indices
    IDX_THUMB_TIP  = 4   # Thumb tip
    IDX_THUMB_IP   = 3   # Thumb IP joint
    IDX_INDEX_MCP  = 5   # Index finger MCP (knuckle)
    IDX_INDEX_PIP  = 7   # Index finger PIP joint
    IDX_INDEX_DIP  = 6   # Index finger DIP joint
    IDX_INDEX_TIP  = 8   # Index finger tip
    IDX_MIDDLE_MCP = 9   # Middle finger MCP (knuckle)
    IDX_MIDDLE_PIP = 11  # Middle finger PIP joint
    IDX_MIDDLE_TIP = 12  # Middle finger tip
    
    for i in range(len(features)):
        row = features[i]
        # Wrist coordinates (first 3 elements - landmark 0)
        wx, wy, wz = row[0], row[1], row[2]
        
        # 1. Normalize Coordinates (63 features)
        norm_coords = []
        points = [] # Store wrist-relative (x, y, z) for each of 21 landmarks
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
        max_val = max([abs(x) for x in norm_coords])
        if max_val > 0:
            norm_coords = [x / max_val for x in norm_coords]
            distances = [d / max_val for d in distances]
            
        # 4. Signed Finger Crossing Features (4 features) - U vs R
        scale = max_val if max_val > 0 else 1.0
        cross_x_tip = (points[IDX_INDEX_TIP][0] - points[IDX_MIDDLE_TIP][0]) / scale
        cross_z_tip = (points[IDX_INDEX_TIP][2] - points[IDX_MIDDLE_TIP][2]) / scale
        cross_x_pip = (points[IDX_INDEX_PIP][0] - points[IDX_MIDDLE_PIP][0]) / scale
        cross_z_pip = (points[IDX_INDEX_PIP][2] - points[IDX_MIDDLE_PIP][2]) / scale
        crossing_features = [cross_x_tip, cross_z_tip, cross_x_pip, cross_z_pip]
        
        # 5. T vs X Discriminating Features (6 features)
        # Feature 5a: Thumb tip Y relative to index MCP Y (in T, thumb is tucked low)
        thumb_y_rel_index = (points[IDX_THUMB_TIP][1] - points[IDX_INDEX_MCP][1]) / scale
        
        # Feature 5b: Thumb tip Y relative to middle MCP Y
        thumb_y_rel_middle = (points[IDX_THUMB_TIP][1] - points[IDX_MIDDLE_MCP][1]) / scale
        
        # Feature 5c: Distance from thumb tip to the midpoint between index MCP and middle MCP
        # In T, thumb is tucked between these two fingers, so this distance is small
        mid_x = (points[IDX_INDEX_MCP][0] + points[IDX_MIDDLE_MCP][0]) / 2.0
        mid_y = (points[IDX_INDEX_MCP][1] + points[IDX_MIDDLE_MCP][1]) / 2.0
        mid_z = (points[IDX_INDEX_MCP][2] + points[IDX_MIDDLE_MCP][2]) / 2.0
        thumb_to_midpoint = math.sqrt(
            (points[IDX_THUMB_TIP][0] - mid_x)**2 +
            (points[IDX_THUMB_TIP][1] - mid_y)**2 +
            (points[IDX_THUMB_TIP][2] - mid_z)**2
        ) / scale
        
        # Feature 5d: Index finger curl angle (dot product between PIP->MCP and PIP->TIP vectors)
        # In X, the index finger is hooked (bent at DIP), giving a smaller angle
        # In T, the index finger is closed into a fist, fully curled
        v1 = (points[IDX_INDEX_MCP][0] - points[IDX_INDEX_PIP][0],
              points[IDX_INDEX_MCP][1] - points[IDX_INDEX_PIP][1],
              points[IDX_INDEX_MCP][2] - points[IDX_INDEX_PIP][2])
        v2 = (points[IDX_INDEX_TIP][0] - points[IDX_INDEX_PIP][0],
              points[IDX_INDEX_TIP][1] - points[IDX_INDEX_PIP][1],
              points[IDX_INDEX_TIP][2] - points[IDX_INDEX_PIP][2])
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2) + 1e-8
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2) + 1e-8
        index_curl_angle = dot / (mag1 * mag2)  # cosine of the angle, range [-1, 1]
        
        # Feature 5e: Thumb tip X relative to index MCP X (lateral thumb position)
        thumb_x_rel_index = (points[IDX_THUMB_TIP][0] - points[IDX_INDEX_MCP][0]) / scale
        
        # Feature 5f: Distance from thumb tip to index PIP
        # In T, thumb tip is very close to index PIP (tucked alongside)
        thumb_to_index_pip = math.sqrt(
            (points[IDX_THUMB_TIP][0] - points[IDX_INDEX_PIP][0])**2 +
            (points[IDX_THUMB_TIP][1] - points[IDX_INDEX_PIP][1])**2 +
            (points[IDX_THUMB_TIP][2] - points[IDX_INDEX_PIP][2])**2
        ) / scale
        
        tx_features = [thumb_y_rel_index, thumb_y_rel_middle, thumb_to_midpoint,
                       index_curl_angle, thumb_x_rel_index, thumb_to_index_pip]
                
        augmented_features.append(norm_coords + distances + crossing_features + tx_features)
        
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
    print(f"[INFO] Feature vector size: {X.shape[1]}")
    
    # Save the label encoding map for real-time inference
    class_path = os.path.join(BASE_DIR, 'models', 'static_classes.npy')
    np.save(class_path, encoder.classes_)
    
    # Split the dataset (80% training, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Validation samples: {len(X_test)}")
    
    # Compute class weights to handle imbalanced dataset
    # T only has 102 samples vs X with 200 samples, which biases the model toward X
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_arr = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights_arr))
    print(f"[INFO] Class weights: { {encoder.classes_[k]: f'{v:.2f}' for k, v in class_weights.items()} }")
    
    # Build the Dense Neural Network Model
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
    
    # Train the model with class weights to prevent bias toward overrepresented classes
    print("[INFO] Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        class_weight=class_weights
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
