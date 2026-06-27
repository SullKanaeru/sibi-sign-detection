"""
Feature extraction script for static hand gestures.
This script extracts 3D hand landmarks (x, y, z) from raw static image datasets
and compiles them into a structured CSV format for model training.
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # Set to True since we are processing static, unconnected images
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATA_DIR = '../data/raw'
OUTPUT_CSV = '../data/static_features.csv'

def extract_landmarks(image_path):
    """
    Reads an image, detects hand landmarks, and flattens them into a 1D array.
    The resulting array contains 63 values (21 landmarks * 3 coordinates).
    Returns None if no hand is detected in the image.
    
    Args:
        image_path (str): The absolute or relative path to the image file.
    Returns:
        list: A flattened list of 63 float values representing (x, y, z) for each landmark.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # We assume one hand per image based on max_num_hands=1
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Flatten the (x, y, z) coordinates into a single list
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
            
        return row
    return None

def main():
    print(f"\n[INFO] Starting feature extraction from directory: {DATA_DIR} ...")
    
    dataset_rows = []
    base_path = Path(DATA_DIR)
    
    if not base_path.exists():
        print(f"[ERROR] Directory {DATA_DIR} not found!")
        return

    # Recursively find all .jpg files in the raw dataset directory
    all_images = list(base_path.rglob("*.jpg"))
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"[ERROR] No .jpg images found in {DATA_DIR}.")
        return
        
    print(f"[INFO] Found {total_images} images. Beginning extraction process...")
    
    processed = 0
    failed = 0
    
    for img_path in all_images:
        # Deduce class label and hand category from the directory structure
        # Expected path format: data/raw/<hand_category>/<class_name>/<id>.jpg
        class_name = img_path.parent.name
        hand_category = img_path.parent.parent.name
        
        landmarks = extract_landmarks(str(img_path))
        
        if landmarks is not None:
            # Prepend the class label and hand category to the feature vector
            row = [class_name, hand_category] + landmarks
            dataset_rows.append(row)
            processed += 1
        else:
            failed += 1
            
        # Logging progress every 100 images to prevent terminal flood
        if (processed + failed) % 100 == 0:
            print(f"       Progress: {processed + failed} / {total_images} processed...")

    # Generate feature column names (x_0, y_0, z_0, ..., x_20, y_20, z_20)
    columns = ['label', 'hand_type']
    for i in range(21):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    # Compile the dataset into a Pandas DataFrame and save it as a CSV file
    df = pd.DataFrame(dataset_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 50)
    print("[INFO] Feature extraction complete!")
    print(f"       Successfully extracted : {processed} images")
    print(f"       Failed to detect hands : {failed} images")
    print(f"       Output saved to        : {OUTPUT_CSV}")
    print("=" * 50)

if __name__ == '__main__':
    main()
