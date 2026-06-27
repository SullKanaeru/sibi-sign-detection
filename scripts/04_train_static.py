"""
Script untuk melatih model klasifikasi gambar statis SIBI
menggunakan Deep Learning (TensorFlow/Keras).
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Konfigurasi
DATA_PATH = '../data/static_features.csv'
MODEL_NAME = '../models/model_static.keras'

def plot_history(history):
    """Fungsi untuk memplot loss dan akurasi selama pelatihan."""
    plt.figure(figsize=(10, 4))
    
    # Plot akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Akurasi Model')
    plt.ylabel('Akurasi')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Model')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('../logs/static_training_history.png')
    print("[INFO] Grafik pelatihan disimpan sebagai '../logs/static_training_history.png'")

def normalize_landmarks(features):
    """
    Normalisasi koordinat dan Feature Engineering (Pairwise Distances).
    1. Mengurangi semua koordinat dengan koordinat pergelangan tangan (wrist).
    2. Menghitung jarak Euclidean antar setiap pasang titik (210 jarak).
    """
    import math
    augmented_features = []
    
    for i in range(len(features)):
        row = features[i]
        wx, wy, wz = row[0], row[1], row[2]
        
        # 1. Normalisasi Koordinat (63 fitur)
        norm_coords = []
        points = [] # Simpan untuk hitung jarak
        for j in range(0, 63, 3):
            nx = row[j] - wx
            ny = row[j+1] - wy
            nz = row[j+2] - wz
            norm_coords.extend([nx, ny, nz])
            points.append((nx, ny, nz))
            
        # 2. Pairwise Distances (21 * 20 / 2 = 210 fitur)
        distances = []
        for p1 in range(21):
            for p2 in range(p1 + 1, 21):
                dx = points[p1][0] - points[p2][0]
                dy = points[p1][1] - points[p2][1]
                dz = points[p1][2] - points[p2][2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                distances.append(dist)
                
        augmented_features.append(norm_coords + distances)
        
    return np.array(augmented_features, dtype=np.float32)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset {DATA_PATH} tidak ditemukan. Harap jalankan extract_static.py terlebih dahulu.")
        return
        
    print("[INFO] Membaca dataset...")
    df = pd.DataFrame(pd.read_csv(DATA_PATH))
    
    # Pisahkan fitur dan label
    # Kolom 0: label (A, B, C...)
    # Kolom 1: hand_type (kanan, kiri)
    # Kolom 2-64: koordinat x, y, z
    y_raw = df['label'].values
    hand_types = df['hand_type'].values
    X_raw = df.iloc[:, 2:].values
    
    # Normalisasi (opsional namun sangat direkomendasikan)
    X = normalize_landmarks(X_raw)
    
    # Encode label menjadi angka (A=0, B=1, dst.)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    print(f"[INFO] Kelas yang ditemukan ({num_classes}): {encoder.classes_}")
    
    # Simpan mapping label agar nanti saat inferensi kita tahu 0 itu huruf apa
    np.save('../models/static_classes.npy', encoder.classes_)
    
    # Membagi data latih dan uji (80% train, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[INFO] Jumlah data latih: {len(X_train)}")
    print(f"[INFO] Jumlah data validasi: {len(X_test)}")
    
    # Membangun Model Jaringan Saraf Tiruan (Dense Neural Network)
    # Dengan 273 fitur (63 koordinat + 210 jarak), kita menggunakan layer yang lebih lebar
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3), # Mencegah overfitting
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax') # Output layer
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # Melatih Model
    print("[INFO] Memulai pelatihan...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # Evaluasi
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"\n[INFO] Akurasi Validasi Akhir: {val_acc*100:.2f}%")
    
    # Simpan Model
    model.save(MODEL_NAME)
    print(f"[INFO] Model berhasil disimpan ke '{MODEL_NAME}'")
    
    # Simpan grafik
    plot_history(history)

if __name__ == '__main__':
    main()
