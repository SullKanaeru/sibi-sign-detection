"""
Script untuk melatih model klasifikasi sekuensial (dinamis) SIBI (J & Z)
menggunakan Long Short-Term Memory (LSTM) dengan TensorFlow/Keras.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pathlib import Path

# Konfigurasi
DATA_DIR = '../data/raw' # Asumsi: dataset dinamis J dan Z ada di dataset/J/ dan dataset/Z/
MODEL_NAME = '../models/model_dynamic.keras'

def plot_history(history):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Akurasi Model LSTM')
    plt.ylabel('Akurasi')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Model LSTM')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('../logs/dynamic_training_history.png')
    print("[INFO] Grafik pelatihan disimpan sebagai '../logs/dynamic_training_history.png'")

def normalize_sequence(seq):
    """
    Sama seperti pada statis, kita bisa mengurangi koordinat setiap frame
    dengan koordinat pergelangan tangannya (wrist = index 0)
    Seq shape awal: (30, 21, 3)
    Output shape: (30, 63) - Flattend untuk LSTM
    """
    seq_normalized = np.zeros((seq.shape[0], 63))
    for frame_idx in range(seq.shape[0]):
        frame_landmarks = seq[frame_idx] # (21, 3)
        # Wrist adalah landmark pertama
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
    print(f"[INFO] Mencari dataset dinamis di '{DATA_DIR}'...")
    base_path = Path(DATA_DIR)
    
    if not base_path.exists():
        print(f"[ERROR] Direktori {DATA_DIR} tidak ditemukan!")
        return

    X = []
    y_raw = []
    
    # Ambil huruf yang memiliki file seq_*.npy
    letters = []
    for d in base_path.iterdir():
        if d.is_dir() and len(list(d.glob("seq_*.npy"))) > 0:
            letters.append(d.name)
            
    print(f"[INFO] Huruf dinamis yang ditemukan: {letters}")
    
    if len(letters) == 0:
        print("[ERROR] Tidak ada data sequence .npy yang ditemukan!")
        return

    # Memuat semua data
    for letter in letters:
        npy_files = list((base_path / letter).glob("seq_*.npy"))
        for npy_file in npy_files:
            sequence = np.load(npy_file) # shape: (30, 21, 3)
            # Normalisasi & Flatten -> (30, 63)
            sequence_norm = normalize_sequence(sequence)
            
            X.append(sequence_norm)
            y_raw.append(letter)

    X = np.array(X)
    print(f"[INFO] Shape Dataset Input (Samples, TimeSteps, Features): {X.shape}")
    
    # Encode label
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)
    
    # Simpan mapping label agar nanti saat inferensi kita tahu 0 itu huruf apa
    np.save('../models/dynamic_classes.npy', encoder.classes_)

    # Membagi data latih dan uji (80% train, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Membangun Model LSTM
    model = tf.keras.models.Sequential([
        # Layer LSTM menerima input shape: (TimeSteps=30, Features=63)
        tf.keras.layers.LSTM(64, return_sequences=True, activation='tanh', input_shape=(X.shape[1], X.shape[2])),
        tf.keras.layers.LSTM(128, return_sequences=False, activation='tanh'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        # Jika cuma 2 kelas, biasanya sigmoid dengan unit 1, tapi kita gunakan softmax agar bisa ditambah kelas dinamis lain nanti
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # Melatih Model
    # LSTM seringkali konvergen lebih cepat dengan dataset kecil, 50-70 epoch sudah cukup
    print("[INFO] Memulai pelatihan LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=70,
        batch_size=32,
        validation_data=(X_test, y_test)
    )
    
    # Evaluasi
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(f"\n[INFO] Akurasi Validasi Akhir LSTM: {val_acc*100:.2f}%")
    
    # Simpan Model
    model.save(MODEL_NAME)
    print(f"[INFO] Model dinamis berhasil disimpan ke '{MODEL_NAME}'")
    
    # Simpan grafik
    plot_history(history)

if __name__ == '__main__':
    main()
