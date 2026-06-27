"""
Script untuk mengekstrak fitur landmark 3D (x, y, z) dari dataset gambar statis
dan menyimpannya ke dalam format CSV untuk pelatihan model.
"""

import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,       # Karena kita memproses gambar statis satu per satu
    max_num_hands=1,
    min_detection_confidence=0.5
)

DATA_DIR = '../data/raw'
OUTPUT_CSV = '../data/static_features.csv'

def extract_landmarks(image_path):
    """
    Membaca gambar, mendeteksi tangan, dan mengembalikan array 1D
    berisi 63 nilai (21 titik * 3 sumbu).
    Kembalikan None jika tangan tidak terdeteksi.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        # Ambil tangan pertama yang terdeteksi
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Ekstrak koordinat (x, y, z)
        # Akan menghasilkan list 63 elemen: [x1, y1, z1, x2, y2, z2, ...]
        row = []
        for landmark in hand_landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
            
        return row
    return None

def main():
    print(f"\n[INFO] Memulai ekstraksi fitur dari direktori: {DATA_DIR} ...")
    
    dataset_rows = []
    
    base_path = Path(DATA_DIR)
    
    if not base_path.exists():
        print(f"[ERROR] Direktori {DATA_DIR} tidak ditemukan!")
        return

    # Loop setiap orientasi tangan (kanan/kiri)
    # Kita menggunakan glob untuk menemukan semua .jpg
    all_images = list(base_path.rglob("*.jpg"))
    total_images = len(all_images)
    
    if total_images == 0:
        print(f"[ERROR] Tidak ada gambar .jpg ditemukan di dalam {DATA_DIR}.")
        return
        
    print(f"[INFO] Ditemukan {total_images} gambar. Mulai memproses...")
    
    processed = 0
    failed = 0
    
    for img_path in all_images:
        # Ambil nama kelas/huruf dari nama foldernya
        # Contoh path: dataset/kanan/A/0.jpg
        # img_path.parent.name -> "A"
        class_name = img_path.parent.name
        hand_category = img_path.parent.parent.name
        
        landmarks = extract_landmarks(str(img_path))
        
        if landmarks is not None:
            # Tambahkan label huruf dan kategori tangan di depan
            row = [class_name, hand_category] + landmarks
            dataset_rows.append(row)
            processed += 1
        else:
            failed += 1
            
        if (processed + failed) % 100 == 0:
            print(f"       Progres: {processed + failed} / {total_images} diproses...")

    # Buat nama kolom untuk CSV
    columns = ['label', 'hand_type']
    for i in range(21):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

    df = pd.DataFrame(dataset_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("=" * 50)
    print("[INFO] Ekstraksi selesai!")
    print(f"       Berhasil diekstrak : {processed} gambar")
    print(f"       Gagal terdeteksi   : {failed} gambar")
    print(f"       Disimpan ke        : {OUTPUT_CSV}")
    print("=" * 50)

if __name__ == '__main__':
    main()
