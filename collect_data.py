import os
import cv2
import mediapipe as mp

# Setup MediaPipe Hands untuk mendeteksi letak tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_DIR = './dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Tentukan kelas/isyarat yang ingin dikumpulkan
# Anda dapat menambahkan huruf lain di dalam array ini
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
hands_categories = ['kanan', 'kiri']
dataset_size = 100 # Jumlah gambar yang akan direkam per kelas per tangan

cap = cv2.VideoCapture(0) # 0 biasanya adalah kamera utama / webcam bawaan

for class_name in classes:
    for hand_cat in hands_categories:
        class_dir = os.path.join(DATA_DIR, hand_cat, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        print(f"\n[INFO] Bersiap merekam isyarat '{class_name}' dengan TANGAN {hand_cat.upper()}")
        print("[INFO] Tekan 'q' pada jendela video saat posisi tangan sudah benar dan siap.")
        
        # Loop untuk menunggu pengguna siap
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1) # Mirror gambar agar intuitif
            
            # Tambahkan teks panduan di layar
            cv2.putText(frame, f'TANGAN {hand_cat.upper()} - Isyarat "{class_name}"', 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Tekan "q" untuk mulai merekam', 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            # Proses gambar menggunakan MediaPipe untuk menampilkan *landmarks*
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                        
            cv2.imshow('Kamera - Pengumpul Data SIBI', frame)
            # Tunggu sampai tombol 'q' ditekan
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        counter = 0
        print(f"[INFO] Mulai merekam TANGAN {hand_cat.upper()} untuk isyarat: {class_name}")
        
        # Loop untuk mulai merekam frame (sebanyak dataset_size)
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # --- SIMPAN GAMBAR MENTAH ---
            img_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            
            # --- VISUALISASI SAAT MEREKAM ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                        
            cv2.putText(frame, f'Merekam TANGAN {hand_cat.upper()} "{class_name}": {counter+1}/{dataset_size}', 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Kamera - Pengumpul Data SIBI', frame)
            cv2.waitKey(50) # Jeda (50ms) antar frame
            counter += 1

print("\n[INFO] Pengumpulan data selesai!")
cap.release()
cv2.destroyAllWindows()
