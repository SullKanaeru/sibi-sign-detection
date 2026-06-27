import os
# Matatkan pesan warning TensorFlow agar terminal lebih rapi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("[INFO] Memuat model statis dan dinamis...")
# Memuat model
try:
    model_static = tf.keras.models.load_model('models/model_static.keras')
    model_dynamic = tf.keras.models.load_model('models/model_dynamic.keras')
    
    # Memuat label huruf
    static_classes = np.load('models/static_classes.npy', allow_pickle=True)
    dynamic_classes = np.load('models/dynamic_classes.npy', allow_pickle=True)
    print(f"[INFO] Kelas Statis : {static_classes}")
    print(f"[INFO] Kelas Dinamis: {dynamic_classes}")
except Exception as e:
    print(f"[ERROR] Gagal memuat model atau label: {e}")
    print("[ERROR] Pastikan train_static.py dan train_dynamic.py sudah berhasil dijalankan.")
    exit(1)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Variabel State untuk Deteksi Dinamis
is_recording_dynamic = False
dynamic_frames = []
MAX_DYNAMIC_FRAMES = 30
dynamic_result_text = ""
dynamic_display_timer = 0 # Berapa frame tulisan hasil dinamis akan bertahan di layar

def extract_and_normalize_landmarks(hand_landmarks):
    """
    Ekstrak 21 titik, kurangi dengan koordinat pergelangan tangan (wrist).
    Mengembalikan 2 array:
    1. norm_coords (63 elemen) untuk model dinamis.
    2. augmented_features (273 elemen: 63 koordinat + 210 jarak) untuk model statis.
    """
    import math
    
    # Ambil koordinat wrist (titik 0)
    wx = hand_landmarks.landmark[0].x
    wy = hand_landmarks.landmark[0].y
    wz = hand_landmarks.landmark[0].z
    
    norm_coords = []
    points = []
    for lm in hand_landmarks.landmark:
        nx = lm.x - wx
        ny = lm.y - wy
        nz = lm.z - wz
        norm_coords.extend([nx, ny, nz])
        points.append((nx, ny, nz))
        
    distances = []
    for p1 in range(21):
        for p2 in range(p1 + 1, 21):
            dx = points[p1][0] - points[p2][0]
            dy = points[p1][1] - points[p2][1]
            dz = points[p1][2] - points[p2][2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            distances.append(dist)
            
    augmented_features = norm_coords + distances
    
    return np.array(norm_coords), np.array(augmented_features)

def main():
    global is_recording_dynamic, dynamic_frames, dynamic_result_text, dynamic_display_timer
    
    cap = cv2.VideoCapture(0)
    print("\n[INFO] Kamera telah aktif!")
    print("[INFO] Tekan 'd' untuk mulai merekam huruf dinamis (J atau Z).")
    print("[INFO] Tekan 'q' untuk keluar.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip layar agar seperti cermin
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Konversi ke RGB untuk MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        current_text = "Tidak ada isyarat"
        box_color = (0, 0, 0)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Gambar titik-titik di layar
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Ekstrak fitur
            features_63, features_273 = extract_and_normalize_landmarks(hand_landmarks)
            
            # ==========================================
            # LOGIKA DINAMIS (Perekaman J & Z)
            # ==========================================
            if is_recording_dynamic:
                dynamic_frames.append(features_63)
                progress = len(dynamic_frames)
                
                # Tampilkan status perekaman
                cv2.putText(frame, f"MEREKAM DINAMIS: {progress}/{MAX_DYNAMIC_FRAMES}", (w//2 - 200, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Jika sudah mencapai 30 frame, hentikan dan prediksi
                if progress == MAX_DYNAMIC_FRAMES:
                    # Model membutuhkan bentuk (1, 30, 63) -> (Batch, TimeSteps, Features)
                    seq_input = np.expand_dims(np.array(dynamic_frames), axis=0)
                    
                    preds = model_dynamic.predict(seq_input, verbose=0)[0]
                    predicted_idx = np.argmax(preds)
                    confidence = preds[predicted_idx]
                    
                    dynamic_result_text = f"Dinamis: {dynamic_classes[predicted_idx]} ({confidence*100:.1f}%)"
                    
                    # Reset state
                    is_recording_dynamic = False
                    dynamic_frames = []
                    # Tampilkan hasil dinamis selama 90 frame (~3 detik)
                    dynamic_display_timer = 90
            else:
                # ==========================================
                # LOGIKA STATIS (Setiap saat jika tidak merekam)
                # ==========================================
                # Jika masih menampilkan hasil dinamis sebelumnya, jangan tebak statis dulu
                if dynamic_display_timer > 0:
                    current_text = dynamic_result_text
                    box_color = (0, 255, 255) # Kuning untuk hasil dinamis
                    dynamic_display_timer -= 1
                else:
                    # Model statis butuh (1, 273)
                    input_static = np.expand_dims(features_273, axis=0)
                    preds = model_static.predict(input_static, verbose=0)[0]
                    predicted_idx = np.argmax(preds)
                    confidence = preds[predicted_idx]
                    
                    # Tampilkan prediksi jika tingkat keyakinan > 60%
                    if confidence > 0.60:
                        current_text = f"{static_classes[predicted_idx]} ({confidence*100:.1f}%)"
                        box_color = (0, 255, 0) # Hijau untuk statis
                    else:
                        current_text = "Kurang yakin"
                        
        else:
            # Jika tidak ada tangan, pastikan timer dinamis tetap berjalan
            if dynamic_display_timer > 0:
                current_text = dynamic_result_text
                box_color = (0, 255, 255)
                dynamic_display_timer -= 1
        
        # Desain UI Sederhana (Background Hitam untuk teks)
        cv2.rectangle(frame, (0, h-60), (w, h), box_color, -1)
        cv2.putText(frame, current_text, (20, h-20), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Panduan shortcut
        cv2.putText(frame, "[D]: Rekam J/Z   [Q]: Keluar", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        cv2.imshow('Aplikasi Penerjemah SIBI Real-time', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') and not is_recording_dynamic:
            is_recording_dynamic = True
            dynamic_frames = [] # Kosongkan buffer
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
