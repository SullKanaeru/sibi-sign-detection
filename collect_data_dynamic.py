"""
SIBI Dynamic Gesture Dataset Collector
Untuk mengumpulkan dataset huruf J dan Z (gerakan dinamis)
Menyimpan landmark sequence dari MediaPipe Hands

Struktur output:
  dataset/
    J/
      seq_0001.npy   # shape: (N_frames, 21, 3)
      seq_0002.npy
      ...
    Z/
      seq_0001.npy
      ...

Requirements:
  pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from pathlib import Path

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
LETTERS          = ["J", "Z"]       # huruf yang dikumpulkan
SAMPLES_PER_LETTER = 200            # target jumlah sample per huruf
FRAMES_PER_SAMPLE  = 30            # jumlah frame per satu rekaman (~1 detik di 30fps)
COUNTDOWN_SEC      = 3             # hitungan mundur sebelum rekam
DATASET_DIR        = "dataset"     # folder output
CAMERA_INDEX       = 0             # ganti jika kamera bukan /dev/video0

# Warna (BGR)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0,   200, 80)
RED    = (0,   60,  220)
YELLOW = (0,   200, 220)
GRAY   = (180, 180, 180)
DARK   = (40,  40,  40)

# ─────────────────────────────────────────
# SETUP MEDIAPIPE
# ─────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────
def extract_landmarks(results):
    """Ambil 21 landmark (x, y, z) dari hasil MediaPipe. Return None jika tidak terdeteksi."""
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21, 3)


def count_saved(letter):
    """Hitung file .npy yang sudah tersimpan untuk huruf tertentu."""
    folder = Path(DATASET_DIR) / letter
    if not folder.exists():
        return 0
    return len(list(folder.glob("seq_*.npy")))


def save_sequence(letter, sequence):
    """Simpan sequence landmark ke file .npy."""
    folder = Path(DATASET_DIR) / letter
    folder.mkdir(parents=True, exist_ok=True)
    idx = count_saved(letter) + 1
    filename = folder / f"seq_{idx:04d}.npy"
    np.save(str(filename), np.array(sequence, dtype=np.float32))
    return str(filename)


def overlay_text(frame, text, pos, size=0.8, color=WHITE, thickness=2, bg=None):
    """Tulis teks dengan optional background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, size, thickness)
    x, y = pos
    if bg is not None:
        pad = 8
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(frame, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


def draw_progress_bar(frame, current, total, x, y, w, h):
    """Gambar progress bar rekaman."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), DARK, -1)
    filled = int(w * (current / total))
    cv2.rectangle(frame, (x, y), (x + filled, y + h), GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)


def draw_sample_progress(frame, saved, target, x, y, w, h):
    """Gambar progress bar jumlah sample."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), DARK, -1)
    filled = int(w * min(saved / target, 1.0))
    color = GREEN if saved >= target else YELLOW
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)


def draw_landmark_on_frame(frame, results):
    """Gambar koneksi landmark tangan di frame."""
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )


# ─────────────────────────────────────────
# LAYAR: PANDUAN GERAKAN
# ─────────────────────────────────────────
GUIDES = {
    "J": [
        "Bentuk tangan seperti huruf 'I'",
        "(kelingking tegak, jempol keluar)",
        "",
        "Gerakkan kelingking membentuk",
        "lengkungan huruf J dari atas ke bawah",
        "lalu ke kiri (seperti menulis J)",
    ],
    "Z": [
        "Arahkan jari telunjuk ke depan",
        "",
        "Gerakkan telunjuk membentuk",
        "zigzag huruf Z:",
        "kanan -> kiri-bawah -> kanan",
    ],
}


def show_guide_screen(cap, letter):
    """Tampilkan layar panduan gerakan sebelum mulai rekam."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Overlay gelap
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Judul
        overlay_text(frame, f"Panduan Gerakan Huruf  {letter}",
                     (w // 2 - 220, 70), size=1.0, color=YELLOW, thickness=2)

        # Instruksi
        for i, line in enumerate(GUIDES[letter]):
            overlay_text(frame, line, (w // 2 - 200, 140 + i * 38),
                         size=0.65, color=WHITE)

        # Contoh visual sederhana (kotak placeholder)
        box_x, box_y, box_w, box_h = w // 2 - 80, 380, 160, 160
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), GRAY, 1)
        overlay_text(frame, f"[{letter}]", (box_x + 55, box_y + 95),
                     size=1.4, color=YELLOW, thickness=3)

        overlay_text(frame, "Tekan  SPASI  untuk mulai",
                     (w // 2 - 160, h - 60), size=0.75, color=GREEN, bg=DARK)
        overlay_text(frame, "Tekan  Q  untuk keluar",
                     (w // 2 - 120, h - 30), size=0.6, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True   # lanjut rekam
        if key == ord('q'):
            return False  # keluar


# ─────────────────────────────────────────
# LAYAR: COUNTDOWN
# ─────────────────────────────────────────
def show_countdown(cap, letter, results_holder):
    """Countdown sebelum rekaman dimulai. Return False jika user quit."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        results_holder[0] = results
        draw_landmark_on_frame(frame, results)

        elapsed = time.time() - start
        remaining = COUNTDOWN_SEC - elapsed
        if remaining <= 0:
            return True

        h, w = frame.shape[:2]
        overlay_text(frame, f"Bersiap...", (w // 2 - 80, h // 2 - 60),
                     size=0.9, color=YELLOW, bg=DARK)
        overlay_text(frame, str(int(remaining) + 1),
                     (w // 2 - 25, h // 2 + 30), size=2.5, color=GREEN, thickness=4)
        overlay_text(frame, f"Huruf: {letter}", (w // 2 - 55, h // 2 + 100),
                     size=0.8, color=WHITE)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False


# ─────────────────────────────────────────
# REKAM SATU SEQUENCE
# ─────────────────────────────────────────
def record_sequence(cap, letter, sample_idx, total_target):
    """
    Rekam satu sequence landmark.
    Return (sequence, status) — status: 'saved' | 'retry' | 'quit'
    """
    sequence = []
    frame_idx = 0

    while frame_idx < FRAMES_PER_SAMPLE:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        draw_landmark_on_frame(frame, results)

        lm = extract_landmarks(results)
        if lm is not None:
            sequence.append(lm)
            frame_idx += 1
            detected = True
        else:
            detected = False

        # ── UI ──
        # Status tangan
        hand_color = GREEN if detected else RED
        hand_text  = "Tangan terdeteksi" if detected else "Tangan tidak terdeteksi!"
        overlay_text(frame, hand_text, (12, 30), size=0.65, color=hand_color, bg=DARK)

        # Label huruf
        overlay_text(frame, letter, (w - 70, 55), size=1.8, color=YELLOW, thickness=3)

        # Progress rekaman
        draw_progress_bar(frame, frame_idx, FRAMES_PER_SAMPLE, 12, h - 70, w - 24, 18)
        overlay_text(frame, f"Merekam... {frame_idx}/{FRAMES_PER_SAMPLE} frame",
                     (12, h - 80), size=0.6, color=WHITE)

        # Progress sample
        draw_sample_progress(frame, sample_idx - 1, total_target, 12, h - 35, w - 24, 12)
        overlay_text(frame, f"Sample: {sample_idx}/{total_target}",
                     (12, h - 45), size=0.55, color=GRAY)

        overlay_text(frame, "R=ulangi  Q=keluar", (w - 200, h - 12),
                     size=0.5, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            return [], 'retry'
        if key == ord('q'):
            return [], 'quit'

    return sequence, 'done'


# ─────────────────────────────────────────
# LAYAR: KONFIRMASI SETELAH REKAM
# ─────────────────────────────────────────
def show_result_screen(cap, letter, saved_path, sample_idx, total_target):
    """Tampilkan konfirmasi setelah sequence berhasil disimpan."""
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        elapsed = time.time() - start
        remaining = max(0, 1.5 - elapsed)  # auto-lanjut setelah 1.5 detik

        overlay_text(frame, "[OK] Tersimpan!", (w // 2 - 110, h // 2 - 30),
                     size=1.1, color=GREEN, thickness=2)
        overlay_text(frame, f"Sample {sample_idx}/{total_target}  --  Huruf {letter}",
                     (w // 2 - 160, h // 2 + 20), size=0.65, color=WHITE)
        overlay_text(frame, f"Lanjut otomatis dalam {remaining:.1f}s ...",
                     (w // 2 - 150, h // 2 + 60), size=0.6, color=GRAY)
        overlay_text(frame, "SPASI=lanjut sekarang   R=ulangi   Q=keluar",
                     (w // 2 - 220, h - 30), size=0.55, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or remaining <= 0:
            return 'next'
        if key == ord('r'):
            return 'retry'
        if key == ord('q'):
            return 'quit'


# ─────────────────────────────────────────
# LAYAR: SELESAI PER HURUF
# ─────────────────────────────────────────
def show_letter_done(cap, letter, saved):
    """Tampilkan layar selesai setelah semua sample satu huruf terkumpul."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        overlay_text(frame, f"Huruf  {letter}  selesai!",
                     (w // 2 - 140, h // 2 - 40), size=1.1, color=GREEN, thickness=2)
        overlay_text(frame, f"{saved} sample berhasil dikumpulkan",
                     (w // 2 - 170, h // 2 + 10), size=0.7, color=WHITE)
        overlay_text(frame, "Tekan SPASI untuk lanjut ke huruf berikutnya",
                     (w // 2 - 210, h - 40), size=0.65, color=YELLOW)
        overlay_text(frame, "Tekan Q untuk keluar",
                     (w // 2 - 100, h - 15), size=0.55, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


# ─────────────────────────────────────────
# LAYAR: RINGKASAN AKHIR
# ─────────────────────────────────────────
def show_summary():
    """Print ringkasan dataset ke terminal."""
    print("\n" + "=" * 50)
    print("  RINGKASAN DATASET")
    print("=" * 50)
    total = 0
    for letter in LETTERS:
        n = count_saved(letter)
        total += n
        status = "Selesai" if n >= SAMPLES_PER_LETTER else f"Kurang {SAMPLES_PER_LETTER - n} sample"
        print(f"  {letter}  :  {n:3d} / {SAMPLES_PER_LETTER} sample   [{status}]")
    print("-" * 50)
    print(f"  Total  :  {total} sample")
    print(f"  Lokasi :  {os.path.abspath(DATASET_DIR)}/")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def main():
    print("\n" + "=" * 50)
    print("  SIBI Dynamic Dataset Collector")
    print(f"  Huruf: {', '.join(LETTERS)}")
    print(f"  Target: {SAMPLES_PER_LETTER} sample per huruf")
    print(f"  Frame per sample: {FRAMES_PER_SAMPLE}")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Kamera index {CAMERA_INDEX} tidak bisa dibuka.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("SIBI Dataset Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SIBI Dataset Collector", 800, 600)

    try:
        for letter in LETTERS:
            saved = count_saved(letter)
            print(f"\n[INFO] Huruf {letter}: sudah ada {saved} sample")

            if saved >= SAMPLES_PER_LETTER:
                print(f"[SKIP] Huruf {letter} sudah mencapai target.")
                continue

            # Tampilkan panduan
            if not show_guide_screen(cap, letter):
                print("[QUIT] Keluar dari program.")
                break

            results_holder = [None]

            while saved < SAMPLES_PER_LETTER:
                sample_idx = saved + 1

                # Countdown
                if not show_countdown(cap, letter, results_holder):
                    print("[QUIT] Keluar dari program.")
                    break

                # Rekam
                sequence, status = record_sequence(cap, letter, sample_idx, SAMPLES_PER_LETTER)

                if status == 'quit':
                    print("[QUIT] Keluar dari program.")
                    break

                if status == 'retry' or len(sequence) < FRAMES_PER_SAMPLE:
                    print(f"[RETRY] Sample {sample_idx} diulang.")
                    continue

                # Simpan
                path = save_sequence(letter, sequence)
                saved += 1
                print(f"[SAVED] {path}  ({saved}/{SAMPLES_PER_LETTER})")

                # Konfirmasi
                action = show_result_screen(cap, letter, path, saved, SAMPLES_PER_LETTER)
                if action == 'quit':
                    print("[QUIT] Keluar dari program.")
                    break
                if action == 'retry':
                    # Hapus yang baru saja disimpan dan ulangi
                    folder = Path(DATASET_DIR) / letter
                    files = sorted(folder.glob("seq_*.npy"))
                    if files:
                        files[-1].unlink()
                        saved -= 1
                    continue

            else:
                # Loop selesai normal
                if not show_letter_done(cap, letter, saved):
                    print("[QUIT] Keluar dari program.")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        show_summary()


if __name__ == "__main__":
    main()
