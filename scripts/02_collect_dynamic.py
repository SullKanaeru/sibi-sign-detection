"""
SIBI Dynamic Gesture Dataset Collector
Used for capturing sequence-based gestures (e.g., J and Z)
Saves MediaPipe Hands landmark sequences directly to disk.

Output structure:
  data/raw/
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
# CONFIGURATION
# ─────────────────────────────────────────
LETTERS          = ["J", "Z"]       # The dynamic gesture classes to collect
SAMPLES_PER_LETTER = 200            # Target number of sequence samples per gesture
FRAMES_PER_SAMPLE  = 30             # Number of frames per sequence (~1 second at 30fps)
COUNTDOWN_SEC      = 3              # Countdown duration before recording begins
DATASET_DIR        = "../data/raw"  # Output directory for the datasets
CAMERA_INDEX       = 0              # Adjust if using an external webcam

# Colors (BGR format for OpenCV)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREEN  = (0,   200, 80)
RED    = (0,   60,  220)
YELLOW = (0,   200, 220)
GRAY   = (180, 180, 180)
DARK   = (40,  40,  40)

# ─────────────────────────────────────────
# MEDIAPIPE INITIALIZATION
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
# UTILITY FUNCTIONS
# ─────────────────────────────────────────
def extract_landmarks(results):
    """Extracts the 21 3D landmarks from MediaPipe results. Returns None if no hand is detected."""
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # Shape: (21, 3)


def count_saved(letter):
    """Counts the number of successfully saved .npy sequences for a given gesture."""
    folder = Path(DATASET_DIR) / letter
    if not folder.exists():
        return 0
    return len(list(folder.glob("seq_*.npy")))


def save_sequence(letter, sequence):
    """Saves a landmark sequence to a .npy file with auto-incrementing filenames."""
    folder = Path(DATASET_DIR) / letter
    folder.mkdir(parents=True, exist_ok=True)
    idx = count_saved(letter) + 1
    filename = folder / f"seq_{idx:04d}.npy"
    np.save(str(filename), np.array(sequence, dtype=np.float32))
    return str(filename)


def overlay_text(frame, text, pos, size=0.8, color=WHITE, thickness=2, bg=None):
    """Draws text on the frame with an optional solid background box for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, size, thickness)
    x, y = pos
    if bg is not None:
        pad = 8
        cv2.rectangle(frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(frame, text, (x, y), font, size, color, thickness, cv2.LINE_AA)


def draw_progress_bar(frame, current, total, x, y, w, h):
    """Draws a progress bar indicating recording duration."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), DARK, -1)
    filled = int(w * (current / total))
    cv2.rectangle(frame, (x, y), (x + filled, y + h), GREEN, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)


def draw_sample_progress(frame, saved, target, x, y, w, h):
    """Draws a progress bar indicating overall collection quota progress."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), DARK, -1)
    filled = int(w * min(saved / target, 1.0))
    color = GREEN if saved >= target else YELLOW
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), GRAY, 1)


def draw_landmark_on_frame(frame, results):
    """Renders the hand landmark skeleton onto the given frame."""
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
# UI SCREENS: GESTURE GUIDES
# ─────────────────────────────────────────
GUIDES = {
    "J": [
        "Shape your hand like the letter 'I'",
        "(pinky extended, thumb out)",
        "",
        "Move your pinky to draw",
        "a 'J' shape in the air:",
        "downwards, then curving left",
    ],
    "Z": [
        "Point your index finger forward",
        "",
        "Use your index finger to trace",
        "a 'Z' shape in the air:",
        "right -> diagonally left-down -> right",
    ],
}


def show_guide_screen(cap, letter):
    """Displays instructional text for the gesture before recording begins."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Apply a dark semi-transparent overlay for readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        overlay_text(frame, f"Gesture Guide for Letter  {letter}",
                     (w // 2 - 250, 70), size=1.0, color=YELLOW, thickness=2)

        # Instructions
        for i, line in enumerate(GUIDES[letter]):
            overlay_text(frame, line, (w // 2 - 220, 140 + i * 38),
                         size=0.65, color=WHITE)

        # Simple placeholder graphic box
        box_x, box_y, box_w, box_h = w // 2 - 80, 380, 160, 160
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), GRAY, 1)
        overlay_text(frame, f"[{letter}]", (box_x + 55, box_y + 95),
                     size=1.4, color=YELLOW, thickness=3)

        # User controls
        overlay_text(frame, "Press  SPACE  to start",
                     (w // 2 - 160, h - 60), size=0.75, color=GREEN, bg=DARK)
        overlay_text(frame, "Press  Q  to quit",
                     (w // 2 - 120, h - 30), size=0.6, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True   # Proceed to record
        if key == ord('q'):
            return False  # Quit


# ─────────────────────────────────────────
# UI SCREENS: COUNTDOWN
# ─────────────────────────────────────────
def show_countdown(cap, letter, results_holder):
    """Displays a countdown timer before recording. Returns False if user quits."""
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
        overlay_text(frame, f"Get Ready...", (w // 2 - 90, h // 2 - 60),
                     size=0.9, color=YELLOW, bg=DARK)
        overlay_text(frame, str(int(remaining) + 1),
                     (w // 2 - 25, h // 2 + 30), size=2.5, color=GREEN, thickness=4)
        overlay_text(frame, f"Letter: {letter}", (w // 2 - 60, h // 2 + 100),
                     size=0.8, color=WHITE)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False


# ─────────────────────────────────────────
# CORE LOGIC: RECORD SINGLE SEQUENCE
# ─────────────────────────────────────────
def record_sequence(cap, letter, sample_idx, total_target):
    """
    Captures a single landmark sequence.
    Returns (sequence, status) where status is one of: 'saved', 'retry', 'quit'
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

        # ── UI Overlays ──
        # Hand detection status
        hand_color = GREEN if detected else RED
        hand_text  = "Hand Detected" if detected else "Hand NOT Detected!"
        overlay_text(frame, hand_text, (12, 30), size=0.65, color=hand_color, bg=DARK)

        # Target letter
        overlay_text(frame, letter, (w - 70, 55), size=1.8, color=YELLOW, thickness=3)

        # Sequence recording progress bar
        draw_progress_bar(frame, frame_idx, FRAMES_PER_SAMPLE, 12, h - 70, w - 24, 18)
        overlay_text(frame, f"Recording... {frame_idx}/{FRAMES_PER_SAMPLE} frames",
                     (12, h - 80), size=0.6, color=WHITE)

        # Overall collection progress bar
        draw_sample_progress(frame, sample_idx - 1, total_target, 12, h - 35, w - 24, 12)
        overlay_text(frame, f"Sample: {sample_idx}/{total_target}",
                     (12, h - 45), size=0.55, color=GRAY)

        overlay_text(frame, "R=Retry  Q=Quit", (w - 180, h - 12),
                     size=0.5, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            return [], 'retry'
        if key == ord('q'):
            return [], 'quit'

    return sequence, 'done'


# ─────────────────────────────────────────
# UI SCREENS: POST-RECORD CONFIRMATION
# ─────────────────────────────────────────
def show_result_screen(cap, letter, saved_path, sample_idx, total_target):
    """Displays a brief confirmation overlay after a sequence is saved."""
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
        remaining = max(0, 1.5 - elapsed)  # Auto-continue after 1.5 seconds

        overlay_text(frame, "[OK] Saved!", (w // 2 - 100, h // 2 - 30),
                     size=1.1, color=GREEN, thickness=2)
        overlay_text(frame, f"Sample {sample_idx}/{total_target}  --  Letter {letter}",
                     (w // 2 - 160, h // 2 + 20), size=0.65, color=WHITE)
        overlay_text(frame, f"Auto-continuing in {remaining:.1f}s ...",
                     (w // 2 - 160, h // 2 + 60), size=0.6, color=GRAY)
        overlay_text(frame, "SPACE=Continue Now   R=Retry   Q=Quit",
                     (w // 2 - 250, h - 30), size=0.55, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') or remaining <= 0:
            return 'next'
        if key == ord('r'):
            return 'retry'
        if key == ord('q'):
            return 'quit'


# ─────────────────────────────────────────
# UI SCREENS: GESTURE COMPLETE
# ─────────────────────────────────────────
def show_letter_done(cap, letter, saved):
    """Displays a completion screen when all target samples for a letter are collected."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        overlay_text(frame, f"Letter  {letter}  Complete!",
                     (w // 2 - 170, h // 2 - 40), size=1.1, color=GREEN, thickness=2)
        overlay_text(frame, f"{saved} samples successfully collected",
                     (w // 2 - 190, h // 2 + 10), size=0.7, color=WHITE)
        overlay_text(frame, "Press SPACE to move to the next letter",
                     (w // 2 - 240, h - 40), size=0.65, color=YELLOW)
        overlay_text(frame, "Press Q to quit",
                     (w // 2 - 100, h - 15), size=0.55, color=GRAY)

        cv2.imshow("SIBI Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return True
        if key == ord('q'):
            return False


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
def show_summary():
    """Prints a dataset collection summary to the terminal."""
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    total = 0
    for letter in LETTERS:
        n = count_saved(letter)
        total += n
        status = "Complete" if n >= SAMPLES_PER_LETTER else f"Missing {SAMPLES_PER_LETTER - n} samples"
        print(f"  {letter}  :  {n:3d} / {SAMPLES_PER_LETTER} samples   [{status}]")
    print("-" * 50)
    print(f"  Total  :  {total} samples")
    print(f"  Path   :  {os.path.abspath(DATASET_DIR)}/")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────
# MAIN EXECUTION LOOP
# ─────────────────────────────────────────
def main():
    print("\n" + "=" * 50)
    print("  SIBI Dynamic Dataset Collector")
    print(f"  Target Letters: {', '.join(LETTERS)}")
    print(f"  Target Quota: {SAMPLES_PER_LETTER} samples per letter")
    print(f"  Frames per Sample: {FRAMES_PER_SAMPLE}")
    print("=" * 50)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open camera with index {CAMERA_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("SIBI Dataset Collector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SIBI Dataset Collector", 800, 600)

    try:
        for letter in LETTERS:
            saved = count_saved(letter)
            print(f"\n[INFO] Letter {letter}: {saved} samples already exist")

            if saved >= SAMPLES_PER_LETTER:
                print(f"[SKIP] Letter {letter} has already met its target quota.")
                continue

            # Display gesture guide
            if not show_guide_screen(cap, letter):
                print("[QUIT] Program exited by user.")
                break

            results_holder = [None]

            while saved < SAMPLES_PER_LETTER:
                sample_idx = saved + 1

                # Display countdown
                if not show_countdown(cap, letter, results_holder):
                    print("[QUIT] Program exited by user.")
                    break

                # Execute recording loop
                sequence, status = record_sequence(cap, letter, sample_idx, SAMPLES_PER_LETTER)

                if status == 'quit':
                    print("[QUIT] Program exited by user.")
                    break

                if status == 'retry' or len(sequence) < FRAMES_PER_SAMPLE:
                    print(f"[RETRY] Discarding and retrying sample {sample_idx}.")
                    continue

                # Save successful recording
                path = save_sequence(letter, sequence)
                saved += 1
                print(f"[SAVED] {path}  ({saved}/{SAMPLES_PER_LETTER})")

                # Show save confirmation
                action = show_result_screen(cap, letter, path, saved, SAMPLES_PER_LETTER)
                if action == 'quit':
                    print("[QUIT] Program exited by user.")
                    break
                if action == 'retry':
                    # Delete the just-saved file to retry
                    folder = Path(DATASET_DIR) / letter
                    files = sorted(folder.glob("seq_*.npy"))
                    if files:
                        files[-1].unlink()
                        saved -= 1
                    continue

            else:
                # Quota met successfully
                if not show_letter_done(cap, letter, saved):
                    print("[QUIT] Program exited by user.")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        show_summary()


if __name__ == "__main__":
    main()
