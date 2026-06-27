import os
# Suppress TensorFlow logging to keep the terminal output clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

print("[INFO] Loading static and dynamic models...")
# Load the pre-trained Keras models
try:
    model_static = tf.keras.models.load_model('models/model_static.keras')
    model_dynamic = tf.keras.models.load_model('models/model_dynamic.keras')
    
    # Load the label encoders to map numeric predictions back to string labels
    static_classes = np.load('models/static_classes.npy', allow_pickle=True)
    dynamic_classes = np.load('models/dynamic_classes.npy', allow_pickle=True)
    print(f"[INFO] Static Classes : {static_classes}")
    print(f"[INFO] Dynamic Classes: {dynamic_classes}")
except Exception as e:
    print(f"[ERROR] Failed to load models or labels: {e}")
    print("[ERROR] Ensure that 04_train_static.py and 05_train_dynamic.py have been executed successfully.")
    exit(1)

# Initialize MediaPipe Hands for real-time tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# State Variables for Dynamic Gesture Recording
is_recording_dynamic = False
dynamic_frames = []
MAX_DYNAMIC_FRAMES = 60
dynamic_result_text = ""
dynamic_display_timer = 0 # Frame countdown for how long the dynamic prediction result remains on screen

def extract_and_normalize_landmarks(hand_landmarks):
    """
    Extracts 21 3D landmarks and applies the same feature engineering pipeline
    used during training to ensure inference is fully consistent.
    
    Feature set (277 features total for the static model):
    1. norm_coords       (63 features): Wrist-relative normalized coordinates for LSTM.
    2. pairwise distances(210 features): Euclidean distances between all landmark pairs.
    3. Scale Invariance              : All features divided by max absolute coordinate.
    4. Signed crossing features (4 features): Signed X/Z differences between index and
       middle fingertips/PIP joints to directly discriminate 'U' (parallel) vs 'R' (crossed).
    
    Returns:
        norm_coords_arr    (np.ndarray, 63): Features for the dynamic LSTM model.
        augmented_features (np.ndarray, 277): Features for the static Dense model.
    """
    import math
    
    # MediaPipe landmark indices for finger crossing detection
    IDX_INDEX_TIP  = 8   # Index finger tip
    IDX_MIDDLE_TIP = 12  # Middle finger tip
    IDX_INDEX_PIP  = 7   # Index finger PIP (proximal interphalangeal) joint
    IDX_MIDDLE_PIP = 11  # Middle finger PIP joint
    
    # Extract the wrist coordinates (landmark index 0)
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
            
    # Scale Invariance: Divide by the maximum absolute coordinate value in this frame
    max_val = max([abs(x) for x in norm_coords])
    if max_val > 0:
        norm_coords = [x / max_val for x in norm_coords]
        distances = [d / max_val for d in distances]
    
    # Signed Finger Crossing Features (4 features)
    # Positive cross_x_tip means index tip is to the RIGHT of middle tip
    # In 'R', fingers are crossed so the sign of these values will differ from 'U'
    scale = max_val if max_val > 0 else 1.0
    cross_x_tip = (points[IDX_INDEX_TIP][0] - points[IDX_MIDDLE_TIP][0]) / scale
    cross_z_tip = (points[IDX_INDEX_TIP][2] - points[IDX_MIDDLE_TIP][2]) / scale
    cross_x_pip = (points[IDX_INDEX_PIP][0] - points[IDX_MIDDLE_PIP][0]) / scale
    cross_z_pip = (points[IDX_INDEX_PIP][2] - points[IDX_MIDDLE_PIP][2]) / scale
    crossing_features = [cross_x_tip, cross_z_tip, cross_x_pip, cross_z_pip]
        
    augmented_features = norm_coords + distances + crossing_features
    
    return np.array(norm_coords), np.array(augmented_features)


def main():
    global is_recording_dynamic, dynamic_frames, dynamic_result_text, dynamic_display_timer
    
    cap = cv2.VideoCapture(0)
    print("\n[INFO] Camera initialized successfully!")
    print("[INFO] Press 'd' to start recording a dynamic gesture (J or Z).")
    print("[INFO] Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for an intuitive mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert BGR to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        current_text = "No gesture detected"
        box_color = (0, 0, 0)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Overlay the hand skeleton on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract both sets of features for the respective models
            features_63, features_273 = extract_and_normalize_landmarks(hand_landmarks)
            
            # ==========================================
            # DYNAMIC LOGIC (Recording J & Z)
            # ==========================================
            if is_recording_dynamic:
                dynamic_frames.append(features_63)
                progress = len(dynamic_frames)
                
                # Display recording status
                cv2.putText(frame, f"RECORDING DYNAMIC: {progress}/{MAX_DYNAMIC_FRAMES}", (w//2 - 200, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                # Upon reaching the target frame count, halt recording and execute inference
                if progress == MAX_DYNAMIC_FRAMES:
                    # Model expects shape (1, 30, 63) -> (Batch, TimeSteps, Features)
                    seq_input = np.expand_dims(np.array(dynamic_frames), axis=0)
                    
                    preds = model_dynamic.predict(seq_input, verbose=0)[0]
                    predicted_idx = np.argmax(preds)
                    confidence = preds[predicted_idx]
                    
                    dynamic_result_text = f"Dynamic: {dynamic_classes[predicted_idx]} ({confidence*100:.1f}%)"
                    
                    # Reset the recording state variables
                    is_recording_dynamic = False
                    dynamic_frames = []
                    # Keep the prediction result visible for 90 frames (~3 seconds)
                    dynamic_display_timer = 90
            else:
                # ==========================================
                # STATIC LOGIC (Continuous inference)
                # ==========================================
                # Suppress static predictions if a dynamic result is currently being displayed
                if dynamic_display_timer > 0:
                    current_text = dynamic_result_text
                    box_color = (0, 255, 255) # Yellow denotes a dynamic inference result
                    dynamic_display_timer -= 1
                else:
                    # Model expects shape (1, 273)
                    input_static = np.expand_dims(features_273, axis=0)
                    preds = model_static.predict(input_static, verbose=0)[0]
                    predicted_idx = np.argmax(preds)
                    confidence = preds[predicted_idx]
                    
                    # Display prediction only if it meets a reasonable confidence threshold
                    if confidence > 0.60:
                        current_text = f"{static_classes[predicted_idx]} ({confidence*100:.1f}%)"
                        box_color = (0, 255, 0) # Green denotes a static inference result
                    else:
                        current_text = "Low confidence"
                        
        else:
            # Continue ticking down the display timer even when no hand is present
            if dynamic_display_timer > 0:
                current_text = dynamic_result_text
                box_color = (0, 255, 255)
                dynamic_display_timer -= 1
        
        # Simple UI Design (Colored background banner for text visibility)
        cv2.rectangle(frame, (0, h-60), (w, h), box_color, -1)
        cv2.putText(frame, current_text, (20, h-20), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Hotkey guide overlay
        cv2.putText(frame, "[D]: Record J/Z   [Q]: Quit", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        cv2.imshow('SIBI Real-time Translator', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d') and not is_recording_dynamic:
            is_recording_dynamic = True
            dynamic_frames = [] # Flush buffer before recording
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
