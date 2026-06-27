"""
Script for collecting static hand gesture images.
It uses MediaPipe to visualize hand landmarks while saving raw images to the dataset directory.
"""
import os
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands for robust 3D hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# Output directory for the raw image dataset
DATA_DIR = '../data/raw'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the target gesture classes (American Sign Language alphabet)
# Note: J and Z are dynamic and handled separately.
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
hands_categories = ['kanan', 'kiri']  # 'kanan' = Right, 'kiri' = Left
dataset_size = 100  # Number of samples to record per class per hand category

# Initialize the default webcam
cap = cv2.VideoCapture(0)

for class_name in classes:
    for hand_cat in hands_categories:
        class_dir = os.path.join(DATA_DIR, hand_cat, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            
        print(f"\n[INFO] Preparing to record gesture '{class_name}' with {hand_cat.upper()} hand")
        print("[INFO] Press 'q' on the video window when your hand is in position and you are ready to start.")
        
        # Idle loop: Wait for the user to get into position and press 'q'
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Flip the frame horizontally for an intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Display instructional text on the frame
            cv2.putText(frame, f'{hand_cat.upper()} HAND - Gesture "{class_name}"', 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press "q" to start recording', 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                        
            # Process the frame with MediaPipe to display real-time landmarks
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
                        
            cv2.imshow('SIBI Dataset Collector', frame)
            
            # Break the idle loop when 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        counter = 0
        print(f"[INFO] Started recording {hand_cat.upper()} hand for gesture: {class_name}")
        
        # Active recording loop: Capture exactly `dataset_size` frames
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            
            # --- SAVE RAW IMAGE ---
            # We save the raw frame without landmark overlays to allow clean feature extraction later
            img_path = os.path.join(class_dir, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            
            # --- VISUALIZATION DURING RECORDING ---
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
                        
            # Display recording progress
            cv2.putText(frame, f'Recording {hand_cat.upper()} "{class_name}": {counter+1}/{dataset_size}', 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('SIBI Dataset Collector', frame)
            cv2.waitKey(50)  # 50ms delay between frames ensures slight natural variation in data
            counter += 1

print("\n[INFO] Data collection complete!")
cap.release()
cv2.destroyAllWindows()
