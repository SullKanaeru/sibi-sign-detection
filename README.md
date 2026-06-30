# SIBI Sign Language Recognition — Computer Vision Pipeline

Real-time Indonesian Sign Language (SIBI) alphabet recognition system built on **MediaPipe Hands** for hand landmark extraction, **TensorFlow/Keras** for classification, and **OpenCV** for webcam inference. The pipeline handles both **static** gestures (A–Y, excluding J and Z) and **dynamic** gestures (J, Z) through separate model architectures.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1 — Static Dataset Collection](#stage-1--static-dataset-collection)
  - [Stage 2 — Dynamic Dataset Collection](#stage-2--dynamic-dataset-collection)
  - [Stage 3 — Feature Extraction](#stage-3--feature-extraction)
  - [Stage 4 — Static Model Training](#stage-4--static-model-training)
  - [Stage 5 — Dynamic Model Training](#stage-5--dynamic-model-training)
  - [Stage 6 — TensorFlow.js Conversion](#stage-6--tensorflowjs-conversion)
- [Inference](#inference)
- [Feature Engineering Details](#feature-engineering-details)
- [Model Architectures](#model-architectures)

---

## Architecture Overview

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Webcam      │───▶│  MediaPipe Hands   │───▶│  Feature Eng.    │───▶│  Keras Model   │
│  (OpenCV)    │    │  21 landmarks (3D) │    │  283 / 63 feat.  │    │  Dense / LSTM  │
└──────────────┘    └───────────────────┘    └──────────────────┘    └───────┬────────┘
                                                                            │
                                                                            ▼
                                                                    ┌────────────────┐
                                                                    │  Predicted      │
                                                                    │  Letter (A-Z)   │
                                                                    └────────────────┘
```

The system operates in two modes:

| Mode     | Gestures | Model        | Input Shape        | Features |
|----------|----------|--------------|--------------------|----------|
| Static   | A–Y (excl. J, Z) | Dense NN | `(1, 283)`   | Engineered landmark features |
| Dynamic  | J, Z     | Bi-layer LSTM | `(1, 60, 63)` | Wrist-normalized coordinate sequences |

---

## Project Structure

```
model-cv/
├── app.py                          # Real-time inference application
├── data/
│   ├── raw/
│   │   ├── kanan/                  # Right-hand static images (A-Y)
│   │   │   ├── A/
│   │   │   │   ├── 0.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   ├── kiri/                   # Left-hand static images (A-Y)
│   │   ├── J/                      # Dynamic sequences (J)
│   │   │   ├── seq_0001.npy        # shape: (60, 21, 3)
│   │   │   └── ...
│   │   └── Z/                      # Dynamic sequences (Z)
│   └── static_features.csv         # Extracted landmark CSV (03_extract_features.py output)
├── models/
│   ├── model_static.keras          # Trained static Dense model
│   ├── model_dynamic.keras         # Trained dynamic LSTM model
│   ├── static_classes.npy          # Label encoder mapping (static)
│   ├── dynamic_classes.npy         # Label encoder mapping (dynamic)
│   ├── tfjs_static/                # TensorFlow.js static model
│   └── tfjs_dynamic/               # TensorFlow.js dynamic model
├── scripts/
│   ├── 01_collect_static.py        # Stage 1: Static image collection
│   ├── 02_collect_dynamic.py       # Stage 2: Dynamic sequence collection
│   ├── 03_extract_features.py      # Stage 3: MediaPipe landmark extraction
│   ├── 04_train_static.py          # Stage 4: Static model training
│   ├── 05_train_dynamic.py         # Stage 5: Dynamic model training
│   └── 06_convert_to_tfjs.py       # Stage 6: Keras → TFJS conversion
└── logs/
    ├── static_training_history.png
    └── dynamic_training_history.png
```

---

## Prerequisites

```
Python 3.10
TensorFlow 2.x
MediaPipe 0.10.x
OpenCV (opencv-contrib-python)
scikit-learn
pandas
matplotlib
numpy
```

Activate the virtual environment before running any script:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat
```

---

## Pipeline Stages

### Stage 1 — Static Dataset Collection

**Script:** `scripts/01_collect_static.py`

Captures raw webcam frames for each static gesture class (A–Y, excluding J and Z) across both left (`kiri`) and right (`kanan`) hands.

```bash
python scripts/01_collect_static.py
```

**Process:**

1. Iterates through 24 static gesture classes × 2 hand categories (kanan, kiri).
2. For each class, enters an idle loop displaying MediaPipe hand landmarks in real-time. The operator positions their hand and presses `q` to begin recording.
3. Captures exactly **100 raw frames** per class per hand at 50ms intervals, introducing natural micro-variations in hand pose.
4. Saves raw `.jpg` frames (without landmark overlays) to `data/raw/<hand_category>/<class_name>/`.

**Output:** `data/raw/kanan/A/0.jpg`, `data/raw/kanan/A/1.jpg`, ..., `data/raw/kiri/Y/99.jpg`

**Data volume:** ~4,800 images (24 classes × 2 hands × 100 frames).

---

### Stage 2 — Dynamic Dataset Collection

**Script:** `scripts/02_collect_dynamic.py`

Captures temporal landmark sequences for dynamic gestures (J, Z) that require hand movement trajectories.

```bash
python scripts/02_collect_dynamic.py
```

**Process:**

1. Displays a visual gesture guide screen with instructions on how to perform J or Z.
2. After the operator presses `SPACE`, a 3-second countdown initiates.
3. Records **60 consecutive frames** of raw MediaPipe hand landmark data per sample.
4. Saves each sequence as a NumPy array with shape `(60, 21, 3)` — 60 timesteps, 21 landmarks, 3 coordinates (x, y, z).
5. Provides real-time UI feedback including hand detection status, recording progress bar, and overall collection quota.
6. Supports mid-recording `R` (retry) and `Q` (quit) controls.

**Output:** `data/raw/J/seq_0001.npy`, ..., `data/raw/Z/seq_0200.npy`

**Target:** 200 sequences per gesture class.

---

### Stage 3 — Feature Extraction

**Script:** `scripts/03_extract_features.py`

Processes the raw static images collected in Stage 1 and extracts 3D hand landmarks using MediaPipe Hands in `static_image_mode=True`.

```bash
python scripts/03_extract_features.py
```

**Process:**

1. Recursively scans all `.jpg` files under `data/raw/`.
2. For each image, runs MediaPipe Hands detection (`max_num_hands=1`) and extracts 21 3D landmarks.
3. Flattens each landmark set into a 63-element vector `[x₀, y₀, z₀, x₁, y₁, z₁, ..., x₂₀, y₂₀, z₂₀]`.
4. Deduces class label and hand category from the directory path structure: `data/raw/<hand_category>/<class_name>/`.
5. Compiles all successful extractions into a single CSV with columns: `label, hand_type, x_0, y_0, z_0, ..., x_20, y_20, z_20`.

**Output:** `data/static_features.csv` (65 columns: 1 label + 1 hand_type + 63 landmark coordinates)

Images where MediaPipe fails to detect a hand are silently skipped and logged as failed.

---

### Stage 4 — Static Model Training

**Script:** `scripts/04_train_static.py`

Trains a Dense Neural Network on the extracted static landmark features with extensive feature engineering.

```bash
python scripts/04_train_static.py
```

**Process:**

1. **Load:** Reads `data/static_features.csv` and separates raw 63-feature landmark vectors from labels.
2. **Feature Engineering:** Applies `normalize_landmarks()` to expand the 63 raw coordinates into a **283-dimensional feature vector** per sample (see [Feature Engineering Details](#feature-engineering-details)).
3. **Label Encoding:** Maps string labels (A, B, ..., Y) to integer indices via `sklearn.LabelEncoder`. Saves the encoding map as `models/static_classes.npy`.
4. **Train/Test Split:** 80/20 stratified split (`random_state=42`).
5. **Class Weight Balancing:** Computes balanced class weights via `sklearn.utils.class_weight.compute_class_weight` to mitigate dataset imbalance (e.g., T: 102 samples vs X: 200 samples).
6. **Model Training:** Trains a 3-layer Dense NN with BatchNormalization, Dropout, and EarlyStopping (`patience=15`, `restore_best_weights=True`). Applies `class_weight` during `model.fit()`.
7. **Model Serialization:** Saves the trained model as `models/model_static.keras`.

**Output:** `models/model_static.keras`, `models/static_classes.npy`, `logs/static_training_history.png`

---

### Stage 5 — Dynamic Model Training

**Script:** `scripts/05_train_dynamic.py`

Trains a stacked LSTM network on temporal landmark sequences for dynamic gesture classification.

```bash
python scripts/05_train_dynamic.py
```

**Process:**

1. **Load:** Auto-detects dynamic gesture classes by scanning for `seq_*.npy` files under `data/raw/`.
2. **Sequence Normalization:** Each sequence `(60, 21, 3)` is processed via `normalize_sequence()`:
   - Subtracts wrist (landmark 0) coordinates from all other landmarks per frame.
   - Applies scale invariance by dividing by the max absolute coordinate per frame.
   - Flattens each frame from `(21, 3)` to `(63,)`, yielding a final shape of `(60, 63)`.
3. **Model Training:** Trains a 2-layer LSTM (64 → 128 units) followed by Dense layers (64 → 32 → `num_classes`). Uses `sparse_categorical_crossentropy` loss with EarlyStopping.

**Output:** `models/model_dynamic.keras`, `models/dynamic_classes.npy`, `logs/dynamic_training_history.png`

---

### Stage 6 — TensorFlow.js Conversion

**Script:** `scripts/06_convert_to_tfjs.py`

Converts both Keras models to TensorFlow.js LayersFormat for browser-based deployment.

```bash
python scripts/06_convert_to_tfjs.py
```

**Process:**

1. Loads each `.keras` model and iterates over all weight tensors.
2. Serializes weights to a single binary shard (`group1-shard1of1.bin`) in float32 format.
3. Constructs a `model.json` manifest containing model topology (from `model.to_json()`) and weight manifest with tensor names, shapes, and dtypes.
4. Saves a `label_map.json` mapping integer class indices to string labels.

**Output:** `models/tfjs_static/model.json`, `models/tfjs_dynamic/model.json`, and their respective binary weight shards.

---

## Inference

**Script:** `app.py`

Real-time webcam inference combining both static and dynamic models in a unified loop.

```bash
python app.py
```

**Controls:**

| Key | Action |
|-----|--------|
| `d` | Start recording a dynamic gesture (60 frames for J/Z) |
| `q` | Quit the application |

**Inference Loop:**

1. **Frame Capture:** Reads and horizontally flips each webcam frame for mirror view.
2. **Hand Detection:** Processes the frame through MediaPipe Hands (`max_num_hands=1`) and renders the hand skeleton overlay.
3. **Feature Extraction:** Calls `extract_and_normalize_landmarks()` which produces two feature vectors simultaneously:
   - `features_63` (63-dim): Wrist-normalized coordinates for the dynamic LSTM model.
   - `features_283` (283-dim): Full engineered feature set for the static Dense model.
4. **Static Path:** When not recording dynamic gestures, the 283-dim vector is fed into `model_static` for per-frame classification. Predictions with confidence < 60% are suppressed.
5. **Dynamic Path:** When `d` is pressed, the system accumulates 60 frames of 63-dim features, then passes the `(1, 60, 63)` tensor to `model_dynamic` for sequence classification. The result is displayed for ~3 seconds (90 frames).
6. **Display:** A colored banner at the bottom of the frame shows the current prediction (green = static, yellow = dynamic).

---

## Feature Engineering Details

The static model uses a **283-dimensional** engineered feature vector per hand frame, constructed from the 21 raw MediaPipe 3D landmarks:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| Wrist-relative normalized coordinates | 63 | `(x - wrist_x, y - wrist_y, z - wrist_z)` for each of 21 landmarks |
| Pairwise Euclidean distances | 210 | `C(21, 2) = 210` distances between all unique landmark pairs |
| U vs R crossing features | 4 | Signed X/Z differences between index tip (lm8) and middle tip (lm12), and between index PIP (lm7) and middle PIP (lm11). Captures finger crossing direction. |
| T vs X discrimination features | 6 | Thumb-tip position relative to index/middle MCPs, thumb-to-finger-midpoint distance, index finger curl angle (cosine), thumb lateral position, thumb-to-index-PIP distance |

All coordinates and distances are divided by the maximum absolute coordinate value per frame (**scale invariance**), ensuring the model is invariant to hand-camera distance.

---

## Model Architectures

### Static Model (Dense NN)

```
Input (283) → Dense(512, ReLU) → BN → Dropout(0.3)
           → Dense(256, ReLU) → BN → Dropout(0.3)
           → Dense(128, ReLU) → BN → Dropout(0.2)
           → Dense(24, Softmax)
```

- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam
- **Regularization:** BatchNormalization + Dropout + EarlyStopping (patience=15) + balanced class weights
- **Input:** 283-dim engineered feature vector
- **Output:** 24-class probability distribution (A–Y, excl. J, Z)

### Dynamic Model (LSTM)

```
Input (60, 63) → LSTM(64, return_sequences=True)
              → LSTM(128, return_sequences=False)
              → Dense(64, ReLU)
              → Dropout(0.2)
              → Dense(32, ReLU)
              → Dense(2, Softmax)
```

- **Loss:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam
- **Regularization:** Dropout + EarlyStopping (patience=15)
- **Input:** 60-frame sequence of 63-dim normalized landmark vectors
- **Output:** 2-class probability distribution (J, Z)
