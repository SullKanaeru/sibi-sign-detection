"""
Minimal Keras-to-TensorFlow.js Converter
Converts .keras models to TFJS LayersFormat manually without
requiring tensorflow_decision_forests, jax, or tensorflow_hub.
"""

import os, sys, json, struct
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def convert_model(model_path, output_dir, label_path=None):
    print(f"\n[INFO] Loading: {model_path}")
    model = tf.keras.models.load_model(model_path)

    os.makedirs(output_dir, exist_ok=True)

    # ---- Build weight manifest ----
    weights_list = []
    all_weights_bytes = bytearray()

    for var in model.weights:
        arr = var.numpy().astype(np.float32)
        raw = arr.tobytes()
        # TFJS expects the name without ':0' suffix
        name = var.name.replace(':0', '')
        weights_list.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "float32"
        })
        all_weights_bytes.extend(raw)

    bin_filename = "group1-shard1of1.bin"
    bin_path = os.path.join(output_dir, bin_filename)
    with open(bin_path, 'wb') as f:
        f.write(all_weights_bytes)

    # ---- Build model.json ----
    model_config = json.loads(model.to_json())
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras {tf.__version__}",
        "convertedBy": "custom-minimal-converter",
        "modelTopology": model_config,
        "weightsManifest": [
            {
                "paths": [bin_filename],
                "weights": weights_list
            }
        ]
    }

    json_path = os.path.join(output_dir, "model.json")
    with open(json_path, 'w') as f:
        json.dump(model_json, f)

    print(f"[OK] Converted! Files: {os.listdir(output_dir)}")
    print(f"     model.json : {os.path.getsize(json_path)/1024:.1f} KB")
    print(f"     weights.bin: {os.path.getsize(bin_path)/1024:.1f} KB")

    # ---- Save label map ----
    if label_path and os.path.exists(label_path):
        classes = np.load(label_path, allow_pickle=True)
        label_map = {str(i): str(c) for i, c in enumerate(classes)}
        lm_path = os.path.join(output_dir, "label_map.json")
        with open(lm_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        print(f"     label_map  : {label_map}")


def main():
    print("=" * 55)
    print("  Keras to TensorFlow.js Converter (Minimal)")
    print("=" * 55)

    convert_model(
        os.path.join(MODELS_DIR, "model_static.keras"),
        os.path.join(MODELS_DIR, "tfjs_static"),
        os.path.join(MODELS_DIR, "static_classes.npy")
    )

    convert_model(
        os.path.join(MODELS_DIR, "model_dynamic.keras"),
        os.path.join(MODELS_DIR, "tfjs_dynamic"),
        os.path.join(MODELS_DIR, "dynamic_classes.npy")
    )

    print("\n[SUCCESS] Both models ready for TensorFlow.js!")
    print("  Usage:")
    print("    const m = await tf.loadLayersModel('./models/tfjs_static/model.json');")
    print("=" * 55)


if __name__ == '__main__':
    main()
