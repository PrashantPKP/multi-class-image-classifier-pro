import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.preprocessing import image as keras_image

# ── Model configuration ──────────────────────────────────────────────────────

# Resolve absolute path relative to this file's location so that the model is
# found correctly regardless of the working directory uvicorn is started from
# (local dev, Docker /app, Render native runner, etc.)
_BASE_DIR = Path(__file__).resolve().parent.parent   # repo root
MODEL_PATH = _BASE_DIR / "models" / "multi_class_classifier_v1.h5"

DEFAULT_CLASS_NAMES = [
    "bike",
    "boll",
    "buildings",
    "bus",
    "car",
    "cats",
    "dogs",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
    "truck",
]

# ── Load class names ──────────────────────────────────────────────────────────

def load_class_names():
    class_file = _BASE_DIR / "models" / "class_names.json"
    if class_file.exists():
        with open(class_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()

# ── Load model once at startup ────────────────────────────────────────────────

def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please place 'multi_class_classifier_v1.keras' inside the 'models/' directory."
        )
    return tf.keras.models.load_model(MODEL_PATH)


model = _load_model()

# ── Public helpers ────────────────────────────────────────────────────────────

def get_model_path() -> Path:
    """Return the path of the currently loaded model."""
    return MODEL_PATH


def reload_model() -> str:
    """Reload the model from disk at runtime (no server restart needed)."""
    global model
    model = _load_model()
    return str(MODEL_PATH)

# ── Inference ─────────────────────────────────────────────────────────────────

def predict_image(img_path):
    # Load and resize to the model's expected input size (224×224)
    img = keras_image.load_img(img_path, target_size=(224, 224))

    # Convert to float32 array — pixel range [0, 255]
    # NOTE: Do NOT call mobilenet_v2.preprocess_input() here.
    # The model was trained with the preprocessing layer baked inside the
    # model graph itself (tf.math.truediv + tf.math.subtract layers).
    # Calling it again here would double-preprocess and corrupt predictions.
    img_array = keras_image.img_to_array(img)          # (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)      # (1, 224, 224, 3)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return predicted_class, confidence