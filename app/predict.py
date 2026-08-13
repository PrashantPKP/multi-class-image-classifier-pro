import tensorflow as tf
import numpy as np
import json
from pathlib import Path
from tensorflow.keras.preprocessing import image as keras_image

MODEL_PATHS = [
    Path("models") / "multi_class_classifier_v1.keras",
    Path("models") / "intel_image_classification_model.keras",
]


def load_model_from_candidates():
    for path in MODEL_PATHS:
        if path.exists():
            m = tf.keras.models.load_model(path)
            # record which file was used
            load_model_from_candidates.selected_path = path
            return m
    raise FileNotFoundError(
        "No trained model found. Expected one of: new_tained_data_v1.keras, new_trained_data_v1.keras, intel_image_classification_model.keras"
    )


# Load model once when application starts
model = load_model_from_candidates()


def reload_model():
    """Reload model from candidate paths at runtime."""
    global model
    model = load_model_from_candidates()
    return str(get_model_path())


def get_model_path():
    return getattr(load_model_from_candidates, "selected_path", None)

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


def load_class_names():
    class_file = Path("models") / "class_names.json"
    if class_file.exists():
        with open(class_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = load_class_names()


def predict_image(img_path):
    # Load image and resize to model's expected input (224x224)
    img = keras_image.load_img(
        img_path,
        target_size=(224, 224)
    )

    # Convert to float32 array with shape (224, 224, 3) — pixel range [0, 255]
    # NOTE: Do NOT call mobilenet_v2.preprocess_input() here.
    # The model was trained with the preprocessing layer baked inside the model
    # graph itself (tf.math.truediv + tf.math.subtract layers). Calling it
    # again here would double-preprocess the image and corrupt all predictions.
    img_array = keras_image.img_to_array(img)  # float32, range [0, 255]
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

    prediction = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return predicted_class, confidence