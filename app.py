from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "mobilenetv3small_float32.tflite"

# IMPORTANT: This order must exactly match the model's training label order.
CLASS_CODES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevi",
    "BCC": "Basal cell carcinoma",
    "AKIEC": "Actinic keratoses / intraepithelial carcinoma",
    "BKL": "Benign keratosis-like lesions",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesions",
}

st.set_page_config(
    page_title="Skin Lesion Classification — Research Demo",
    page_icon="🔬",
    layout="centered",
)


class ModelRuntime:
    """Cached LiteRT interpreter protected against concurrent Streamlit sessions."""

    def __init__(self, model_path: Path) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.lock = threading.Lock()

    def predict(self, image_tensor: np.ndarray) -> np.ndarray:
        with self.lock:
            self.interpreter.set_tensor(self.input_detail["index"], image_tensor)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_detail["index"])
        return np.asarray(result[0], dtype=np.float32)


@st.cache_resource(show_spinner="Loading the research model…")
def load_model() -> ModelRuntime:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Copy "
            "mobilenetv3small_float32.tflite into the app's models/ folder."
        )
    return ModelRuntime(MODEL_PATH)


def preprocess_image(image: Image.Image, input_shape: tuple[int, ...]) -> np.ndarray:
    """Resize RGB image and scale pixels to [0, 1], matching the project pipeline."""
    if len(input_shape) != 4 or input_shape[0] not in (1, -1):
        raise ValueError(f"Unexpected model input shape: {input_shape}")

    height, width = int(input_shape[1]), int(input_shape[2])
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0).astype(np.float32)


st.title("🔬 Skin Lesion Classification")
st.caption("MobileNetV3Small · TensorFlow Lite · Research prototype")

st.warning(
    "Research demonstration only — this model is not clinically validated and "
    "cannot diagnose or rule out skin cancer. Do not use its output to make "
    "healthcare decisions. For a changing, bleeding, painful, or concerning lesion, "
    "seek assessment from a qualified healthcare professional."
)

st.markdown(
    "Upload a dermoscopic or skin-lesion image to see the model's experimental "
    "class scores. The app processes the image on its hosting server; do not upload "
    "identifiable or sensitive images unless you understand and accept that."
)

uploaded_file = st.file_uploader(
    "Choose a skin-lesion image",
    type=["jpg", "jpeg", "png", "webp"],
    help="For research testing, use an image similar to the model's training data.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Run experimental classification", type="primary", use_container_width=True):
            with st.spinner("Running model inference…"):
                runtime = load_model()
                input_shape = tuple(int(value) for value in runtime.input_detail["shape"])
                tensor = preprocess_image(image, input_shape)
                scores = runtime.predict(tensor)

            if scores.size != len(CLASS_CODES):
                st.error(
                    f"Expected {len(CLASS_CODES)} outputs but received {scores.size}. "
                    "Check that the selected model matches the configured class order."
                )
            else:
                best_index = int(np.argmax(scores))
                best_code = CLASS_CODES[best_index]
                st.subheader("Model output (not a diagnosis)")
                st.markdown(f"**Highest-scoring class:** `{best_code}` — {CLASS_NAMES[best_code]}")

                score_sum = float(np.sum(scores))
                looks_like_probabilities = (
                    np.all(np.isfinite(scores))
                    and np.all(scores >= -1e-6)
                    and np.all(scores <= 1.00001)
                    and abs(score_sum - 1.0) < 0.02
                )
                if looks_like_probabilities:
                    st.caption(
                        "Scores are the model's softmax outputs. They are not calibrated "
                        "clinical probabilities or measures of diagnostic certainty."
                    )
                    display_values = np.clip(scores, 0.0, 1.0) * 100.0
                    chart_title = "Model output scores (%)"
                    score_table = {
                        "Class": [f"{code} — {CLASS_NAMES[code]}" for code in CLASS_CODES],
                        "Model score (%)": [round(float(v), 2) for v in display_values],
                    }
                else:
                    st.caption("The output does not look like a normalized probability distribution; displaying raw model scores.")
                    display_values = scores
                    chart_title = "Raw model output scores"
                    score_table = {
                        "Class": [f"{code} — {CLASS_NAMES[code]}" for code in CLASS_CODES],
                        "Model score": [round(float(v), 6) for v in display_values],
                    }

                st.markdown(f"**{chart_title}**")
                chart_data = {
                    "Class": [f"{code} — {CLASS_NAMES[code]}" for code in CLASS_CODES],
                    "Score": [float(v) for v in display_values],
                }
                st.bar_chart(chart_data, x="Class", y="Score", horizontal=True)
                st.dataframe(score_table, hide_index=True, use_container_width=True)
                st.info(
                    "The result is an experimental classification from a model trained on a specific "
                    "dataset. Image quality, camera differences, skin tone, and class imbalance may "
                    "affect performance. It must not replace professional evaluation."
                )
    except Exception as exc:
        st.error(f"Could not process this image or run the model: {exc}")

with st.expander("About this research prototype"):
    st.write(
        "This interface wraps the project's MobileNetV3Small TensorFlow Lite model. "
        "The model was evaluated experimentally on the project's held-out test split; "
        "it has not undergone clinical validation or prospective evaluation."
    )
    st.write("Configured model input: 224 × 224 RGB image with pixel values scaled to [0, 1].")
