"""Food image classifier + calorie estimator (Gradio demo).
Upload a photo of food -> the model predicts one of 101 Food-101 dishes,
looks up approximate calories for a typical serving, and shows a Grad-CAM
heatmap of the image regions that drove the prediction.
Works both in the GitHub repo layout (app/app.py, models/, data/) and on
Hugging Face Spaces (all files in the same folder as app.py).
"""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.40   # below this, the app says "not sure"

HERE = Path(__file__).resolve().parent
SEARCH_DIRS = [HERE, HERE.parent, HERE / "models", HERE / "data",
               HERE.parent / "models", HERE.parent / "data"]


def find_file(name: str) -> Path:
    for d in SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find '{name}'. Looked in: {[str(d) for d in SEARCH_DIRS]}")


model = tf.keras.models.load_model(find_file("food_classifier.keras"), compile=False)
class_names = json.load(open(find_file("class_names.json")))
cal_df = pd.read_csv(find_file("calories.csv")).set_index("class")

missing = set(class_names) - set(cal_df.index)
if missing:
    raise ValueError(f"calories.csv is missing classes: {sorted(missing)}")

# Pieces of the model needed for Grad-CAM (backbone -> pooling -> classifier)
backbone = next(l for l in model.layers if l.name.startswith("efficientnet"))
pool = next(l for l in model.layers
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
head = model.layers[-1]

def jet(v: np.ndarray) -> np.ndarray:
    """Simple 'jet'-style colormap for values in [0, 1] (avoids a matplotlib dependency)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def forward_with_cam(x: tf.Tensor):
    """One forward pass that returns class probabilities and a 224x224 Grad-CAM map."""
    with tf.GradientTape() as tape:
        conv = backbone(x, training=False)              # (1, 7, 7, 1280)
        tape.watch(conv)
        preds = head(pool(conv))                        
        class_idx = tf.argmax(preds[0])
        score = tf.gather(preds[0], class_idx)
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2))        # (1, 1280)
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()
    return preds[0].numpy(), cam


def predict(image, portion):
    if image is None:
        return {}, "Upload a food photo to get started.", None

    img = Image.fromarray(image).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32)            
    probs, cam = forward_with_cam(tf.constant(arr[None]))

    top = probs.argsort()[-3:][::-1]
    labels = {class_names[i].replace("_", " ").title(): float(probs[i]) for i in top}

    overlay = (0.6 * arr / 255.0 + 0.4 * jet(cam)).clip(0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    best = class_names[top[0]]
    conf = float(probs[top[0]])
    if conf < CONFIDENCE_THRESHOLD:
        msg = (f"### Not sure\nMy best guess is **{best.replace('_', ' ')}** "
               f"({conf:.0%} confidence), which is too low to estimate calories. "
               "Try a clearer, closer photo of a single dish.")
    else:
        kcal = round(float(cal_df.loc[best, "kcal"]) * float(portion))
        serving = cal_df.loc[best, "serving"]
        msg = (f"### {best.replace('_', ' ').title()}  ({conf:.0%} confidence)\n"
               f"**~{kcal} kcal** for {portion:g} x typical serving ({serving})\n\n"
               "_Rough estimate only. Real portions and recipes vary._")
    return labels, msg, overlay


with gr.Blocks(title="Food Classifier & Calorie Estimator") as demo:
    gr.Markdown(
        "# 🍽️ Food Classifier & Calorie Estimator\n"
        "Upload a photo of a dish. The model (EfficientNetB0, trained on Food-101, "
        "101 classes) predicts what it is, estimates calories for a typical serving, "
        "and shows a **Grad-CAM** heatmap of what it looked at.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Food photo")
            portion = gr.Slider(0.5, 3.0, value=1.0, step=0.25,
                                label="Portion size (x typical serving)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=3, label="Top predictions")
            out_msg = gr.Markdown()
            out_cam = gr.Image(label="Grad-CAM: where the model looked")

    btn.click(predict, [inp, portion], [out_label, out_msg, out_cam])
    inp.change(predict, [inp, portion], [out_label, out_msg, out_cam])
    portion.release(predict, [inp, portion], [out_label, out_msg, out_cam])

    gr.Markdown(
        "Calories are approximate per-serving values, not verified nutrition data. "
        "Covers 101 dishes only. Not a dietary or medical tool.")

if __name__ == "__main__":
    demo.launch()