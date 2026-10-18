"""Food image classifier + calorie estimator (Gradio demo).
 
Upload a photo of food -> the model predicts one of 101 Food-101 dishes,
looks up approximate calories for a typical serving, and shows a Grad-CAM
heatmap of the image regions that drove the prediction.
 
Works both in the GitHub repo layout (app/app.py, models/, data/) and on
Hugging Face Spaces (all files in the same folder as app.py).
"""
import warnings
warnings.filterwarnings("ignore")
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
SEARCH_DIRS = [HERE, HERE.parent,
               HERE.parent / "models", HERE.parent / "data"]

def find_file(filename: str) -> Path:
    """Search for a file in the repo layout or on Spaces."""
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename} in {SEARCH_DIRS}")

model = tf.keras.models.load_model(find_file("food_classifier.keras"), compile=False)
class_names = json.load(open(find_file("class_names.json")))
cal_df = pd.read_csv(find_file("calories.csv")).set_index("class")

missing = set(class_names) - set(cal_df.index)
if missing:
    raise ValueError(f"calories.csv is missing classes: {sorted(missing)}")

