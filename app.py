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
