# Python script for captioning dataset images using BLIP

from flask import Flask, request, jsonify, make_response
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import datetime
import torch
import os
import io

app = Flask(__name__)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load BLIP model and processor once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

IMAGE_FOLDER = "/Users/cassandralanza/Documents/SnapshotsDataset"
MAX_IMAGES = 28

# set image paths
image_paths = [
    os.path.join(IMAGE_FOLDER, f)
    for f in sorted(os.listdir(IMAGE_FOLDER))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
][:MAX_IMAGES]

if not image_paths:
    raise FileNotFoundError("No valid images found in folder.")

captions = [] # array of captions

for i, path in enumerate(image_paths, start=1):
    print(f"Analyzing image {i}/{len(image_paths)}: {os.path.basename(path)}\n")

    # read image
    with open(path, "rb") as img_file:
        image_bytes = img_file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Generate caption
    init_time = datetime.datetime.now()
    inputs = processor(image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    API_resp_time = datetime.datetime.now() - init_time

    captions.append((os.path.basename(path), caption, API_resp_time))
    print(f" Caption: {caption}\n")

    output_path = os.path.join(IMAGE_FOLDER, "captions_BLIP.txt")
    with open(output_path, "w") as f:
            for filename, caption, API_resp_time in captions:
                f.write(f"{filename}:: {caption}:: {API_resp_time.total_seconds()}\n")

    print(f"All captions save to {output_path}")
