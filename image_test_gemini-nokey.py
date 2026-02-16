# Python script for captioning dataset images using Gemini 2.5 Flash Lite

from google import genai
from google.genai import types
import datetime
import os

IMAGE_FOLDER = "/Users/cassandralanza/Documents/SnapshotsDataset"
MAX_IMAGES = 28
MODEL_NAME = "gemini-2.5-flash-lite"
prompt = (
    "You are an autonomous vehicle providing concise alerts for passengers."
    "Alerts must be purely informational; do not provide advice or instructions."
    "Use a neutral tone. Keep the alert to a single short phrase under 10 words."
    "Example alerts: 'Animal on road ahead', 'Pedestrian near crosswalk', 'Sharp turn ahead'."
)

os.environ["GEMINI_API_KEY"] = "InsertKeyHere"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client();

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

    with open(path, "rb") as img_file:
        image_bytes = img_file.read()

    init_time = datetime.datetime.now()
    print(init_time)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
            ),
            prompt
        ]
    )
    API_resp_time = datetime.datetime.now() - init_time
    print(API_resp_time)

    caption = response.text.strip()
    captions.append((os.path.basename(path), caption, API_resp_time))
    print(f" Caption: {caption}\n")

    output_path = os.path.join(IMAGE_FOLDER, "captions_Gemini.txt")
    with open(output_path, "w") as f:
            for filename, caption, API_resp_time in captions:
                f.write(f"{filename}:: {caption}:: {API_resp_time.total_seconds()}\n")

    print(f"All captions save to {output_path}")
