# Python script for captioning dataset images using GPT 4.1 Mini

from openai import OpenAI
import datetime
import base64
import os

IMAGE_FOLDER = "/Users/cassandralanza/Documents/SnapshotsDataSet"
MAX_IMAGES = 28
MODEL_NAME = "gpt-4.1-mini"
prompt = (
    "You are an autonomous vehicle providing concise alerts for passengers."
    "Alerts must be purely informational; do not provide advice or instructions."
    "Use a neutral tone. Keep the alert to a single short phrase under 10 words."
    "Example alerts: 'Animal on road ahead', 'Pedestrian near crosswalk', 'Sharp turn ahead'."
)

os.environ["OPENAI_API_KEY"] = "InsertKeyHere"
GEMINI_API_KEY = os.getenv("OPEN_API_KEY")
client = OpenAI();

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
    print(f"Analyzing image {i}/{len(image_paths)}: {os.path.basename(path)}")

    with open(path, "rb") as img_file:
        image_bytes = img_file.read()
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")


    init_time = datetime.datetime.now()
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                ]
            }
        ],
    )
    API_resp_time = datetime.datetime.now() - init_time
    print(API_resp_time)

    caption = response.output_text.strip()
    captions.append((os.path.basename(path), caption, API_resp_time))
    print(f" Caption: {caption}\n")

    output_path = os.path.join(IMAGE_FOLDER, "captions_GPT.txt")
    with open(output_path, "w") as f:
            for filename, caption, API_resp_time in captions:
                f.write(f"{filename}:: {caption}:: {API_resp_time.total_seconds()}\n")

    print(f"All captions save to {output_path}")
