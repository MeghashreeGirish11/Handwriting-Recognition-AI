from flask import Flask, render_template, request, jsonify

import cv2
import numpy as np

from PIL import Image

import torch

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

app = Flask(__name__)

# ------------------------------------
# Load TrOCR Model
# ------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading TrOCR model...")

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten"
)

model.to(device)

print("===================================")
print("AI Model Loaded Successfully!")
print("Running on:", device)
print("===================================")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recognize", methods=["POST"])
def recognize():

    if "image" not in request.files:
        return jsonify({
            "text": "No image uploaded."
        })

    image = request.files["image"]

    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(
        image.read(),
        np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Remove noise
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu Thresholding
    thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    # Save processed image
    cv2.imwrite(
        "processed.png",
        thresh
    )

    # ------------------------------------
    # TrOCR Inference
    # ------------------------------------

    # Convert grayscale image to RGB
    rgb = cv2.cvtColor(
        thresh,
        cv2.COLOR_GRAY2RGB
    )

    # Convert to PIL Image
    pil_image = Image.fromarray(rgb)

    # Prepare image
    pixel_values = processor(
        images=pil_image,
        return_tensors="pt"
    ).pixel_values

    pixel_values = pixel_values.to(device)

    # Run model
    generated_ids = model.generate(pixel_values)

    # Decode output
    text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    print("OCR OUTPUT:", repr(text))

    return jsonify({
        "text": text
    })


if __name__ == "__main__":
    app.run(debug=True)