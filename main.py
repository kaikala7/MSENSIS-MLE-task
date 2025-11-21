import os
import io
import time
import argparse
from typing import Dict

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import ViTForImageClassification, AutoImageProcessor

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# =========================
# Global configuration
# =========================

# Paths for dataset and labels
DATA_DIR = "dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

# Column names in labels.csv
IMAGE_COL = "image_name"
LABEL_COL = "label"

# Pretrained ViT model from Hugging Face
PRETRAINED_MODEL_NAME = "google/vit-base-patch16-224-in21k"
NUM_CLASSES = 2  # cat vs dog

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where we save the fine-tuned model
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, "cats_dogs_vit.pt")

# Label encoding
LABEL_TO_ID: Dict[str, int] = {"cat": 0, "dog": 1}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

# =========================
# Dataset definition
# =========================

class CatsDogsDataset(Dataset):
    """
    Simple Dataset that:
    - reads image names and labels from labels.csv
    - cleans labels (drop NaNs, normalize strings)
    - maps labels to ids
    - loads images and applies the ViT image processor
    """

    def __init__(self, csv_path: str, images_dir: str, processor: AutoImageProcessor) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Labels CSV not found at {csv_path}")

        self.df = pd.read_csv(csv_path)

        if IMAGE_COL not in self.df.columns or LABEL_COL not in self.df.columns:
            raise ValueError(
                f"Expected columns '{IMAGE_COL}' and '{LABEL_COL}' in {csv_path}. "
                f"Found columns: {list(self.df.columns)}"
            )

        # Keep only the columns we actually use
        self.df = self.df[[IMAGE_COL, LABEL_COL]].copy()

        # Drop rows with missing labels
        self.df = self.df.dropna(subset=[LABEL_COL])

        # Normalize labels: lowercase, strip spaces
        self.df[LABEL_COL] = self.df[LABEL_COL].astype(str).str.strip().str.lower()

        # Keep only 'cat' and 'dog'
        self.df = self.df[self.df[LABEL_COL].isin(LABEL_TO_ID.keys())]

        # Map labels to integer ids
        self.df["label_id"] = self.df[LABEL_COL].map(LABEL_TO_ID)

        # Make sure filenames are strings
        self.df["image_name"] = self.df[IMAGE_COL].astype(str)

        self.images_dir = images_dir
        self.processor = processor

        if len(self.df) == 0:
            raise ValueError(
                "No valid labeled samples after cleaning. "
                "Check LABEL_COL/IMAGE_COL and label values."
            )

        # Quick sanity check: label distribution
        label_array = self.df["label_id"].to_numpy()
        unique_labels, counts = np.unique(label_array, return_counts=True)
        print("Label distribution (id: count):", dict(zip(unique_labels, counts)))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        image_name = row["image_name"]
        image_path = os.path.join(self.images_dir, image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)  # (C, H, W)

        label_id = int(row["label_id"])
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": label_tensor}


def create_dataloaders(processor: AutoImageProcessor):
    """
    Create train/validation DataLoaders using an 80/20 split.
    """
    dataset = CatsDogsDataset(LABELS_CSV, IMAGES_DIR, processor)

    val_ratio = 0.2
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# =========================
# Training function
# =========================

def train_custom_model() -> None:
    """
    Fine-tune the pretrained ViT model on the local cats/dogs dataset.
    Saves the best model (by val accuracy) to CUSTOM_MODEL_PATH.
    """
    print("Loading ViT processor and creating DataLoaders...")
    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
    train_loader, val_loader = create_dataloaders(processor)

    print("Initializing ViT model for fine-tuning...")
    model = ViTForImageClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # ---- Training ----
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [train]",
            total=len(train_loader),
            leave=False,
        )

        for batch in train_pbar:
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = pixel_values.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += batch_size

            current_train_loss = running_loss / max(1, running_total)
            current_train_acc = running_correct / max(1, running_total)
            train_pbar.set_postfix(
                loss=f"{current_train_loss:.4f}",
                acc=f"{current_train_acc:.4f}",
            )

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [val]",
            total=len(val_loader),
            leave=False,
        )

        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch["pixel_values"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                batch_size = pixel_values.size(0)
                val_loss += loss.item() * batch_size
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

                current_val_loss = val_loss / max(1, val_total)
                current_val_acc = val_correct / max(1, val_total)
                val_pbar.set_postfix(
                    loss=f"{current_val_loss:.4f}",
                    acc=f"{current_val_acc:.4f}",
                )

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {avg_val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Save the best model so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CUSTOM_MODEL_PATH)
            print(f"New best model saved to {CUSTOM_MODEL_PATH}")

    duration = time.time() - start_time
    print(f"\nTraining completed in {duration:.1f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

# =========================
# Inference helpers
# =========================

# Global processor + model used for inference (API and web UI)
inference_processor: AutoImageProcessor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
inference_model: ViTForImageClassification = ViTForImageClassification.from_pretrained(
    PRETRAINED_MODEL_NAME,
    num_labels=NUM_CLASSES,
    id2label=ID_TO_LABEL,
    label2id=LABEL_TO_ID,
)

def load_finetuned_weights_if_available() -> None:
    """
    If we have a fine-tuned model on disk, load its weights into inference_model.
    Otherwise, we fall back to the base pretrained ViT.
    """
    global inference_model
    if os.path.exists(CUSTOM_MODEL_PATH):
        print(f"Loading fine-tuned weights from {CUSTOM_MODEL_PATH} for inference...")
        state_dict = torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE)
        inference_model.load_state_dict(state_dict)
    else:
        print("Fine-tuned model not found. Inference will use base pre-trained ViT weights.")

    inference_model.to(DEVICE)
    inference_model.eval()

load_finetuned_weights_if_available()

def prepare_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw image bytes into a ViT-ready tensor on DEVICE.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = inference_processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    return pixel_values

def predict_image(image_bytes: bytes) -> Dict[str, object]:
    """
    Run a forward pass on a single uploaded image and return label + confidence.
    """
    pixel_values = prepare_image(image_bytes)
    with torch.no_grad():
        outputs = inference_model(pixel_values=pixel_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        conf_values, pred_ids = torch.max(probs, dim=-1)
        pred_id = int(pred_ids[0].item())
        confidence = float(conf_values[0].item())

    label = ID_TO_LABEL.get(pred_id, str(pred_id))
    return {"label": label, "confidence": confidence}

# =========================
# FastAPI app
# =========================

app = FastAPI(title="Cats & Dogs Classifier")

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """
    Simple HTML upload form so you can test the model from the browser.
    """
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cats &amp; Dogs Classifier</title>
</head>
<body>
    <h1>Cats &amp; Dogs Classifier</h1>
    <p>Upload an image of a cat or dog to get a prediction.</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/png,image/jpeg" required>
        <br><br>
        <button type="submit">Classify</button>
    </form>
    <p>API documentation available at <a href="/docs">/docs</a>.</p>
</body>
</html>
"""
    return HTMLResponse(content=html)

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, object]:
    """
    POST endpoint that accepts an image file and returns JSON with
    predicted_label and confidence.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")

    image_bytes = await file.read()

    try:
        result = predict_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "predicted_label": result["label"],
        "confidence": result["confidence"],
    }

@app.get("/health")
async def health() -> Dict[str, str]:
    """
    Simple health-check endpoint.
    """
    return {"status": "ok"}

# =========================
# Entry point (CLI)
# =========================

def main() -> None:
    """
    CLI entry:
    - `--train` to fine-tune the model on the local dataset
    - no args: start the FastAPI server
    """
    parser = argparse.ArgumentParser(description="Cats & Dogs classifier app")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the ViT model on the local cats & dogs dataset.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the FastAPI server (when not training).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the FastAPI server (when not training).",
    )

    args = parser.parse_args()

    if args.train:
        train_custom_model()
    else:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
