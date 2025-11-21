# MSENSIS-MLE task – Cats vs Dogs Classifier (ViT + FastAPI)

This project implements a simple ML engineering pipeline for classifying images of cats and dogs using a pretrained Vision Transformer (ViT) from Hugging Face, trained and served with PyTorch and FastAPI.

The solution covers:

1. Download and use of a pretrained model from Hugging Face  
2. Loading, cleaning and preprocessing of the provided dataset with Pandas and NumPy  
3. Training a deep learning model using PyTorch  
4. Exposing FastAPI endpoints that accept an image and return the predicted class  
5. Providing a minimal Web UI to interact with the model from the browser  

A short 1–2 page report (`report.pdf`) with comments and a description of the approach is included in the same folder as the code.

---

## 1. Project structure

Inside the `MSENSIS-MLE task` (or `vitcatsanddogs`) folder:

- `main.py` – main training + inference + FastAPI app  
- `test.py` – script to check for missing/corrupt images  
- `clean_labels.py` – script to clean `labels.csv` from bad entries  
- `requirements.txt` – Python dependencies  
- `README.md` – this file  
- `report.pdf` – short report (1–2 pages)  
- `models/`  
  - `cats_dogs_vit.pt` – fine-tuned model weights (created after training)  
- `dataset/`  
  - `images/` – all cat/dog images (provided dataset)  
  - `labels.csv` – CSV with image names and labels  

Note: the file `models/cats_dogs_vit.pt` is created after running training once.

---

## 2. Dataset format

The code expects the dataset in this layout, relative to `main.py`:

- `dataset/images/` with files like `0.jpg`, `1.jpg`, …  
- `dataset/labels.csv`

The `labels.csv` file must contain at least the following columns:

- `image_name` – the image filename, e.g. `1234.jpg`  
- `label` – the class label: `Cat` or `Dog` (case-insensitive). Some entries may be missing.

During loading:

- Rows with missing labels (NaN/None) are dropped.  
- Labels are normalized to lowercase (`"Cat"` → `"cat"`, `"Dog"` → `"dog"`).  
- Only rows with `label` in `{ "cat", "dog" }` are kept.  
- Labels are mapped to integers:
  - `cat` → `0`  
  - `dog` → `1`  

If needed, missing or corrupted images can be detected with `test.py` and removed from `labels.csv` using `clean_labels.py`.

---

## 3. Installation

### 3.1. Clone the repository

- `git clone <your-repo-url>.git`  
- `cd "MSENSIS-MLE task"` (or `cd vitcatsanddogs`)

### 3.2. Create and activate a virtual environment

- Create: `python -m venv venv`  
- Activate on Windows PowerShell: `.env\Scripts\Activate.ps1`  
- Activate on Linux/macOS: `source venv/bin/activate`

### 3.3. Install dependencies

- `pip install -r requirements.txt`

This installs:

- `torch`, `torchvision`  
- `transformers`  
- `fastapi`, `uvicorn`  
- `pandas`, `numpy`  
- `Pillow`  
- `tqdm`  
- and other small utilities

---

## 4. (Optional) Check and clean the dataset

If you are not sure that all images are present and readable, you can:

### 4.1. Check for missing / corrupt images

Run:

- `python test.py`

This script will:

- Load `dataset/labels.csv`  
- Check that every `image_name` exists under `dataset/images/`  
- Try to open each image with PIL  
- Print a summary of missing or corrupted files  

### 4.2. Clean `labels.csv` automatically

Run:

- `python clean_labels.py`

This will:

- Drop all rows in `labels.csv` that point to missing or corrupt images  
- Save a backup as `dataset/labels_backup.csv`  
- Overwrite `dataset/labels.csv` with the cleaned version  

After cleaning, `python test.py` should report:

- `Missing image files: 0`  
- `Corrupted/unreadable images: 0`  
- `Total bad rows in CSV: 0`

---

## 5. Training the model

To fine-tune the ViT model on the cats/dogs dataset, run:

- `python main.py --train`

What happens:

- A pretrained ViT model is loaded from Hugging Face: `google/vit-base-patch16-224-in21k`.  
- The classification head is replaced with a 2-class head (`cat` / `dog`).  
- The dataset is loaded and cleaned with Pandas and NumPy.  
- A random 80/20 train/validation split is created using `torch.utils.data.random_split`.  
- A training loop runs for `EPOCHS` (currently set to 1 by default).  
- The model is optimized with AdamW (learning rate `5e-5`).  
- Training and validation loss/accuracy are shown with tqdm progress bars.  
- The best model (by validation accuracy) is saved to `models/cats_dogs_vit.pt`.

---

## 6. Running the API and Web UI

After training (or even before, if you just want to test with the base pretrained model), you can start the FastAPI app:

- `python main.py`

By default it starts on `http://0.0.0.0:8000`.

### 6.1. Web UI

Open a browser and go to:

- `http://localhost:8000/`

You will see a simple HTML page where you can:

- upload a JPG/PNG image of a cat or dog  
- click “Classify”  
- receive the predicted label and confidence score  

### 6.2. API endpoints

Interactive API docs are available at:

- `http://localhost:8000/docs`

Endpoints:

- `GET /health`  
  - Returns a simple JSON status: `{"status": "ok"}`  

- `POST /predict`  
  - Accepts an uploaded image file (`image/jpeg` or `image/png`)  
  - Returns JSON with:
    - `predicted_label` – `"cat"` or `"dog"`  
    - `confidence` – probability in `[0, 1]`  

When `main.py` starts, it tries to load `models/cats_dogs_vit.pt`.  
If the file is present, the API uses the fine-tuned model; otherwise it falls back to the base pretrained ViT weights.

---

## 7. Model details

- Base model: `google/vit-base-patch16-224-in21k` (Vision Transformer pretrained on ImageNet-21k)  
- Framework: PyTorch  
- Input preprocessing:
  - Uses `AutoImageProcessor` from Hugging Face for ViT  
  - Resizes and normalizes images to the format expected by the pretrained model  
- Loss: cross-entropy (handled internally by `ViTForImageClassification` when labels are passed)  
- Optimizer: AdamW  
- Metric: accuracy on both training and validation sets  

---

## 8. How to run everything from scratch

1. Clone the repo and `cd` into the `MSENSIS-MLE task` folder  
2. Create and activate a virtual environment  
3. Install dependencies with `pip install -r requirements.txt`  
4. Place the dataset under `dataset/` (`images/` + `labels.csv`)  
5. (Optional) Run `python test.py` and `python clean_labels.py` to clean the dataset  
6. Train the model with `python main.py --train`  
7. Start the API/Web UI with `python main.py`  
8. Open `http://localhost:8000/` and upload a cat/dog image  

---

## 9. Report

A short report (`report.pdf`) is included in this folder. It briefly describes:

- the choice of pretrained ViT model and why transfer learning is used  
- how the dataset is cleaned and preprocessed (Pandas + NumPy)  
- the training procedure and metrics  
- how the FastAPI endpoints and Web UI are used to serve the model  
- limitations and potential future improvements (more epochs, early stopping, pseudo-labeling, etc.)  
