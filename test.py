import os  # provides utilities for file and directory handling
import pandas as pd  # used to load and inspect the CSV file
from PIL import Image, UnidentifiedImageError  # used to open images and detect corrupted ones

# =========================
# Configuration (match main.py)
# =========================

DATA_DIR = "dataset"  # folder where dataset is stored relative to this script
IMAGES_DIR = os.path.join(DATA_DIR, "images")  # folder containing all images
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")  # path to the labels CSV file

IMAGE_COL = "image_name"  # name of the column that holds the image file names
LABEL_COL = "label"  # name of the column that holds the labels (cat/dog)

# =========================
# Functions
# =========================

def check_dataset() -> None:  # main function to check images and labels
    if not os.path.exists(LABELS_CSV):  # check that labels.csv exists
        print(f"[ERROR] labels.csv not found at: {LABELS_CSV}")  # print error message
        return  # exit the function early

    if not os.path.isdir(IMAGES_DIR):  # check that images directory exists
        print(f"[ERROR] images directory not found at: {IMAGES_DIR}")  # print error message
        return  # exit the function early

    print(f"Using labels CSV: {LABELS_CSV}")  # show which CSV we are using
    print(f"Using images dir: {IMAGES_DIR}")  # show which images folder we are using

    df = pd.read_csv(LABELS_CSV)  # load the labels CSV into a DataFrame
    print(f"Total rows in CSV: {len(df)}")  # print total number of rows in CSV

    if IMAGE_COL not in df.columns:  # check if the image column exists
        print(f"[ERROR] Column '{IMAGE_COL}' not found in CSV. Columns are: {list(df.columns)}")  # print error
        return  # exit early

    if LABEL_COL not in df.columns:  # check if the label column exists
        print(f"[WARNING] Column '{LABEL_COL}' not found in CSV. Columns are: {list(df.columns)}")  # warn user

    bad_rows = []  # list to store indices of rows with missing/corrupted images
    missing_files = []  # list of image paths that are missing
    corrupt_files = []  # list of image paths that are corrupted/unreadable

    # iterate over each row in the CSV
    for idx, row in df.iterrows():  # enumerate through rows with index
        image_name = str(row[IMAGE_COL])  # get the image file name as a string
        image_path = os.path.join(IMAGES_DIR, image_name)  # build full path to the image file

        if not os.path.exists(image_path):  # if the file does not exist at that path
            print(f"[MISSING] {image_path}")  # log missing file
            bad_rows.append(idx)  # remember this row index as bad
            missing_files.append(image_path)  # remember missing file path
            continue  # skip to the next row

        try:
            # try opening the image file in a context manager
            with Image.open(image_path) as img:  # open the image file
                img.verify()  # verify that this is a valid, non-corrupt image
        except (UnidentifiedImageError, OSError) as e:  # handle corrupted/unreadable images
            print(f"[CORRUPT] {image_path} ({e})")  # log corrupted image
            bad_rows.append(idx)  # remember this row index as bad
            corrupt_files.append(image_path)  # remember corrupted file path
            continue  # move on to the next row

    # =========================
    # Summary
    # =========================

    print("\n=== SUMMARY ===")  # print summary header
    print(f"Total rows checked: {len(df)}")  # total number of rows in CSV
    print(f"Missing image files: {len(missing_files)}")  # how many files were missing
    print(f"Corrupted/unreadable images: {len(corrupt_files)}")  # how many were corrupted
    print(f"Total bad rows in CSV: {len(bad_rows)}")  # total number of problematic rows

    if bad_rows:  # if we found any bad rows
        print("\nExamples of bad entries:")  # print header for examples
        for i, idx in enumerate(bad_rows[:10]):  # show up to first 10 bad entries
            row = df.iloc[idx]  # get the row by index
            print(f"  CSV index {idx}: image_name={row[IMAGE_COL]!r}, label={row.get(LABEL_COL, 'N/A')!r}")  # print info
        print("\nIf you want to clean them, you can either:")  # print suggestions
        print("  - manually remove these rows from labels.csv, or")  # option 1
        print("  - write a cleaner script that drops these indices and rewrites labels.csv.")  # option 2
    else:
        print("No missing or corrupted images were detected. ✅")  # happy path message

# =========================
# Entry point
# =========================

if __name__ == "__main__":  # when running this file directly
    check_dataset()  # run the dataset check
