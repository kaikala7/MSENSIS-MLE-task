import os  # for working with file and directory paths
import pandas as pd  # for loading and modifying the CSV file
from PIL import Image, UnidentifiedImageError  # for opening images and detecting corrupted ones

# =========================
# Configuration (match main.py / test.py)
# =========================

DATA_DIR = "dataset"  # folder where dataset is stored relative to this script
IMAGES_DIR = os.path.join(DATA_DIR, "images")  # folder containing all images
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")  # path to the labels CSV file

IMAGE_COL = "image_name"  # column name that holds image file names
LABEL_COL = "label"  # column name that holds labels (cat/dog)

# =========================
# Cleaning function
# =========================

def clean_labels() -> None:  # main function that cleans labels.csv
    if not os.path.exists(LABELS_CSV):  # check if labels.csv exists
        print(f"[ERROR] labels.csv not found at: {LABELS_CSV}")  # print error message
        return  # stop function if file is missing

    if not os.path.isdir(IMAGES_DIR):  # check if the images directory exists
        print(f"[ERROR] images directory not found at: {IMAGES_DIR}")  # print error message
        return  # stop function if folder is missing

    print(f"Using labels CSV: {LABELS_CSV}")  # show which CSV is used
    print(f"Using images dir: {IMAGES_DIR}")  # show which images folder is used

    df = pd.read_csv(LABELS_CSV)  # load labels.csv into a DataFrame
    print(f"Original number of rows: {len(df)}")  # print how many rows we start with

    if IMAGE_COL not in df.columns:  # ensure IMAGE_COL exists in the CSV
        print(f"[ERROR] Column '{IMAGE_COL}' not found in CSV. Columns are: {list(df.columns)}")  # print error
        return  # stop if wrong column name

    bad_indices = []  # list to store row indices that are bad (missing/corrupt images)

    # go through every row in the DataFrame
    for idx, row in df.iterrows():  # iterate with index and row
        image_name = str(row[IMAGE_COL])  # get file name as string
        image_path = os.path.join(IMAGES_DIR, image_name)  # build absolute path to image

        if not os.path.exists(image_path):  # if image file does not exist
            print(f"[MISSING] {image_path}")  # log missing image
            bad_indices.append(idx)  # mark this row as bad
            continue  # skip to the next row

        try:
            with Image.open(image_path) as img:  # try to open the image
                img.verify()  # verify the image is not corrupted
        except (UnidentifiedImageError, OSError) as e:  # catch corrupted/unreadable images
            print(f"[CORRUPT] {image_path} ({e})")  # log corrupted image
            bad_indices.append(idx)  # mark this row as bad
            continue  # skip to the next row

    print(f"\nTotal bad rows found: {len(bad_indices)}")  # number of rows to remove

    if not bad_indices:  # if there are no bad rows
        print("No missing or corrupted images detected. Nothing to clean. ✅")  # nothing to do
        return  # stop function

    df_clean = df.drop(index=bad_indices)  # drop all bad rows from the DataFrame

    backup_path = LABELS_CSV.replace(".csv", "_backup.csv")  # path where backup CSV will be saved
    df.to_csv(backup_path, index=False)  # save original CSV as backup
    print(f"Backup of original labels saved to: {backup_path}")  # log backup save

    df_clean.to_csv(LABELS_CSV, index=False)  # overwrite labels.csv with cleaned version
    print(f"Cleaned labels saved to: {LABELS_CSV}")  # log cleaned CSV save
    print(f"Rows after cleaning: {len(df_clean)}")  # print how many rows remain

# =========================
# Entry point
# =========================

if __name__ == "__main__":  # this runs only when file is executed directly
    clean_labels()  # call the cleaning function
