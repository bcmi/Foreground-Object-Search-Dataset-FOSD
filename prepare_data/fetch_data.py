import sys,os
PROJ_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_dir)
import pickle
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import threading
from time import ctime
import argparse


def fetch_data(open_image_source_dir, split):
    if split == "train":
        target_bg_dir = os.path.join(PROJ_dir, f"data/{split}/bg")
    elif split == "test":
        target_bg_dir = os.path.join(PROJ_dir, f"data/{split}/bg_set1")

    target_fg_dir = os.path.join(PROJ_dir, f"data/{split}/fg")
    target_mask_dir = os.path.join(PROJ_dir, f"data/{split}/labels/masks")
    img_source_dir = os.path.join(open_image_source_dir, "train/data")
    mask_source_dir = os.path.join(open_image_source_dir, "train/labels/masks")

    seg_file_path = os.path.join(PROJ_dir, f"data/{split}/labels/segmentations.csv")
    selected_segmentations = pd.read_csv(seg_file_path)

    print("Segmentation file read.")

    for idx, row in tqdm(selected_segmentations.iterrows()):
        
        maskPath = row[1]
        xmin, xmax, ymin, ymax = row[5:9]
        imageId = maskPath.split(".")[0].split("_")[0]
        target_bg = os.path.join(img_source_dir, imageId + ".jpg")
        key = "" + imageId[0]
        if key.isalpha():
            key = key.upper()
        target_mask = os.path.join(mask_source_dir, key, maskPath)

        target_mask_subdir = os.path.join(target_mask_dir, key)
        if not os.path.exists(target_mask_subdir):
            os.mkdir(target_mask_subdir)

        target_bg_img = Image.open(target_bg)
        target_mask_img = Image.open(target_mask)
        target_mask_img = target_mask_img.resize(target_bg_img.size)
        target_mask_img.save(os.path.join(target_mask_subdir, maskPath))

        width, height = target_bg_img.size
        XMin = int(xmin * width)
        XMax = int(xmax * width)
        YMin = int(ymin * height)
        YMax = int(ymax * height)
        
        target_fg_img = Image.fromarray(255 - (np.array(target_mask_img).astype("uint8")) * 255).convert("RGB")
        
        
        target_fg_img.paste(target_bg_img, mask=target_mask_img)
        target_fg_img = target_fg_img.crop([XMin, YMin, XMax, YMax])
        target_fg_img.convert("RGB").save(os.path.join(target_fg_dir, maskPath))
        
        gray_hole = Image.fromarray(np.zeros(shape=(YMax - YMin, XMax - XMin, 3)).astype("uint8") + 128).convert("RGB")
        target_bg_img.paste(gray_hole, [XMin, YMin])
        target_bg_img.convert("RGB").save(os.path.join(target_bg_dir, maskPath))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--open_images_dir", type=str, help="Path to OpenImages v6 dataset.")
    args = parser.parse_args()
    fetch_data(args.open_images_dir, "train")
    fetch_data(args.open_images_dir, "test")
