"""
Script for generating face-blurred images
It assumes:
1. Original ILSVRC images are in data/train and data/val.
2. Face annotations are in data/face_annotations_ILSVRC.json.
It will save blurred images to data/train_blurred and data/val_blurred.
"""
from glob import glob
import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter
from typing import Any
import sys
import numpy as np

from utils.utils import ensure_dir

# sys.path.append(".")
# from common import FACE_ANNOTATIONS_PATH, TRAIN_IMAGES_PATH, VAL_IMAGES_PATH

# face_annotations = {
#     x["url"]: x["bboxes"] for x in json.load(open(FACE_ANNOTATIONS_PATH))
# }


def blur_box(img_path, bboxs) -> Any:
    """
    Apply a variant of Gaussian blurring.
    See the appendix for detail.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f'Exception on file {img_path}.')
        print(e)
        return False
    blur_img_path = img_path.replace('clean_resize', 'clean_resize_blur')
    ensure_dir(os.path.dirname(blur_img_path))
    mask = Image.new(mode="L", size=img.size, color="white")
    max_diagonal = 0

    for bbox in bboxs: # face_annotations[url]:
        bbox = {
            "x0": bbox[0], # left
            "y0": bbox[1], # top
            "x1": bbox[2], # right
            "y1": bbox[3] # bottom
        }
        if bbox["x0"] >= bbox["x1"] or bbox["y0"] >= bbox["y1"]:
            continue
        diagonal = max(bbox["x1"] - bbox["x0"], bbox["y1"] - bbox["y0"])
        max_diagonal = max(max_diagonal, diagonal)
        bbox = [
            bbox["x0"] - 0.1 * diagonal,
            bbox["y0"] - 0.1 * diagonal,
            bbox["x1"] + 0.1 * diagonal,
            bbox["y1"] + 0.1 * diagonal,
        ]
        draw = ImageDraw.Draw(mask)
        draw.rectangle(bbox, fill="black")
    blurred_img = img.filter(ImageFilter.GaussianBlur(0.1 * max_diagonal))
    blurred_mask = mask.filter(ImageFilter.GaussianBlur(0.1 * max_diagonal))
    img = Image.composite(img, blurred_img, blurred_mask)
    img.save(blur_img_path)
    return True


# if __name__ == "__main__":
#     image_paths = glob(os.path.join(TRAIN_IMAGES_PATH, "n*/*.jpg")) + glob(
#         os.path.join(VAL_IMAGES_PATH, "n*/*.jpg")
#     )
#     print("%d images to process" % len(image_paths))
#
#     for path in tqdm(image_paths):
#         img = Image.open(path).convert("RGB")
#         url = os.path.sep.join(path.split(os.path.sep)[-3:])[:-4] + ".JPEG"
#         if url.startswith("val"):
#             url = "val/" + url.split(os.path.sep)[-1]
#         if url in face_annotations:  # face annotations available
#             img = blur(img, url)
#             target_path = path.replace("train/", "train_blurred/").replace(
#                 "val/", "val_blurred/"
#             )
#             dir, _ = os.path.split(target_path)
#             if not os.path.exists(dir):
#                 os.makedirs(dir)
#             img.save(target_path)