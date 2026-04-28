#!/usr/bin/env python3
"""
Generate all masks with SamAutomaticMaskGenerator and save each mask as a separate PNG.

Example:
python generate_all_masks_separate.py \
    --input_image frame000002.jpg \
    --output_path output_masks/ \
    --checkpoint sam_vit_h_4b8939.pth \
    --model_type vit_h
"""

## make it output the 3 biggest masks in the picture
## or the main object in the picture
## also research how else can we generate masks
import argparse
import os
from datetime import datetime
import json

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def save_mask_png(mask_arr, out_path):
    """mask_arr: binary (0/255) uint8"""
    cv2.imwrite(out_path, mask_arr)

def main():
    p = argparse.ArgumentParser(description="Generate and save all SAM masks separately.")
    p.add_argument("--input_image", required=True, help="Path to input image")
    p.add_argument("--output_path", required=True, help="Output directory (will be created if needed)")
    p.add_argument("--checkpoint", default="sam_vit_h_4b8939.pth", help="SAM checkpoint path")
    p.add_argument("--model_type", default="vit_h", choices=["vit_h","vit_l","vit_b"], help="SAM model type")
    # optional params to tune automatic generator
    p.add_argument("--pred_iou_thresh", type=float, default=0.88, help="pred_iou_thresh for generator")
    p.add_argument("--stability_score_thresh", type=float, default=0.95, help="stability_score_thresh for generator")
    p.add_argument("--min_mask_region_area", type=int, default=100, help="min_mask_region_area to ignore tiny masks")
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # load image (BGR) and convert to RGB for SAM
    img_bgr = cv2.imread(args.input_image)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {args.input_image}")
    img_rgb = img_bgr[:, :, ::-1]

    # load SAM model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area
    )

    print("Generating masks...")
    masks = mask_generator.generate(img_rgb)  # list of dicts
    n_masks = len(masks)
    print(f"Found {n_masks} masks.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # optional: write a small JSON with metadata (bbox, area, iou, score) for each mask
    metadata = []
    for i, m in enumerate(masks):
        seg = m["segmentation"]  # boolean mask or ndarray
        # ensure boolean -> uint8 0/255
        if seg.dtype == bool:
            mask_uint8 = (seg.astype("uint8") * 255)
        else:
            mask_uint8 = (seg.astype("uint8") * 255)

        fname = f"{timestamp}_mask_{i:03d}.png"
        out_file = os.path.join(args.output_path, fname)
        save_mask_png(mask_uint8, out_file)

        # collect metadata
        meta = {
            "filename": fname,
            "bbox": m.get("bbox"),
            "area": int(m.get("area", int(np.sum(mask_uint8 > 0)))),
            "predicted_iou": float(m.get("predicted_iou", -1)) if "predicted_iou" in m else None,
            "stability_score": float(m.get("stability_score", -1)) if "stability_score" in m else None
        }
        metadata.append(meta)

    # save metadata.json
    meta_path = os.path.join(args.output_path, f"{timestamp}_masks_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {n_masks} masks to: {args.output_path}")
    print(f"Metadata saved to: {meta_path}")

if __name__ == "__main__":
    main()

