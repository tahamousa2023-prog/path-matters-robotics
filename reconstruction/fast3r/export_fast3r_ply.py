import argparse, os, glob
import numpy as np
import torch
import open3d as o3d

from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.inference_multiview import inference
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.models.fast3r import Fast3R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_ply", required=True)
    ap.add_argument("--max_images", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--max_points", type=int, default=800000)
    ap.add_argument("--conf_percentile", type=float, default=85.0)  # keep top conf %
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imgs = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    imgs = imgs[::args.stride]
    if args.max_images > 0:
        imgs = imgs[:args.max_images]
    if len(imgs) < 2:
        raise RuntimeError(f"Need >=2 images, got {len(imgs)} in {args.image_dir}")

    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512").to(device)
    model.eval()
    lit = MultiViewDUSt3RLitModule.load_for_inference(model)
    lit.eval()

    images = load_images(imgs, size=args.image_size, verbose=False)
    output_dict, _ = inference(images, model, device, dtype=torch.float32, verbose=False, profiling=False)

    # Move tensors to CPU for easier handling
    for pred in output_dict["preds"]:
        for k, v in list(pred.items()):
            if isinstance(v, torch.Tensor):
                pred[k] = v.cpu()
    for view in output_dict["views"]:
        for k, v in list(view.items()):
            if isinstance(v, torch.Tensor):
                view[k] = v.cpu()

    # Align local points into global frame
    lit.align_local_pts3d_to_global(preds=output_dict["preds"], views=output_dict["views"], min_conf_thr_percentile=args.conf_percentile)

    all_pts, all_col = [], []
    for view, pred in zip(output_dict["views"], output_dict["preds"]):
        pts = pred["pts3d_local_aligned_to_global"][0]  # (H,W,3)
        conf = pred["conf_local"][0]                   # (H,W)
        if "valid_mask" in view:
            valid = view["valid_mask"][0].bool()
        else:
            valid = torch.ones_like(conf, dtype=torch.bool)

        # threshold by percentile on valid points
        c = conf[valid].reshape(-1)
        thr = torch.quantile(c, args.conf_percentile / 100.0) if c.numel() else 0.0
        keep = valid & (conf >= thr)

        pts = pts[keep].numpy()

        # colors from image (Fast3R uses img in [-1,1] in their eval code)
        img = view.get("img", None)
        if img is not None:
            img = img[0].permute(1, 2, 0)     # (H,W,3)
            img = ((img + 1.0) / 2.0).clamp(0, 1)
            col = (img[keep].numpy() * 255).astype(np.uint8)
        else:
            col = np.zeros((len(pts), 3), dtype=np.uint8)

        all_pts.append(pts)
        all_col.append(col)

    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    col = np.concatenate(all_col, axis=0) if all_col else np.zeros((0, 3), dtype=np.uint8)

    # Random cap
    if args.max_points > 0 and len(pts) > args.max_points:
        idx = np.random.choice(len(pts), args.max_points, replace=False)
        pts, col = pts[idx], col[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((col.astype(np.float64) / 255.0))
    os.makedirs(os.path.dirname(args.out_ply), exist_ok=True)
    o3d.io.write_point_cloud(args.out_ply, pcd)
    print(f"[Fast3R] wrote {args.out_ply}  points={len(pts)}")

if __name__ == "__main__":
    main()