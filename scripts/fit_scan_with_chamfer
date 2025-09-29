"""
Fit a PCA 3DMM to a scan mesh using 3D landmarks and Chamfer distance.

Example:
python scripts/fit_scan_with_chamfer.py \
  --components pca/pca_components.npy \
  --pca_var   pca/pca_explained_variance.npy \
  --mean_obj  mean/mean_mesh.obj \
  --scan      scans/geo_scan.obj \
  --mesh_lm   data/mesh_landmarks.npy \
  --scan_lm   data/scan_landmarks.npy \
  --out       outputs/fitted.obj \
  --device    cuda:0
"""

import os, argparse, numpy as np, torch
from src.fitting.chamfer_3dmm_fitter import Chamfer3DMMFitter, FitConfig

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", required=True)
    ap.add_argument("--pca_var",   required=True)
    ap.add_argument("--mean_obj",  required=True)
    ap.add_argument("--scan",      required=True)
    ap.add_argument("--mesh_lm",   required=True)  # npy list of indices in 3DMM mean mesh
    ap.add_argument("--scan_lm",   required=True)  # npy list of indices in target scan
    ap.add_argument("--out",       required=True)
    ap.add_argument("--device",    default="cpu")
    # Optional tuners
    ap.add_argument("--iters_s1", type=int, default=500)
    ap.add_argument("--iters_s2", type=int, default=1000)
    ap.add_argument("--iters_s3", type=int, default=300)
    return ap.parse_args()

def main():
    args = parse_args()
    mesh_lm = np.load(args.mesh_lm).tolist()
    scan_lm = np.load(args.scan_lm).tolist()

    fitter = Chamfer3DMMFitter(
        target_mesh_path=args.scan,
        components_path=args.components,
        pca_var_path=args.pca_var,
        mean_obj_path=args.mean_obj,
        mesh_lm_idx=mesh_lm,
        scan_lm_idx=scan_lm,
        device=args.device
    )

    cfg = FitConfig(iters_s1=args.iters_s1, iters_s2=args.iters_s2, iters_s3=args.iters_s3)
    logs = fitter.fit(cfg)
    fitter.save_current_mesh(args.out)
    Chamfer3DMMFitter.plot_loss_curve(logs, cfg)

if __name__ == "__main__":
    main()
