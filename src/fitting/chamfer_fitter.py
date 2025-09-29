import os, math
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import so3_exp_map, so3_log_map
import matplotlib.pyplot as plt


# --------------------------
# Config
# --------------------------
@dataclass
class FitConfig:
    # Stage 1: landmarks
    iters_s1: int = 300
    lr_lm: float = 1e-2

    # Stage 2: chamfer (coarse)
    iters_s2: int = 200
    lr_cd: float = 5e-3
    n_sample_points: int = 20000
    w_chamfer: float = 1.0
    w_lm_aux: float = 10.0

    # Stage 3: free-vertex refinement
    iters_s3: int = 300
    lr_v2s: float = 5e-4
    w_v2s: float = 1.0
    w_laplacian: float = 1e-4
    w_normal: float = 1e-2
    w_pca_prior: float = 1e-3
    pca_prior_sigma: float = 1.0

    # Pose
    use_scale: bool = True

    # Logging
    log_every: int = 50


# --------------------------
# Fitter
# --------------------------
class Chamfer3DMMFitter(nn.Module):
    """
    Fits a 3DMM to a target scan mesh with 3D landmarks (stage 1)
    and Chamfer distance (stage 2), then free-vertex refinement (stage 3).
    Optimizes: PCA shape coeffs (betas), pose (R,t,[s]), and optional per-vertex offsets.
    """

    def __init__(self, target_mesh_path, components_path, pca_var_path, mean_obj_path,
                 mesh_lm_idx, scan_lm_idx, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

        # ---- Load 3DMM (mean geometry comes from OBJ for faces) ----
        B = np.load(components_path).astype(np.float32)            # (K, 3N)
        pca_var = np.load(pca_var_path).astype(np.float32)         # (K,)
        pca_std = np.sqrt(pca_var)

        mean_verts, faces, _ = load_obj(mean_obj_path)             # mean verts (N,3), faces idx
        mean = mean_verts.numpy().astype(np.float32)               # (N,3)

        K, threeN = B.shape
        N = mean.shape[0]
        assert threeN == 3 * N, "PCA components shape must match mean vertices."

        # Store as (3N, K) for (3N,K) @ (K,) -> (3N,)
        self.register_buffer("basis", torch.from_numpy(B.T).to(self.device))      # (3N, K)
        self.register_buffer("pca_std", torch.from_numpy(pca_std).to(self.device))# (K,)
        self.register_buffer("mean", torch.from_numpy(mean).to(self.device))      # (N, 3)
        self.register_buffer("faces", faces.verts_idx.to(self.device))            # (F,3)
        self.N, self.K = N, K

        self.mesh_lm_idx = torch.as_tensor(mesh_lm_idx, dtype=torch.long, device=self.device)
        self.scan_lm_idx = torch.as_tensor(scan_lm_idx, dtype=torch.long, device=self.device)

        # ---- Target scan mesh ----
        self.target_mesh = load_objs_as_meshes([target_mesh_path], device=self.device)
        target_verts = self.target_mesh.verts_packed()
        self.target_lm = target_verts[self.scan_lm_idx]                                # (M,3)

        # ---- Parameters ----
        self.betas = nn.Parameter(torch.zeros(K, device=self.device))                  # 3DMM params
        self.offsets = nn.Parameter(torch.zeros(N, 3, device=self.device), requires_grad=False)  # free verts (stage 3)
        self.rvec = nn.Parameter(torch.zeros(3, device=self.device))                   # axis-angle
        self.tvec = nn.Parameter(torch.zeros(3, device=self.device))                   # translation
        self.log_scale = nn.Parameter(torch.zeros(1, device=self.device))              # log s

    # ---------- synthesis & transforms ----------
    def _synthesize_vertices(self):
        """
        verts_model = mean + reshape( basis @ (betas * std), (N,3) ) + offsets
        """
        beta_scaled = self.betas * self.pca_std                   # (K,)
        delta_flat = self.basis @ beta_scaled                     # (3N,)
        verts = self.mean + delta_flat.view(self.N, 3) + self.offsets
        return verts

    def _transform_vertices(self, verts):
        """
        Row-vector convention: world = s * (verts @ R) + t
        """
        R = so3_exp_map(self.rvec[None, :])[0]                    # (3,3)
        s = torch.exp(self.log_scale)[0]
        return (s * (verts @ R)) + self.tvec

    def _current_mesh(self):
        verts = self._transform_vertices(self._synthesize_vertices())
        return Meshes(verts=[verts], faces=[self.faces])

    # ---------- Procrustes init ----------
    @torch.no_grad()
    def _procrustes_alignment(self, verts):
        """
        Compute similarity transform that best aligns model landmarks to target landmarks.
        Uses SVD-based Procrustes with correct scale/rotation for row-vector convention.
        """
        X = verts[self.mesh_lm_idx]                               # (M,3)
        Y = self.target_lm                                        # (M,3)

        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)

        # Cross-covariance
        H = Xc.T @ Yc                                             # (3,3)
        U, S, Vh = torch.linalg.svd(H)
        R = Vh.T @ U.T                                            # R = V U^T

        # reflection fix
        if torch.linalg.det(R) < 0:
            Vh[-1, :] *= -1
            R = Vh.T @ U.T

        # scale (trace(S) / ||Xc||^2)
        s = (S.sum() / (Xc.pow(2).sum() + 1e-12)).item()

        # translation (row-vector)
        x_mean = X.mean(dim=0, keepdim=True)                      # (1,3)
        y_mean = Y.mean(dim=0, keepdim=True)
        t = (y_mean - s * (x_mean @ R)).squeeze(0)                # (3,)

        # write to parameters
        self.rvec.copy_(so3_log_map(R[None]).squeeze(0))
        self.log_scale.copy_(torch.tensor([math.log(max(s, 1e-8))], device=self.device))
        self.tvec.copy_(t)

        return {"R": R.detach().clone(), "t": t.detach().clone(), "s": float(s)}

    # ---------- losses ----------
    @staticmethod
    def _lm_loss(pred_lm, target_lm):
        return F.l1_loss(pred_lm, target_lm)

    # ---------- fit pipeline ----------
    def fit(self, cfg: FitConfig):
        loss_log = []

        # Stage 0: pose init with landmarks
        pred_verts0 = self._synthesize_vertices()
        self._procrustes_alignment(pred_verts0)

        # --- Stage 1: landmarks only (betas + pose) ---
        print("== Stage 1: Landmark fitting ==")
        params = [self.betas, self.rvec, self.tvec]
        if cfg.use_scale:
            params.append(self.log_scale)
        optim = torch.optim.Adam(params, lr=cfg.lr_lm)

        for it in range(cfg.iters_s1):
            optim.zero_grad(set_to_none=True)
            pred_verts = self._transform_vertices(self._synthesize_vertices())
            pred_lm = pred_verts[self.mesh_lm_idx]
            lm_loss = self._lm_loss(pred_lm, self.target_lm)
            lm_loss.backward()
            optim.step()

            loss_log.append({"stage": "s1", "iter": int(it), "lm": float(lm_loss.detach().cpu())})
            if it % cfg.log_every == 0 or it == cfg.iters_s1 - 1:
                print(f"[LM {it:04d}] lm={lm_loss.item():.6f}")

        # --- Stage 2: Chamfer distance (coarse) ---
        print("\n== Stage 2: Chamfer distance fitting ==")
        optim = torch.optim.Adam(params, lr=cfg.lr_cd)
        for it in range(cfg.iters_s2):
            optim.zero_grad(set_to_none=True)

            pred_verts = self._transform_vertices(self._synthesize_vertices())
            pred_lm = pred_verts[self.mesh_lm_idx]
            lm_loss = self._lm_loss(pred_lm, self.target_lm)

            src_mesh = self._current_mesh()
            src_pts = sample_points_from_meshes(src_mesh, num_samples=cfg.n_sample_points)   # (1,P,3)
            tgt_pts = sample_points_from_meshes(self.target_mesh, num_samples=cfg.n_sample_points)
            cd, _ = chamfer_distance(src_pts, tgt_pts)

            total = cfg.w_chamfer * cd + cfg.w_lm_aux * lm_loss
            total.backward()
            optim.step()

            loss_log.append({
                "stage": "s2", "iter": int(it),
                "chamfer": float(cd.detach().cpu()),
                "lm": float(lm_loss.detach().cpu()),
                "total": float(total.detach().cpu()),
            })
            if it % cfg.log_every == 0 or it == cfg.iters_s2 - 1:
                print(f"[S2 {it:04d}] total={total.item():.6f}  cd={cd.item():.6f}  lm={lm_loss.item():.6f}")

        # --- Stage 3: free-vertex refinement (optional) ---
        print("\n== Stage 3: Free-vertex refinement ==")
        for p in [self.betas, self.rvec, self.tvec, (self.log_scale if cfg.use_scale else None)]:
            if p is not None:
                p.requires_grad_(False)
        self.offsets.requires_grad_(True)

        optim = torch.optim.Adam([self.offsets], lr=cfg.lr_v2s)
        for it in range(cfg.iters_s3):
            optim.zero_grad(set_to_none=True)

            src_mesh = self._current_mesh()
            src_pts = sample_points_from_meshes(src_mesh, num_samples=cfg.n_sample_points)
            tgt_pts = sample_points_from_meshes(self.target_mesh, num_samples=cfg.n_sample_points)
            v2s, _ = chamfer_distance(src_pts, tgt_pts)

            model_space_mesh = Meshes(verts=[self._synthesize_vertices()], faces=[self.faces])
            lap = mesh_laplacian_smoothing(model_space_mesh)
            nrm = mesh_normal_consistency(model_space_mesh)
            prior = (self.betas / cfg.pca_prior_sigma).pow(2).mean()
            off_l2 = (self.offsets ** 2).mean()

            total = cfg.w_v2s * v2s + cfg.w_laplacian * lap + cfg.w_normal * nrm + cfg.w_pca_prior * prior + 1e-4 * off_l2
            total.backward()
            optim.step()

            loss_log.append({
                "stage": "s3", "iter": int(it),
                "v2s": float(v2s.detach().cpu()),
                "laplacian": float(lap.detach().cpu()),
                "nrm": float(nrm.detach().cpu()),
                "prior": float(prior.detach().cpu()),
                "total": float(total.detach().cpu()),
            })
            if it % cfg.log_every == 0 or it == cfg.iters_s3 - 1:
                print(f"[S3 {it:04d}] total={total.item():.6f}  v2s={v2s.item():.6f}  lap={lap.item():.6f}")

        return loss_log

    # ---------- I/O & plots ----------
    @torch.no_grad()
    def save_current_mesh(self, out_obj_path: str):
        mesh = self._current_mesh()
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
        save_obj(out_obj_path, verts, faces)
        print(f"[save] {out_obj_path}")

    @staticmethod
    def plot_loss_curve(loss_log, cfg: FitConfig):
        s1 = [e for e in loss_log if e["stage"] == "s1"]
        s2 = [e for e in loss_log if e["stage"] == "s2"]
        s3 = [e for e in loss_log if e["stage"] == "s3"]

        plt.figure(figsize=(11, 6))
        if s1:
            plt.plot([e["iter"] for e in s1], [e["lm"] for e in s1], label="S1: LM")
        if s2:
            off = cfg.iters_s1
            plt.plot([off + e["iter"] for e in s2], [e["chamfer"] for e in s2], label="S2: Chamfer")
            plt.plot([off + e["iter"] for e in s2], [e["lm"] for e in s2], label="S2: LM (aux)", alpha=0.6)
        if s3:
            off = cfg.iters_s1 + cfg.iters_s2
            plt.plot([off + e["iter"] for e in s3], [e["v2s"] for e in s3], label="S3: Chamfer (free verts)")

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("3DMM Scan Fitting Loss")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.show()
