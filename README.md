# 3DMM-Scan-Fitter

Fit a PCA-based **3D Morphable Model (3DMM)** to a scanned face mesh using **3D landmarks** and **Chamfer distance**, with an optional **free-vertex refinement** stage.  
Built on **PyTorch** and **PyTorch3D**.

---

## Features
- Landmark-only initialization with Procrustes pose estimation (R, t, s)  
- Chamfer distance optimization between model and scan  
- Free-vertex refinement with Laplacian & normal regularization  
- Clean, scriptable pipeline and easy visualization of loss curves

---

## Fit Stages
### **Stage 1 — Landmark Fitting:**
Optimizes PCA coefficients and pose to match sparse 3D landmarks.

### **Stage 2 — Chamfer Fitting:**
Minimizes Chamfer distance between model and scan point clouds for dense alignment.

### **Stage 3 — Free-Vertex Refinement (Optional):**
Unlocks per-vertex offsets with Laplacian and normal regularization for fine detail fitting.

---
## Quick Start
- pca_components.npy — PCA components (K, 3N)
- pca_explained_variance.npy — Variance per component (K,)
- mean_mesh.obj — Mean mesh with face topology from 3DMM
- scan.obj — Target scanned mesh (Point Cloud)
- mesh_landmarks.npy — Landmark indices in the mean mesh
- scan_landmarks.npy — Landmark indices in the scan (mapped from mesh_landmarks)

## Requirements
- Python 3.10 or 3.11  
- CUDA driver compatible with **cu121** wheels (driver 12.4+ works fine)  
- **PyTorch 2.4.1** and **PyTorch3D 0.7.6**

**Install PyTorch (CUDA 12.1 build) first:**
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install pytorch3d==0.7.6
pip install -r requirements.txt
