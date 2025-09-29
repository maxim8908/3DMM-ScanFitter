# 3DMM-Scan-Fitter

Fit a PCA-based **3D Morphable Model (3DMM)** to a scanned face mesh using **3D landmarks** and **Chamfer distance**, with an optional **free-vertex refinement** stage.  
Built on **PyTorch** and **PyTorch3D**.

## Features
- Landmark-only initialization with Procrustes pose (R, t, s)
- Chamfer distance fitting between model and scan
- Optional free-vertex refinement with Laplacian & normal regularization
- Clean, scriptable pipeline + loss plotting

## Requirements
- Python 3.10 or 3.11
- CUDA driver compatible with **cu121** wheels (e.g., driver 12.4 is OK)
- **PyTorch 2.4.1** and **PyTorch3D 0.7.6**

Install PyTorch (cu121 build) first:
```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
# 3DMM-ScanFitter
