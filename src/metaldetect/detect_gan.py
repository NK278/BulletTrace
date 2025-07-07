import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# region‐props & distance imports
from skimage.measure       import label, regionprops
from scipy.ndimage         import distance_transform_edt
from skimage.segmentation  import find_boundaries

# ----- Configuration -----
# Point this to your trained GAN generator
generator_path = '/Users/nishchal_mac/Desktop/FDTD_Bio/models/generator.h5'

# Thresholds for detection
eps_val    = 1.0
eps_tol    = 0.1
sig_thresh = 1e5

# Skin material params
eps_skin_val = 7.9753
sig_skin_val = 36.397
eps_skin_tol = 0.1
sig_skin_tol = 0.5

# Paths for data and results
data_path   = '/Users/nishchal_mac/Desktop/FDTD_Bio/data/FDTD_case_0009_mx40_my9.mat'
output_dir  = './results'
# --------------------------

# 1) Load GAN generator
generator = load_model(generator_path)
print(f"Loaded GAN generator from {generator_path}")

# 2) Load new sample
dat = loadmat(data_path, squeeze_me=True, struct_as_record=False)
Ez = dat.get('Ez_refl')
if Ez is None:
    raise ValueError(f"Ez_refl not found in {data_path}")

# 3) Prepare for generator: (H1, W1) -> (1, H1, W1, 1)
X_new = Ez[np.newaxis, ..., np.newaxis]

# 4) Predict ε and σ maps
#    generator.predict returns a list: [eps_batch, sig_batch]
eps_pred_batch, sig_pred_batch = generator.predict(X_new, verbose=0)
eps_map = eps_pred_batch[0, ..., 0]   # shape (H2, W2)
sig_map = sig_pred_batch[0, ..., 0]

# 5) Metal mask via thresholding
metal_mask = (sig_map > sig_thresh) & np.isclose(eps_map, eps_val, atol=eps_tol)
print("🔔 Metal detected in this sample." if metal_mask.any()
      else "✅ No metal detected in this sample.")

# 6) Visualize & save
os.makedirs(output_dir, exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15,5))

axes[0].set_title("Predicted ε")
im0 = axes[0].imshow(eps_map, aspect='auto')
fig.colorbar(im0, ax=axes[0])

axes[1].set_title("Predicted σ")
im1 = axes[1].imshow(sig_map, aspect='auto')
fig.colorbar(im1, ax=axes[1])

axes[2].set_title("Metal mask")
axes[2].imshow(metal_mask, cmap='gray', aspect='auto')

plt.tight_layout()
out_fig = os.path.join(output_dir, 'gan_detection_results.png')
plt.savefig(out_fig)
print(f"Saved visualization to {out_fig}")

# ----- Metal region stats -----
total_metal_px = metal_mask.sum()
print(f"Total metal area: {total_metal_px} pixels")

labeled_metal = label(metal_mask, connectivity=2)
for i, prop in enumerate(regionprops(labeled_metal), 1):
    area_px      = prop.area
    eq_diam_px   = prop.equivalent_diameter
    minr, minc, maxr, maxc = prop.bbox
    width_px     = maxc - minc
    height_px    = maxr - minr
    print(f"\nMetal blob {i}:")
    print(f"  • Area:                {area_px} px")
    print(f"  • Equivalent diameter: {eq_diam_px:.1f} px")
    print(f"  • Bounding box size:   {width_px}×{height_px} px")

# ----- Skin region stats -----
skin_mask = (
    np.isclose(eps_map, eps_skin_val, atol=eps_skin_tol) &
    np.isclose(sig_map, sig_skin_val, atol=sig_skin_tol)
)
total_skin_px = skin_mask.sum()
print(f"\nTotal skin area: {total_skin_px} pixels")

labeled_skin = label(skin_mask, connectivity=2)
skin_props   = regionprops(labeled_skin)
for i, prop in enumerate(skin_props, 1):
    area_px      = prop.area
    eq_diam_px   = prop.equivalent_diameter
    minr, minc, maxr, maxc = prop.bbox
    width_px     = maxc - minc
    height_px    = maxr - minr
    cy, cx       = prop.centroid
    print(f"\nSkin blob {i}:")
    print(f"  • Area:                {area_px} px")
    print(f"  • Equivalent diameter: {eq_diam_px:.1f} px")
    print(f"  • Bounding box size:   {width_px}×{height_px} px")
    print(f"  • Centroid:            ({cy:.1f}, {cx:.1f}) px")

# Visualize skin centroids
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(skin_mask, cmap='gray', origin='lower')
for prop in skin_props:
    y0, x0 = prop.centroid
    ax.plot(x0, y0, 'bo', markersize=4)
ax.set(title='Skin Regions & Centroids', xlabel='X (px)', ylabel='Y (px)')
plt.show()

# ----- Distance between surfaces -----
dist_to_skin  = distance_transform_edt(~skin_mask)
dist_to_metal = distance_transform_edt(~metal_mask)
metal_boundary = find_boundaries(metal_mask, mode='outer')
skin_boundary  = find_boundaries(skin_mask,  mode='outer')

d_metal2skin_px = dist_to_skin[metal_boundary]
delt_x = float(dat.get('dx', 1.0))
d_metal2skin_phys = d_metal2skin_px * delt_x

print("\nMetal→Skin gap (physical units):")
print(f"  • Min: {d_metal2skin_phys.min():.4f}")
print(f"  • Avg: {d_metal2skin_phys.mean():.4f}")
print(f"  • Max: {d_metal2skin_phys.max():.4f}")
