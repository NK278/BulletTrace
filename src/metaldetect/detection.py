import os
import numpy as np
from scipy.io import loadmat
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Additional imports for region properties and distance calculations
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

from pathlib import Path

# ----- Configuration -----
saved_dir       = '/Users/nishchal_mac/Desktop/FDTD_Bio/src/training_data_models'
sample_path     = '/Users/nishchal_mac/Desktop/FDTD_Bio/Chest_Data/BulletData_size01x01_sample09.mat'
eps_model_path  = os.path.join(saved_dir, 'rf_eps_model.pkl')
sig_model_path  = os.path.join(saved_dir, 'rf_sigma_model.pkl')
clf_model_path  = os.path.join(saved_dir, 'clf.pkl')  # or set to '/path/to/metal_clf.pkl'

eps_val       = 1.0
eps_tol       = 0.1
sig_thresh    = 1e5

# Output directory for visualization
output_dir    = './results'
# --------------------------

# Load regression models
reg_eps = joblib.load(eps_model_path)
reg_sig = joblib.load(sig_model_path)

# Load flattened train/test arrays for metal classifier
def load_train_data(dir):
    return (np.load(os.path.join(dir, 'Xtr_img.npy')), 
            np.load(os.path.join(dir, 'Xte_img.npy')), 
            np.load(os.path.join(dir, 'y_eps_tr_img.npy')), 
            np.load(os.path.join(dir, 'y_eps_te_img.npy')), 
            np.load(os.path.join(dir, 'y_sig_tr_img.npy')), 
            np.load(os.path.join(dir, 'y_sig_te_img.npy')))

train_dir, _ = os.path.split(eps_model_path)
train_dir     = os.path.join(train_dir, 'train_data')
Xtr_img, Xte_img, y_eps_tr_img, y_eps_te_img, y_sig_tr_img, y_sig_te_img = load_train_data(train_dir)

y_eps_train = np.load(os.path.join(train_dir, 'y_eps_train.npy'))

# Build metal labels from ground truth
y_metal_tr = ((y_sig_tr_img > sig_thresh) &
              (np.isclose(y_eps_tr_img, eps_val, atol=eps_tol))).astype(int)
y_metal_te = ((y_sig_te_img > sig_thresh) &
              (np.isclose(y_eps_te_img, eps_val, atol=eps_tol))).astype(int)

# Load or train metal classifier
if Path(clf_model_path).is_file():
    clf = joblib.load(clf_model_path)
else:
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=2)
    clf.fit(Xtr_img, y_metal_tr)
    metal_pred = clf.predict(Xte_img)
    print(classification_report(y_metal_te.ravel(), metal_pred.ravel(),
                                target_names=["non-metal", "metal"]))
    joblib.dump(clf,clf_model_path)
    if metal_pred.any():
        print("🔔 Metal detected in test set.")
    else:
        print("✅ No metal detected in test set.")

# Load new sample and pixel size
dat = loadmat(sample_path, squeeze_me=True, struct_as_record=False)
delt_x = float(dat.get('dx', 1.0))  # physical unit per pixel; adjust key if different
Ez = dat.get('Ez_refl')
if Ez is None:
    raise ValueError(f"Ez_refl not found in {sample_path}")

# Determine image dimensions from training shape _,H,W
dummy = y_eps_train
_, H, W = dummy.shape

# Flatten input for prediction
X_new = Ez.reshape(1, -1)

# Predict ε and σ maps
eps_flat = reg_eps.predict(X_new).ravel()
sig_flat = reg_sig.predict(X_new).ravel()
eps_map  = eps_flat.reshape(H, W)
sig_map  = sig_flat.reshape(H, W)

# Classify sample using metal classifier
metal_flat_pred = clf.predict(X_new).ravel()
metal_mask      = metal_flat_pred.reshape(H, W).astype(bool)
if metal_mask.any():
    print("🔔 Metal detected in this sample.")
else:
    print("✅ No metal detected in this sample.")

# Visualize and save results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].set_title("Predicted ε")
im0 = axes[0].imshow(eps_map, aspect='auto')
fig.colorbar(im0, ax=axes[0])

axes[1].set_title("Predicted σ")
im1 = axes[1].imshow(sig_map, aspect='auto')
fig.colorbar(im1, ax=axes[1])

axes[2].set_title("Metal mask")
axes[2].imshow(metal_mask, cmap='gray', aspect='auto')

plt.tight_layout()
os.makedirs(output_dir, exist_ok=True)
out_fig = os.path.join(output_dir, 'detection_results.png')
plt.savefig(out_fig)
print(f"Saved visualization to {out_fig}")

# ----- Metal region stats -----
total_metal_px = metal_mask.sum()
print(f"Total metal area: {total_metal_px} pixels")

labeled_metal = label(metal_mask, connectivity=2)
metal_props   = regionprops(labeled_metal)
for i, prop in enumerate(metal_props, 1):
    area_px      = prop.area
    eq_diam_px   = prop.equivalent_diameter
    minr, minc, maxr, maxc = prop.bbox
    width_px     = maxc - minc
    height_px    = maxr - minr
    print(f"\nBlob {i}:")
    print(f"  • Area:                {area_px} px")
    print(f"  • Equivalent diameter: {eq_diam_px:.1f} px")
    print(f"  • Bounding box size:   {width_px}×{height_px} px")

# ----- Skin region stats -----
eps_skin_val = 7.9753
sig_skin_val = 36.397
eps_skin_tol = 0.1
sig_skin_tol = 0.5

skin_mask_flat = (
    np.isclose(eps_flat, eps_skin_val, atol=eps_skin_tol) &
    np.isclose(sig_flat, sig_skin_val, atol=sig_skin_tol)
)
skin_mask = skin_mask_flat.reshape(H, W)

total_skin_px = skin_mask.sum()
print(f"Total skin area: {total_skin_px} pixels")

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
ax.set_title('Approximate Skin Regions & Centroids')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.show()

# ----- Distance between surfaces -----
# Distance from metal boundary to skin and vice versa

dist_to_skin  = distance_transform_edt(~skin_mask)
dist_to_metal = distance_transform_edt(~metal_mask)

metal_boundary = find_boundaries(metal_mask, mode='outer')
skin_boundary  = find_boundaries(skin_mask, mode='outer')

d_metal2skin_px = dist_to_skin[metal_boundary]
d_skin2metal_px = dist_to_metal[skin_boundary]

# Convert to physical units (adjust factor if needed)
delt_x=0.0005
d_metal2skin_phys = d_metal2skin_px * delt_x * 100  

d_metal2skin_min = d_metal2skin_phys.min()
d_metal2skin_avg = d_metal2skin_phys.mean()
d_metal2skin_max = d_metal2skin_phys.max()

print("Metal→Skin gap (phys):  min {:.4f}, avg {:.4f}, max {:.4f}".format(
    d_metal2skin_min, d_metal2skin_avg, d_metal2skin_max
))
