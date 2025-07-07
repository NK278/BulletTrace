import os
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

# ----- Configuration -----
saved_model_dir = '/Users/nishchal_mac/Desktop/FDTD_Bio/cnn_models'
train_data_dir  = '/Users/nishchal_mac/Desktop/FDTD_Bio/train_data'
saved_cnn       = os.path.join(saved_model_dir, 'fft_efficientnet.h5')

# FFT band selection parameters
f_low, f_high = 0, 4e12  # Hz band for features

# Time-step from FDTD parameters
dt = ( (3e8 / 60e9) / 10 ) / (20 * 3e8)  # simplified delx/(20*c)

# Detection thresholds
eps_val, eps_tol = 1.0, 0.1
sig_thresh = 1e5

# Paths
sample_path = '/Users/nishchal_mac/Desktop/FDTD_Bio/Chest_DataSet/FDTD_sim_data_mx45_my77.mat'
output_dir  = './results'
os.makedirs(output_dir, exist_ok=True)

# 1) Load scaler fitted on training FFT features
Xf_train = np.load(os.path.join(train_data_dir, 'X_fft_train.npy'))  # shape (N_train, M*5)
scaler   = StandardScaler().fit(Xf_train)

# 2) Load trained CNN model
model = load_model(saved_cnn, compile=False)

# 3) Load new sample Ez_refl
data    = loadmat(sample_path, squeeze_me=True, struct_as_record=False)
Ez_refl = data.get('Ez_refl')  # expected shape (M, P)
if Ez_refl is None:
    raise KeyError(f"Ez_refl not found in {sample_path}")
M, P = Ez_refl.shape

# 4) FFT magnitude and band extraction
Xf_full = np.abs(np.fft.rfft(Ez_refl[None,...], axis=2))  # (1, M, P//2+1)
freqs   = np.fft.rfftfreq(P, d=dt)
mask    = (freqs >= f_low) & (freqs <= f_high)
Xf_band = Xf_full[:, :, mask]  # shape (1, M, B)

# 5) Scale features
B = Xf_band.shape[2]
Xf_flat = Xf_band.reshape(1, -1)
Xf_scaled_flat = scaler.transform(Xf_flat)
Xf_scaled      = Xf_scaled_flat.reshape(1, M, B)

# 6) Prepare input for EfficientNet: replicate to 3 channels and resize to 32x32
Xc = np.repeat(Xf_scaled[..., None], 3, axis=-1)  # (1, M, B, 3)
Xc_resized = tf.image.resize(Xc, [32, 32])

# 7) Predict ε and σ maps
eps_pred, sig_pred = model.predict(Xc_resized)
eps_flat = eps_pred.ravel()
sig_flat = sig_pred.ravel()

# 8) Reshape predictions back to image-shaped maps
n_out = eps_flat.size
dim   = int(np.sqrt(n_out))
eps_map = eps_flat.reshape(dim, dim)
sig_map = sig_flat.reshape(dim, dim)

# 9) Build masks for metal and skin
total_metal = (sig_map > sig_thresh) & (np.isclose(eps_map, eps_val, atol=eps_tol))
eps_skin, sig_skin = 7.9753, 36.397
eps_tol_skin, sig_tol_skin = 0.1, 0.5
skin_mask = (
    np.isclose(eps_map, eps_skin, atol=eps_tol_skin) &
    np.isclose(sig_map, sig_skin, atol=sig_tol_skin)
)

# 10) Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(eps_map, cmap='viridis');    axes[0].set_title('Predicted ε_map')
axes[1].imshow(sig_map, cmap='inferno');    axes[1].set_title('Predicted σ_map')
axes[2].imshow(total_metal, cmap='gray'); axes[2].set_title('Metal mask')
plt.tight_layout()
outfile = os.path.join(output_dir, 'detection_fft_cnn.png')
plt.savefig(outfile)
print(f"Saved detection visualization to {outfile}")

# 11) Region properties and metal→skin gap calculations
labeled_m = label(total_metal)
props_m   = regionprops(labeled_m)
print(f"Total metal pixels: {total_metal.sum()}")
for i, prop in enumerate(props_m, 1):
    print(f"Metal blob {i}: area={prop.area} px")

labeled_s = label(skin_mask)
props_s   = regionprops(labeled_s)
print(f"Total skin pixels: {skin_mask.sum()}")
for i, prop in enumerate(props_s, 1):
    y0, x0 = prop.centroid
    print(f"Skin blob {i}: area={prop.area} px, centroid=({y0:.1f},{x0:.1f})")

dist_to_skin = distance_transform_edt(~skin_mask)
bound_m      = find_boundaries(total_metal)
dists_mm     = dist_to_skin[bound_m] * (delx * 1000)
print(f"Metal→Skin gap (mm): min={dists_mm.min():.2f}, avg={dists_mm.mean():.2f}, max={dists_mm.max():.2f}")
