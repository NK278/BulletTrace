import os
import numpy as np
from scipy.io import loadmat
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

# ----- Configuration -----
saved_model_dir  = '/Users/nishchal_mac/Desktop/FDTD_Bio/models'
train_data_dir   = '/Users/nishchal_mac/Desktop/FDTD_Bio/train_data'
eps_model_path   = os.path.join(saved_model_dir, 'rf_eps_model_new.pkl')
sig_model_path   = os.path.join(saved_model_dir, 'rf_sigma_model_new.pkl')

# Thresholds for metal detection
eps_val    = 1.0
eps_tol    = 0.1
sig_thresh = 1e5

# FFT parameters
f_low = 0  # Hz
f_high = 4e12  # Hz

# Sampling step dt (from FDTD):
c = 3e8       # m/s
f0 = 60e9     # Hz
lam = c / f0  # m
delx = lam / 10
dt = delx / (20 * c)

# New sample path
sample_path  = '/Users/nishchal_mac/Desktop/FDTD_Bio/Chest_DataSet/FDTD_sim_data_mx45_my77.mat'
output_dir   = './results'
os.makedirs(output_dir, exist_ok=True)

# Utility: FFT feature extractor

def extract_fft_features(X: np.ndarray, dt: float, f_low: float, f_high: float) -> np.ndarray:
    N, M, P = X.shape
    Xf = np.fft.rfft(X, axis=2)
    freqs = np.fft.rfftfreq(P, d=dt)
    mask = (freqs >= f_low) & (freqs <= f_high)
    fband = freqs[mask]
    feats = np.zeros((N, M * 5), dtype=float)
    for i in range(N):
        vec = []
        for m in range(M):
            mag = np.abs(Xf[i, m, mask])
            idx_peak = np.argmax(mag)
            vec.extend([
                fband[idx_peak],        # peak freq
                mag[idx_peak],           # peak mag
                np.sum(mag**2),          # energy
                (mag*fband).sum()/mag.sum(),  # centroid
                np.sqrt(((fband - (mag*fband).sum()/mag.sum())**2 * mag).sum()/mag.sum())  # bandwidth
            ])
        feats[i, :] = vec
    return feats

# 1) Load models
eps_reg = joblib.load(eps_model_path)
sig_reg = joblib.load(sig_model_path)

# 2) Load training FFT features to fit scaler
Xf_train = np.load(os.path.join(train_data_dir, 'X_fft_train.npy'))
scaler = StandardScaler().fit(Xf_train)

# 3) Load new sample Ez_refl from .mat
data = loadmat(sample_path, squeeze_me=True, struct_as_record=False)
Ez_refl = data.get('Ez_refl')  # expected shape (M, P)
if Ez_refl is None:
    raise KeyError(f"Ez_refl not found in {sample_path}")

# Determine output image size from regressor output dimension
# Here we infer H*W = eps_reg.n_outputs_
n_out = eps_reg.n_outputs_
# Assuming square images:
H = W = int(np.sqrt(n_out))

# 4) Extract FFT features for new sample
X_new = Ez_refl.reshape(1, Ez_refl.shape[0], Ez_refl.shape[1])
Xf_new = extract_fft_features(X_new, dt, f_low, f_high)
Xf_new_s = scaler.transform(Xf_new)

# 5) Predict permittivity and conductivity maps
eps_flat = eps_reg.predict(Xf_new_s).ravel()
sig_flat = sig_reg.predict(Xf_new_s).ravel()
eps_map = eps_flat.reshape(H, W)
sig_map = sig_flat.reshape(H, W)

# 6) Metal mask by threshold on predicted maps
metal_mask = (sig_map > sig_thresh) & (np.isclose(eps_map, eps_val, atol=eps_tol))
print('Metal detected:' if metal_mask.any() else 'No metal detected')

# 7) Skin mask by known tissue values
eps_skin = 7.9753
sig_skin = 36.397
eps_tol_skin = 0.1
sig_tol_skin = 0.5
skin_mask = (
    np.isclose(eps_map, eps_skin, atol=eps_tol_skin) &
    np.isclose(sig_map, sig_skin, atol=sig_tol_skin)
)

# 8) Visualization
fig, axarr = plt.subplots(1, 3, figsize=(15,5))
axarr[0].imshow(eps_map, cmap='viridis')
axarr[0].set_title('Predicted ε_map')
fig.colorbar(axarr[0].imshow(eps_map, cmap='viridis'), ax=axarr[0])

axarr[1].imshow(sig_map, cmap='inferno')
axarr[1].set_title('Predicted σ_map')
fig.colorbar(axarr[1].imshow(sig_map, cmap='inferno'), ax=axarr[1])

axarr[2].imshow(metal_mask, cmap='gray')
axarr[2].set_title('Metal mask')

plt.tight_layout()
outfig = os.path.join(output_dir, 'detection_results_fft.png')
plt.savefig(outfig)
print(f"Saved visualization to {outfig}")

# 9) Region properties and distance metrics
# Metal regions
labeled_m = label(metal_mask)
props_m = regionprops(labeled_m)
print(f"Total metal pixels: {metal_mask.sum()}")
for i, p in enumerate(props_m,1):
    print(f"Metal blob {i}: area={p.area}, eq_diameter={p.equivalent_diameter:.1f}")

# Skin regions
labeled_s = label(skin_mask)
props_s = regionprops(labeled_s)
print(f"Total skin pixels: {skin_mask.sum()}")
for i, p in enumerate(props_s,1):
    y0,x0 = p.centroid
    print(f"Skin blob {i}: area={p.area}, centroid=({y0:.1f},{x0:.1f})")

# Distance transforms
dist_to_skin  = distance_transform_edt(~skin_mask)
dist_to_metal = distance_transform_edt(~metal_mask)
bound_metal = find_boundaries(metal_mask)
bound_skin  = find_boundaries(skin_mask)
d_m2s = dist_to_skin[bound_metal] * (delx*1000)
print(f"Metal→Skin gap (mm): min {d_m2s.min():.2f}, avg {d_m2s.mean():.2f}, max {d_m2s.max():.2f}")
