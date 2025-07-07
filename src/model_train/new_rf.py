import os
import numpy as np
from loader.dataloader import load_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def extract_fft_features(X: np.ndarray, dt: float, f_low: float, f_high: float) -> np.ndarray:
    """
    Extracts spectral features from time-series data via FFT.

    Parameters
    ----------
    X      : array, shape (N, M, P)
             N samples, M channels (probes), P time-steps
    dt     : float
             Time step size in seconds
    f_low  : float
             Lower frequency bound (Hz)
    f_high : float
             Upper frequency bound (Hz)

    Returns
    -------
    feats  : array, shape (N, M*5)
             For each sample and channel, features: [f_peak, mag_peak, energy, centroid, bandwidth]
    """
    N, M, P = X.shape
    # FFT and frequency axis
    Xf    = np.fft.rfft(X, axis=2)
    freqs = np.fft.rfftfreq(P, d=dt)

    # Band mask
    mask = (freqs >= f_low) & (freqs <= f_high)
    fband = freqs[mask]

    feats = np.zeros((N, M * 5), dtype=float)
    for i in range(N):
        vec = []
        for m in range(M):
            mag = np.abs(Xf[i, m, mask])
            # 1) peak frequency & magnitude
            idx_peak = np.argmax(mag)
            f_peak   = fband[idx_peak]
            mag_peak = mag[idx_peak]
            # 2) total energy in band
            energy   = np.sum(mag**2)
            # 3) spectral centroid
            centroid = (mag * fband).sum() / mag.sum()
            # 4) spectral bandwidth (RMS width)
            bw       = np.sqrt(((fband - centroid)**2 * mag).sum() / mag.sum())
            vec.extend([f_peak, mag_peak, energy, centroid, bw])
        feats[i, :] = vec
    return feats


def train_rf_models(
    data_dir: str,
    n_estimators: int = 100,
    random_state: int = 42,
    test_size: float = 0.2,
    f_low: float = 1e9,
    f_high: float = 3e9
) -> None:
    """
    Train RandomForestRegressor models on FFT-derived features to predict epsilon_r and sigma.

    Saves:
      - 'X_fft_train.npy', 'X_fft_test.npy' in './train_data'
      - 'rf_eps_model.pkl', 'rf_sigma_model.pkl' in './models'
    """
    # Load raw data
    X_array, y_eps_array, y_sig_array = load_data(data_dir)

    # Train/test split on raw time-series
    X_tr, X_te, y_eps_tr, y_eps_te, y_sig_tr, y_sig_te = train_test_split(
        X_array, y_eps_array, y_sig_array,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    print(f"X_tr shape: {X_tr.shape}   X_te shape: {X_te.shape}")
    print(f"y_eps_tr shape: {y_eps_tr.shape}   y_sig_tr shape: {y_sig_tr.shape}\n")

    # Compute dt from FDTD parameters
    c      = 3e8      # m/s
    f0     = 60e9     # Hz
    lam    = c / f0   # wavelength
    delx   = lam / 10
    dt     = delx / (20 * c)

    # Extract FFT-based features
    Xf_tr = extract_fft_features(X_tr, dt, f_low, f_high)
    Xf_te = extract_fft_features(X_te, dt, f_low, f_high)
    print(f"Xf_tr shape: {Xf_tr.shape}   Xf_te shape: {Xf_te.shape}\n")

    # Flatten target images for multi-output regression
    n_tr = y_eps_tr.shape[0]
    n_te = y_eps_te.shape[0]
    y_eps_tr_flat = y_eps_tr.reshape(n_tr, -1)
    y_eps_te_flat = y_eps_te.reshape(n_te, -1)
    y_sig_tr_flat = y_sig_tr.reshape(n_tr, -1)
    y_sig_te_flat = y_sig_te.reshape(n_te, -1)

    # Save feature arrays
    save_dir = os.path.join(os.getcwd(), 'train_data')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_fft_train.npy'), Xf_tr)
    np.save(os.path.join(save_dir, 'X_fft_test.npy'),  Xf_te)
    print(f"Saved FFT feature arrays to '{save_dir}'\n")

    # Scale features
    scaler = StandardScaler().fit(Xf_tr)
    Xf_tr_s = scaler.transform(Xf_tr)
    Xf_te_s = scaler.transform(Xf_te)
    y_sig_tr_log = np.log1p(y_sig_tr_flat)
    y_sig_te_log = np.log1p(y_sig_te_flat)

    # Ensure models directory
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Train epsilon_r model (multi-output)
    rf_eps = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=2
    )
    rf_eps.fit(Xf_tr_s, y_eps_tr_flat)
    eps_path = os.path.join(model_dir, 'rf_eps_model_new.pkl')
    joblib.dump(rf_eps, eps_path)
    print(f"Saved epsilon_r model to '{eps_path}'")

    # Train sigma model (multi-output)
    rf_sig = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        verbose=2
    )
    rf_sig.fit(Xf_tr_s, y_sig_tr_flat)
    sig_path = os.path.join(model_dir, 'rf_sigma_model_new.pkl')
    joblib.dump(rf_sig, sig_path)
    print(f"Saved sigma model to '{sig_path}'\n")

    # Evaluate on test set
    eps_score = rf_eps.score(Xf_te_s, y_eps_te_flat)
    sig_score = rf_sig.score(Xf_te_s, y_sig_te_flat)
    print(f"Test R^2 for epsilon_r: {eps_score:.3f}")
    print(f"Test R^2 for sigma:       {sig_score:.3f}\n")