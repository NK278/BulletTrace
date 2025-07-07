import os
import numpy as np
from loader.dataloader import load_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_rf_models(
    data_dir: str,
    n_estimators: int = 100,
    random_state: int = 42,
    test_size: float = 0.2
) -> None:
    """
    Train RandomForestRegressor models to predict epsilon_r and sigma from Ez_refl.
    Saves trained models to disk as 'rf_eps_model.pkl' and 'rf_sigma_model.pkl'.

    Args:
        data_dir: Path to directory with .mat files.
        n_estimators: Number of trees in the forest.
        random_state: Random seed.
        test_size: Proportion of data to use as test set.
    """
    # Load and stack data
    X_array, y_eps_array, y_sig_array = load_data(data_dir)

    X_train, X_test, y_eps_train, y_eps_test, y_sig_train, y_sig_test = train_test_split(
    X_array,
    y_eps_array,
    y_sig_array,
    test_size=0.2,
    random_state=42,     
    shuffle=True
)

    print("X_train shape:", X_train.shape)
    print("X_test  shape:", X_test.shape)
    print("y_eps_train shape:", y_eps_train.shape)
    print("y_sig_train shape:", y_sig_train.shape)

    # 1) Flatten each image into a vector
    N_train, H1, W1 = X_train.shape
    N_test,  _,  _  = X_test.shape
    _,      H2, W2 = y_eps_train.shape

    Xtr_img = X_train.reshape(N_train,  H1 * W1)      # (N_train, H1*W1)
    Xte_img = X_test.reshape(N_test,    H1 * W1)      # (N_test,  H1*W1)

    y_eps_tr_img = y_eps_train.reshape(N_train, H2 * W2)  # (N_train, H2*W2)
    y_eps_te_img = y_eps_test.reshape(N_test,   H2 * W2)  # (N_test,  H2*W2)

    y_sig_tr_img = y_sig_train.reshape(N_train, H2 * W2)  # same shape
    y_sig_te_img = y_sig_test.reshape(N_test,   H2 * W2)
    # Save flattened arrays for detection.py
    d='/Users/nishchal_mac/Desktop/FDTD_Bio/src/training_data_models'
    save_dir = os.path.join(d, 'train_data')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_eps_train.npy'), y_eps_train)
    np.save(os.path.join(save_dir, 'y_sig_train.npy'), y_sig_train)
    np.save(os.path.join(save_dir, 'Xtr_img.npy'), Xtr_img)
    np.save(os.path.join(save_dir, 'Xte_img.npy'), Xte_img)
    np.save(os.path.join(save_dir, 'y_eps_tr_img.npy'), y_eps_tr_img)
    np.save(os.path.join(save_dir, 'y_eps_te_img.npy'), y_eps_te_img)
    np.save(os.path.join(save_dir, 'y_sig_tr_img.npy'), y_sig_tr_img)
    np.save(os.path.join(save_dir, 'y_sig_te_img.npy'), y_sig_te_img)
    print(f"Saved train/test arrays to '{save_dir}'")
        # Train model for epsilon_r
    rf_eps = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=2
        )
    rf_eps.fit(Xtr_img, y_eps_tr_img)
        # Save epsilon model
    eps_model_path = os.path.join(d, 'rf_eps_model.pkl')
    joblib.dump(rf_eps, eps_model_path)
    print(f"Saved epsilon_r model to '{eps_model_path}'")

        # Train model for sigma
    rf_sig = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            verbose=2
        )
    rf_sig.fit(Xtr_img, y_sig_tr_img)
        # Save sigma model
    sig_model_path = os.path.join(d, 'rf_sigma_model.pkl')
    joblib.dump(rf_sig, sig_model_path)
    print(f"Saved sigma model to '{sig_model_path}'")

        # Evaluate on test set
    eps_score = rf_eps.score(Xte_img, y_eps_te_img)
    sig_score = rf_sig.score(Xte_img, y_sig_te_img)
    print(f"Test R^2 for epsilon_r model: {eps_score:.3f}")
    print(f"Test R^2 for sigma model: {sig_score:.3f}")


