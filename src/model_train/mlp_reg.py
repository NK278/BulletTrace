import os
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from sklearn.model_selection import train_test_split
from loader.dataloader import load_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone

def train_mlp(
    train_dir:str,
    hidden_layer_sizes=(512, 256, 128),
    alpha=1e-4,
    random_state=42,
    max_iter=200
):
    """
    Train a sklearn MLPRegressor on your flattened Ez_refl → [eps, sigma].
    Saves two models: one for eps and one for sigma (or you can do a joint multi-output).
    """
    train_dir = train_dir
 # Load and stack data
    X_array, y_eps_array, y_sig_array = load_data(train_dir)

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
    
    # pipeline
    pipe=Pipeline([
        ('scaler',StandardScaler()),
        ('pca',PCA(n_components=0.95)),
        ('mlp',MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4,
                max_iter=max_iter,
                random_state=random_state,
                verbose=False
        ))
    ])
    # train mlp
    eps_pipe=clone(pipe).fit(Xtr_img, y_eps_tr_img)
    sig_pipe=clone(pipe).fit(Xtr_img, y_sig_tr_img)
    # mlp_eps = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter, verbose=True)
    # mlp_sig = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state, max_iter=max_iter, verbose=True)
    # mlp_eps.fit(Xtr_img, y_eps_tr_img)
    # mlp_sig.fit(Xtr_img, y_sig_tr_img)
    mlp_eps=eps_pipe
    mlp_sig=sig_pipe
    # Evaluate separately
    print("\nSeparate models:")
    for name, model, y_te, y_pred in [
        ('eps', mlp_eps, y_eps_te_img, mlp_eps.predict(Xte_img)),
        ('sig', mlp_sig, y_sig_te_img, mlp_sig.predict(Xte_img))
    ]:
        print(f"{name} Test R²:", r2_score(y_te, y_pred), 
              f"MSE: {mean_squared_error(y_te, y_pred):.3f}",
              f"MAE: {mean_absolute_error(y_te, y_pred):.3f}")

    # Save separate
    joblib.dump(mlp_eps, os.path.join('src/training_data_models', 'mlp_eps_model.pkl'))
    joblib.dump(mlp_sig, os.path.join('src/training_data_models', 'mlp_sigma_model.pkl'))
    print("Saved separate MLPs for eps & sigma.")

    return  mlp_eps, mlp_sig


    