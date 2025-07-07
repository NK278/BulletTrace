import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from typing import Tuple
from scipy.io import loadmat


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all .mat files from the given directory, extract Ez_refl, epsilon_r, and sigma,
    and stack them into 3D arrays.

    Args:
        data_dir: Path to the directory containing .mat files.

    Returns:
        A tuple of three numpy arrays:
            - X_array: stacked Ez_refl with shape (N, H, W)
            - y_eps_array: stacked epsilon_r with shape (N, H, W)
            - y_sig_array: stacked sigma with shape (N, H, W)
    """
    # find all files ending with '.mat'
    mat_files = [
        os.path.join(data_dir, fname)
        for fname in os.listdir(data_dir)
        if fname.endswith('.mat')
    ]

    X_list = []
    eps_list = []
    sig_list = []

    for matfile in mat_files:
        mat = loadmat(matfile, squeeze_me=True, struct_as_record=False)
        X_list.append(mat.get('Ez_refl'))
        eps_list.append(mat.get('epsilon_r'))
        sig_list.append(mat.get('sigma'))

    # Stack into 3D arrays of shape (N, H, W)
    X_array = np.stack(X_list, axis=0)
    y_eps_array = np.stack(eps_list, axis=0)
    y_sig_array = np.stack(sig_list, axis=0)
    print("X_array   shape:", X_array.shape)
    print("y_eps_array shape:", y_eps_array.shape)
    print("y_sig_array shape:", y_sig_array.shape) 

    return X_array, y_eps_array, y_sig_array

# ------------------ Configuration ------------------
data_dir = '/Users/nishchal_mac/Desktop/FDTD_Bio/data'
test_size = 0.2
random_state = 42
batch_size = 16
epochs = 50
patience = 5  # early stopping
# FFT band
f_low = 0
f_high = 4e12

# ------------------ Load & Prepare Data ------------------
X_array, y_eps_array, y_sig_array = load_data(data_dir)
# X_array: (N, M, P); y_*: (N, H, W)
N, M, P = X_array.shape
_, H, W = y_eps_array.shape

# Compute FDTD dt
c, f0 = 3e8, 60e9
lam = c / f0
delx = lam / 10
dt = delx / (20 * c)

# FFT magnitude and band selection
Xf = np.abs(np.fft.rfft(X_array, axis=2))  # (N, M, P//2+1)
freqs = np.fft.rfftfreq(P, d=dt)
mask = (freqs >= f_low) & (freqs <= f_high)
Xf_band = Xf[:, :, mask]  # (N, M, B)

# Scale features
N, M, B = Xf_band.shape
Xf_flat = Xf_band.reshape(N, -1)
scaler = StandardScaler().fit(Xf_flat)
Xf_scaled = scaler.transform(Xf_flat).reshape(N, M, B)

# Replicate channels to 3 for ImageNet backbone
a_inputs = np.stack([Xf_scaled]*3, axis=-1)  # shape (N, M, B, 3)

# Split
a_train, a_test, eps_tr, eps_te, sig_tr, sig_te = train_test_split(
    a_inputs, y_eps_array.reshape(N, -1), y_sig_array.reshape(N, -1),
    test_size=test_size, random_state=random_state, shuffle=True
)
# reshape targets
eps_tr = eps_tr.reshape(-1, H*W)
eps_te = eps_te.reshape(-1, H*W)
sig_tr = sig_tr.reshape(-1, H*W)
sig_te = sig_te.reshape(-1, H*W)

# ------------------ Build Model with EfficientNetB0 Backbone + Resize ------------------
# EfficientNet requires at least 32×32 input, so we resize our (M,B) FFT maps.
inputs = layers.Input(shape=(M, B, 3), name='fft_input')
# Upsample to 32×32
x = layers.Resizing(32, 32, interpolation='bilinear', name='resize')(inputs)
# Load pretrained backbone on 32×32 inputs
base_model = EfficientNetB0(
    include_top=False,
    input_shape=(32, 32, 3),
    weights='imagenet'
)
# Freeze backbone initially
base_model.trainable = False
x = base_model(x, training=False)
# Head
x = layers.GlobalAveragePooling2D(name='gap')(x)
x = layers.Dense(256, activation='relu', name='dense_head')(x)
x = layers.Dropout(0.3, name='dropout_head')(x)
# Two regression heads
eps_out = layers.Dense(H*W, name='eps')(x)
sig_out = layers.Dense(H*W, name='sig')(x)
# Build and compile
model = models.Model(inputs=inputs, outputs=[eps_out, sig_out], name='fft_efficientnet_model')
model.compile(optimizer='adam', loss='mse')
# ------------------ Training ------------------
es = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

history = model.fit(
    a_train, [eps_tr, sig_tr],
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[es]
)

# ------------------ Fine-tune Backbone ------------------
for layer in base_model.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='mse')
history_fine = model.fit(
    a_train, [eps_tr, sig_tr],
    validation_split=0.2,
    epochs=epochs//2,
    batch_size=batch_size,
    callbacks=[es]
)

# ------------------ Evaluate ------------------
results = model.evaluate(a_test, [eps_te, sig_te], verbose=2)
print(f"Test Loss (eps+sig): {results[0]:.4f}")

# ------------------ Save ------------------
os.makedirs('cnn_models', exist_ok=True)
model.save('cnn_models/fft_efficientnet.h5')
print("Saved model to 'cnn_models/fft_efficientnet.h5'")
