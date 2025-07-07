import os
import numpy as np
from loader.dataloader import load_data
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
import math


def make_generator(H1, W1, H2, W2, f1=64, f2=128):
    """
    Generator: input image -> epsilon & sigma maps at resolution H2 x W2.
    """
    inp = layers.Input((H1, W1, 1), name='generator_input')
    x = layers.Conv2D(f1, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(f2, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Resizing(H2, W2, interpolation='bilinear')(x)
    eps = layers.Conv2D(1, 1, activation='linear', name='eps')(x)
    sig = layers.Conv2D(1, 1, activation='linear', name='sig')(x)
    return Model(inp, [eps, sig], name='Generator')


def make_discriminator(H_map, W_map):
    """
    PatchGAN discriminator operating at map resolution H_map x W_map.
    Resizes input image to match map dims, then classifies.
    """
    inp_img = layers.Input((None, None, 1), name='disc_img_input')
    inp_maps = layers.Input((H_map, W_map, 2), name='disc_maps_input')

    # resize image to map resolution
    x = layers.Resizing(H_map, W_map, interpolation='bilinear', name='disc_resize_img')(inp_img)
    x = layers.Concatenate()([x, inp_maps])

    # PatchGAN conv stack
    x = layers.Conv2D(64, 4, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 4, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, 4, strides=1, padding='same', activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    validity = layers.Conv2D(1, 4, strides=1, padding='same', activation='sigmoid', name='validity')(x)

    return Model([inp_img, inp_maps], validity, name='Discriminator')


def build_cgan(generator, discriminator, lr=1e-4):
    """
    Freeze discriminator; build adversarial model generator -> discriminator.
    """
    discriminator.trainable = False

    img_in = generator.input
    eps_out, sig_out = generator(img_in)
    fake_maps = layers.Concatenate(name='cgan_fake_maps')([eps_out, sig_out])
    validity = discriminator([img_in, fake_maps])

    cgan = Model(img_in, [validity, eps_out, sig_out], name='cGAN')
    cgan.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=['binary_crossentropy', 'mse', 'mse'],
        loss_weights=[1.0, 100.0, 100.0]
    )
    return cgan


def train_cgan(train_dir,
               epochs=100,
               batch_size=16,
               f1=64,
               f2=128,
               lr=1e-4):
    """
    Load data, split, instantiate generator, discriminator at map resolution,
    then train GAN with alternating updates.
    """
    # load and split
    X, y_eps, y_sig = load_data(train_dir)
    Xtr, Xval, eps_tr, eps_val, sig_tr, sig_val = train_test_split(
        X, y_eps, y_sig, test_size=0.2, random_state=42
    )

    # add channel dims
    Xtr = Xtr[..., None];   Xval = Xval[..., None]
    eps_tr = eps_tr[..., None];  sig_tr = sig_tr[..., None]

    # image dims and map dims
    H1, W1 = Xtr.shape[1:3]
    H2, W2 = eps_tr.shape[1:3]

    # instantiate models
    gen = make_generator(H1, W1, H2, W2, f1, f2)
    disc = make_discriminator(H2, W2)

    # compile discriminator
    disc.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # build and compile combined model
    cgan = build_cgan(gen, disc, lr)

    # training loop
    # ensure at least one step per epoch
    steps = math.ceil(Xtr.shape[0] / batch_size)
    for epoch in range(1, epochs+1):
        for _ in range(steps):
            idx = np.random.randint(0, Xtr.shape[0], batch_size)
            imgs = Xtr[idx]
            real_eps = eps_tr[idx]
            real_sig = sig_tr[idx]

            # generate fake maps
            eps_fake, sig_fake = gen.predict(imgs, verbose=0)
            fake_maps = np.concatenate([eps_fake, sig_fake], axis=-1)
            real_maps = np.concatenate([real_eps, real_sig], axis=-1)

            # labels for discriminator
            real_labels = np.ones((batch_size,) + disc.output_shape[1:])
            fake_labels = np.zeros((batch_size,) + disc.output_shape[1:])

            # train discriminator
            d_loss_real = disc.train_on_batch([imgs, real_maps], real_labels)
            d_loss_fake = disc.train_on_batch([imgs, fake_maps], fake_labels)

            # train generator via adversarial model
            g_loss = cgan.train_on_batch(
                imgs,
                [np.ones((batch_size,) + disc.output_shape[1:]), real_eps, real_sig]
            )

        # end epoch logging
        print(
            f"Epoch {epoch}/{epochs} "
            f"[D_real loss: {d_loss_real[0]:.4f}, acc: {d_loss_real[1]:.4f}] "
            f"[D_fake loss: {d_loss_fake[0]:.4f}] "
            f"[G loss: {g_loss[0]:.4f}]"
        )

    # save models
    os.makedirs('models', exist_ok=True)
    gen.save('models/generator.h5')
    disc.save('models/discriminator.h5')

    return gen, disc



