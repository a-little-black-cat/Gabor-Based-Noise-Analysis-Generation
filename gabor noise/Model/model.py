"""TensorFlow models for learning local texture patches."""

from __future__ import annotations

import os

# Avoid a TensorFlow oneDNN graph-remapping issue seen during discriminator gradients.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

tf.config.optimizer.set_experimental_options({"remapping": False})

keras = tf.keras
layers = tf.keras.layers


class PixelNorm(layers.Layer):
    """Normalize each pixel across channels without moving statistics."""

    def call(self, inputs):
        return inputs * tf.math.rsqrt(
            tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + 1e-8
        )


class TextureGAN:
    """DCGAN-style generator and discriminator for RGB texture patches."""

    def __init__(self, patch_size=64, channels=3, latent_dim=128, condition_dim=63):
        if patch_size < 16 or patch_size & (patch_size - 1):
            raise ValueError("patch_size must be a power of two and at least 16")
        if channels != 3:
            raise ValueError("TextureGAN currently expects RGB images")
        """TensorFlow models for learning local texture patches.

        The models do not know where the training image came from.  Pass RGB patches
        from the Gabor pipeline, or any other image source, to ``TextureGAN.train``.
        """
        self.patch_size = patch_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        start_size = self.patch_size // 16
        model = keras.Sequential(name="texture_generator")
        #model.add(layers.Input(shape=(self.latent_dim,)))

        combined_input = layers.Input(
            shape=(self.latent_dim + self.condition_dim,),
            name="noise_and_condition",
        )
        
        model.add(combined_input)
        
        model.add(layers.Dense(start_size * start_size * 256, use_bias=False))
        model.add(PixelNorm())
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Reshape((start_size, start_size, 256)))
        filters = 128
        current_size = start_size
        while current_size < self.patch_size:
            model.add(layers.UpSampling2D(2, interpolation="nearest"))
            model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False))
            model.add(PixelNorm())
            model.add(layers.LeakyReLU(alpha=0.2))
            filters = max(filters // 2, 32)
            current_size *= 2
        model.add(layers.Conv2D(self.channels, 3, padding="same", activation="tanh"))
        return model

    def build_discriminator(self, condition_dim=63):
        image_input = layers.Input(
            shape=(self.patch_size, self.patch_size, self.channels),
            name="image_input",
        )
        condition_input = layers.Input(
            shape=(condition_dim,),
            name="condition_input",
        )

        image_features = image_input
        filters = 64
        current_size = self.patch_size
        while current_size > 4:
            image_features = layers.Conv2D(
                filters,
                4,
                strides=2,
                padding="same",
            )(image_features)
            image_features = layers.LeakyReLU(alpha=0.2)(image_features)
            image_features = layers.Dropout(0.3)(image_features)
            filters = min(filters * 2, 512)
            current_size //= 2

        image_features = layers.Flatten()(image_features)

        condition_features = layers.Dense(64)(condition_input)
        condition_features = layers.LeakyReLU(alpha=0.2)(condition_features)
        condition_features = layers.Dense(64)(condition_features)

        combined_features = layers.Concatenate()(
            [image_features, condition_features]
        )
        output = layers.Dense(1, name="real_fake_logit")(combined_features)

        return keras.Model(
            inputs=[image_input, condition_input],
            outputs=output,
            name="conditional_texture_discriminator",
        )

GANNetwork = TextureGAN