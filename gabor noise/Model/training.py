"""Train the texture GAN from a NumPy image, image file, or patch array."""

from pathlib import Path
import os

import numpy as np

# Avoid a TensorFlow oneDNN graph-remapping issue during discriminator gradients.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from PIL import Image

from model import TextureGAN


def load_rgb_image(image):
    if isinstance(image, (str, Path)):
        image = np.asarray(Image.open(image).convert("RGB"))
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def extract_patches(image, patch_size=64, stride=32):
    image = load_rgb_image(image)
    height, width, _ = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError("image must be at least as large as patch_size")
    patches = []
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patches.append(image[top:top + patch_size, left:left + patch_size])
    return np.asarray(patches, dtype=np.float32) * 2.0 - 1.0


def extract_random_patches(image, patch_size=64, count=256, seed=42):
    """Extract independently sampled crops and random mirror augmentations."""
    image = load_rgb_image(image)
    height, width, _ = image.shape
    if height < patch_size or width < patch_size:
        raise ValueError("image must be at least as large as patch_size")
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = np.random.default_rng(seed)
    patches = []
    for _ in range(count):
        top = int(rng.integers(height - patch_size + 1))
        left = int(rng.integers(width - patch_size + 1))
        patch = image[top:top + patch_size, left:left + patch_size].copy()
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        if rng.random() < 0.5:
            patch = patch[::-1]
        patches.append(patch)
    return np.asarray(patches, dtype=np.float32) * 2.0 - 1.0


def parameters_to_vector(parameters, condition_dim=63):
    """Convert extracted Gabor parameters into a fixed-size float vector."""
    values = [
        parameters["frequency"],
        parameters["theta"],
        parameters["sigma_x"],
        parameters["sigma_y"],
        parameters["noise_amplitude"],
        *parameters["color_mean"],
        *parameters["color_std"],
        parameters["luminance_mean"],
        parameters["luminance_std"],
        parameters["specular_mean"],
        parameters["specular_std"],
    ]
    for component in parameters.get("components", [])[:8]:
        values.extend([
            component["frequency"],
            component["theta"],
            component["sigma_x"],
            component["sigma_y"],
            component["energy"],
            component["response_std"],
        ])
    if len(values) > condition_dim:
        raise ValueError("Gabor parameter vector exceeds condition_dim")
    values.extend([0.0] * (condition_dim - len(values)))
    return np.asarray(values, dtype=np.float32)


def create_samples_gif(
    sample_directory,
    output_path,
    duration=500,
    columns=4,
):
    """Create a GIF whose frames are grids of samples from each epoch folder."""
    sample_directory = Path(sample_directory)
    output_path = Path(output_path)
    epoch_directories = sorted(
        path for path in sample_directory.glob("epoch_*") if path.is_dir()
    )
    if not epoch_directories:
        raise ValueError(f"No epoch sample folders found in {sample_directory}")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    frames = []
    for epoch_directory in epoch_directories:
        sample_paths = sorted(epoch_directory.glob("sample_*.png"))
        if not sample_paths:
            continue
        samples = [Image.open(path).convert("RGB") for path in sample_paths]
        sample_width, sample_height = samples[0].size
        rows = (len(samples) + columns - 1) // columns
        frame = Image.new("RGB", (columns * sample_width, rows * sample_height))
        for index, sample in enumerate(samples):
            frame.paste(
                sample,
                ((index % columns) * sample_width, (index // columns) * sample_height),
            )
            sample.close()
        frames.append(frame)

    if not frames:
        raise ValueError(f"No PNG samples found in {sample_directory}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    for frame in frames:
        frame.close()

class PatchGAN:
    def __init__(
        self,
        source_image=None,
        gabor_parameters=None,
        condition_vector=None,
        patch_size=64,
        latent_dim=128,
        learning_rate=2e-4,
    ):
        self.source_image = source_image
        self.gabor_parameters = gabor_parameters
        self.condition_dim = 63
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.condition_vector = self._resolve_condition(condition_vector)
        self.gan = TextureGAN(
            patch_size=patch_size,
            latent_dim=latent_dim,
            condition_dim=self.condition_dim,
        )
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def _resolve_condition(self, condition_vector=None):
        if condition_vector is None and self.gabor_parameters is not None:
            condition_vector = parameters_to_vector(
                self.gabor_parameters,
                self.condition_dim,
            )
        if condition_vector is None:
            return np.zeros(self.condition_dim, dtype=np.float32)
        condition_vector = np.asarray(condition_vector, dtype=np.float32).reshape(-1)
        if condition_vector.shape != (self.condition_dim,):
            raise ValueError("condition_vector must have shape (63,)")
        return condition_vector

    def train(
        self,
        image=None,
        gabor_parameters=None,
        condition_vector=None,
        learning_rate=None,
        epochs=500,
        batch_size=32,
        stride=32,
        seed=42,
        sample_directory=None,
        sample_interval=10,
        sample_count=8,
        checkpoint_directory=None,
        resume=False,
        steps_per_epoch=None,
    ):
        """Train the GAN and optionally save fixed-noise samples during training.

        Samples are saved in ``sample_directory/epoch_NNNN``. The final epoch
        is also saved when it does not fall exactly on ``sample_interval``.
        """
        print("Calling train()...")
        image = self.source_image if image is None else image
        if image is None:
            raise ValueError("Provide a source image or set source_image before training")
        if learning_rate is not None:
            if learning_rate <= 0.0:
                raise ValueError("learning_rate must be positive")
            self.generator_optimizer.learning_rate.assign(learning_rate)
            self.discriminator_optimizer.learning_rate.assign(learning_rate)
        if sample_directory is not None and sample_interval < 1:
            raise ValueError("sample_interval must be at least 1")
        if sample_count < 1:
            raise ValueError("sample_count must be at least 1")
        if steps_per_epoch is not None and steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be at least 1")
        print("Initializing training...")

        
        image_array = load_rgb_image(image)
        crop_count = max(
            256,
            int(np.ceil(
                image_array.shape[0] * image_array.shape[1] / self.patch_size**2
            )),
        )
        if steps_per_epoch is not None:
            crop_count = min(crop_count, steps_per_epoch * batch_size)
        patches = extract_random_patches(image_array, self.patch_size, crop_count, seed)
        condition = self._resolve_condition(
            condition_vector if condition_vector is not None else (
                parameters_to_vector(gabor_parameters, self.condition_dim)
                if gabor_parameters is not None else self.condition_vector
            )
        )
        conditions = np.repeat(condition[None, :], len(patches), axis=0)
        dataset = tf.data.Dataset.from_tensor_slices((patches, conditions)).shuffle(
            len(patches), seed=seed, reshuffle_each_iteration=True
        ).batch(batch_size)
        history = {"generator_loss": [], "discriminator_loss": []}
        sample_directory = Path(sample_directory) if sample_directory is not None else None
        checkpoint_manager = None
        start_epoch = 0
        if checkpoint_directory is not None:
            checkpoint = tf.train.Checkpoint(
                generator=self.gan.generator,
                discriminator=self.gan.discriminator,
                generator_optimizer=self.generator_optimizer,
                discriminator_optimizer=self.discriminator_optimizer,
            )
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint,
                Path(checkpoint_directory),
                max_to_keep=3,
            )
            if resume and checkpoint_manager.latest_checkpoint:
                checkpoint.restore(checkpoint_manager.latest_checkpoint)
                metadata_path = Path(checkpoint_manager.latest_checkpoint + ".npz")
                if metadata_path.is_file():
                    metadata = np.load(metadata_path)
                    start_epoch = int(metadata["epoch"])
                    history = {
                        "generator_loss": metadata["generator_loss"].tolist(),
                        "discriminator_loss": metadata["discriminator_loss"].tolist(),
                    }
        for epoch in range(start_epoch + 1, epochs + 1):
            generator_losses = []
            discriminator_losses = []
            batches = dataset
            if steps_per_epoch is not None:
                batches = dataset.take(steps_per_epoch)
            for real_images, batch_conditions in batches:
                generator_loss, discriminator_loss = self._train_step(
                    real_images,
                    batch_conditions,
                )
                generator_losses.append(float(generator_loss))
                discriminator_losses.append(float(discriminator_loss))
            history["generator_loss"].append(float(np.mean(generator_losses)))
            history["discriminator_loss"].append(float(np.mean(discriminator_losses)))
            if sample_directory is not None and (epoch % sample_interval == 0 or epoch == epochs):
                self.save_samples(
                    sample_directory / f"epoch_{epoch:04d}",
                    sample_count,
                    seed,
                    crop_margin=self.patch_size // 16,
                )
            if checkpoint_manager is not None:
                checkpoint_path = checkpoint_manager.save(checkpoint_number=epoch)
                np.savez(
                    checkpoint_path + ".npz",
                    epoch=epoch,
                    generator_loss=np.asarray(history["generator_loss"], dtype=np.float32),
                    discriminator_loss=np.asarray(history["discriminator_loss"], dtype=np.float32),
                )
        return history

    @tf.function
    def _train_step(self, real_images, conditions):
        noise = tf.random.normal((tf.shape(real_images)[0], self.latent_dim))
        generator_input = tf.concat([noise, conditions], axis=1)
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            fake_images = self.gan.generator(generator_input, training=True)
            real_logits = self.gan.discriminator([real_images, conditions], training=True)
            fake_logits = self.gan.discriminator([fake_images, conditions], training=True)
            generator_loss = self.loss(tf.ones_like(fake_logits), fake_logits)
            discriminator_loss = self.loss(tf.ones_like(real_logits), real_logits)
            discriminator_loss += self.loss(tf.zeros_like(fake_logits), fake_logits)
        generator_gradients = generator_tape.gradient(generator_loss, self.gan.generator.trainable_variables)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.gan.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.gan.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.gan.discriminator.trainable_variables))
        return generator_loss, discriminator_loss

    def generate(self, count=1, seed=None, condition_vector=None, crop_margin=0):
        if crop_margin < 0 or crop_margin * 2 >= self.patch_size:
            raise ValueError(
                "crop_margin must be non-negative and smaller than half patch_size"
            )
        if seed is not None:
            tf.random.set_seed(seed)
        noise = tf.random.normal((count, self.latent_dim))
        condition = self._resolve_condition(condition_vector)
        conditions = tf.repeat(condition[None, :], count, axis=0)
        generator_input = tf.concat([noise, conditions], axis=1)
        images = (self.gan.generator(generator_input, training=False) + 1.0) / 2.0
        if crop_margin:
            images = images[:, crop_margin:-crop_margin, crop_margin:-crop_margin, :]
        return images

    def save_samples(self, directory, count=8, seed=42, crop_margin=0):
        """Generate and save RGB patches to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        samples = self.generate(
            count=count, seed=seed, crop_margin=crop_margin
        ).numpy()
        for index, sample in enumerate(samples):
            image = np.clip(sample * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(image, mode="RGB").save(directory / f"sample_{index:03d}.png")

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.gan.generator.save_weights(directory / "generator.weights.h5")
        self.gan.discriminator.save_weights(directory / "discriminator.weights.h5")

    def get_training_data(self):
        # Load the input image and extract patches for training
        return extract_patches(self.source_image)

    def generate_patches(self, image, patch_size=(64, 64), stride=32):
        size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        return extract_patches(image, size, stride)
