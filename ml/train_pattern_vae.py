"""
Lightweight VAE for Bullet Pattern Generation
Optimized for Jetson Nano deployment
Target: <1MB model size, fast inference
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
from datetime import datetime

# Configuration
DATASET_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/patterns_dataset/patterns_dataset.npy"
MODEL_OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/models"
LATENT_DIM = 32  # Seed dimension for controllable generation
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

# Model will be tiny: ~50K parameters for Jetson Nano
ENCODER_FILTERS = [16, 32, 64]  # Small filter counts
DECODER_FILTERS = [64, 32, 16]

class PatternVAE(keras.Model):
    def __init__(self, latent_dim=32):
        super(PatternVAE, self).__init__()
        self.latent_dim = latent_dim

        # Data augmentation (training only)
        self.augmentation = keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.4),  # Up to ~144 degrees
            layers.RandomZoom(0.2),
        ])

        # Encoder (only needed for training)
        self.encoder = self.build_encoder()

        # Decoder (this is what runs during inference)
        self.decoder = self.build_decoder()

        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self):
        """Encode [32, 32, 2] -> latent vector [32]"""
        encoder_inputs = keras.Input(shape=(32, 32, 2))
        x = encoder_inputs

        # Downsample: 32 -> 16 -> 8 -> 4
        for filters in ENCODER_FILTERS:
            x = layers.Conv2D(filters, 3, strides=2, padding="same", activation="relu")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)

        # VAE parameters
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        # Sampling layer
        z = Sampling()([z_mean, z_log_var])

        return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        """Decode latent vector [32] -> [32, 32, 2]"""
        latent_inputs = keras.Input(shape=(self.latent_dim,))

        x = layers.Dense(4 * 4 * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((4, 4, 64))(x)

        # Upsample: 4 -> 8 -> 16 -> 32
        for filters in DECODER_FILTERS:
            x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same", activation="relu")(x)

        # Output: 2 channels (intensity, direction)
        decoder_outputs = layers.Conv2DTranspose(2, 3, padding="same", activation="sigmoid")(x)

        return keras.Model(latent_inputs, decoder_outputs, name="decoder")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # Apply augmentation during training
        augmented_data = self.augmentation(data, training=True)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(augmented_data)
            reconstruction = self.decoder(z)

            # Reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction),
                    axis=(1, 2)
                )
            )

            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

            # Total loss
            total_loss = reconstruction_loss + 0.01 * kl_loss  # Low KL weight for sharper patterns

        # Compute gradients and update weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class Sampling(layers.Layer):
    """Sampling layer for VAE reparameterization trick"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def load_dataset():
    """Load preprocessed dataset"""
    print(f"Loading dataset from: {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    print(f"Dataset shape: {data.shape}")
    print(f"Memory: {data.nbytes / 1024 / 1024:.2f} MB")
    return data

def train_model():
    """Main training function"""
    print("="*60)
    print("Bullet Pattern VAE Training")
    print("="*60)

    # Load data
    dataset = load_dataset()

    # Train/val split
    split_idx = int(0.9 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create model
    vae = PatternVAE(latent_dim=LATENT_DIM)

    # Compile
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

    # Print model summary
    print("\n" + "="*60)
    print("Encoder Architecture:")
    vae.encoder.summary()
    print("\n" + "="*60)
    print("Decoder Architecture:")
    vae.decoder.summary()

    # Count parameters
    total_params = vae.encoder.count_params() + vae.decoder.count_params()
    decoder_params = vae.decoder.count_params()
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Decoder parameters: {decoder_params:,} (used at inference)")
    print(f"  Estimated model size: ~{decoder_params * 4 / 1024:.1f} KB")

    # Callbacks
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_OUTPUT_PATH, f"vae_best_{timestamp}.h5"),
            save_best_only=True,
            monitor="val_total_loss"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_total_loss",
            patience=15,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_total_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    history = vae.fit(
        train_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val_data, val_data),
        callbacks=callbacks
    )

    # Save decoder separately (this is what we deploy)
    decoder_path = os.path.join(MODEL_OUTPUT_PATH, f"pattern_decoder_{timestamp}.h5")
    vae.decoder.save(decoder_path)
    print(f"\n✓ Decoder saved to: {decoder_path}")

    # Save training config
    config = {
        'latent_dim': LATENT_DIM,
        'patch_size': 32,
        'encoder_filters': ENCODER_FILTERS,
        'decoder_filters': DECODER_FILTERS,
        'total_params': int(total_params),
        'decoder_params': int(decoder_params),
        'timestamp': timestamp,
        'final_loss': float(history.history['total_loss'][-1]),
        'final_val_loss': float(history.history['val_total_loss'][-1])
    }

    with open(os.path.join(MODEL_OUTPUT_PATH, f"config_{timestamp}.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"  Final loss: {config['final_loss']:.4f}")
    print(f"  Final val loss: {config['final_val_loss']:.4f}")

    return vae, history

if __name__ == "__main__":
    vae, history = train_model()
