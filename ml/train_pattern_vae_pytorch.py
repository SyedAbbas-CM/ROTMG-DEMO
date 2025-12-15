"""
Lightweight VAE for Bullet Pattern Generation (PyTorch)
Optimized for Jetson Nano deployment
Target: <1MB model size, fast inference
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import os
import json
from datetime import datetime

# Configuration
DATASET_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/patterns_dataset/patterns_dataset.npy"
MODEL_OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/models"
LATENT_DIM = 32  # Seed dimension for controllable generation
BATCH_SIZE = 32
EPOCHS = 1000  # Much longer training for better convergence
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 10  # Epochs to linearly warmup KL weight
MAX_KL_WEIGHT = 0.05  # Lower KL weight for better reconstruction (was 0.1)

# Check for Apple Silicon MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class PatternDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = torch.FloatTensor(data).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, [2])
            # Random vertical flip
            if np.random.rand() > 0.5:
                x = torch.flip(x, [1])
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            if k > 0:
                x = torch.rot90(x, k, [1, 2])

        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()

        # Downsample: 32 -> 16 -> 8 -> 4
        self.conv1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)  # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 8 -> 4

        # Flatten size: 64 * 4 * 4 = 1024
        self.fc = nn.Linear(64 * 4 * 4, 128)

        # VAE parameters
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)

        # Upsample: 4 -> 8 -> 16 -> 32
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 4 -> 8
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)  # 8 -> 16
        self.deconv3 = nn.ConvTranspose2d(16, 2, 3, stride=2, padding=1, output_padding=1)   # 16 -> 32

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 64, 4, 4)  # Reshape
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # Output in [0, 1]
        return x

class PatternVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(PatternVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mean, logvar

def vae_loss(recon_x, x, mean, logvar, kl_weight=1.0):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)

    # Total loss with annealed KL weight
    # KL weight starts at 0 and gradually increases to MAX_KL_WEIGHT
    # This allows model to first learn good reconstructions, then add structure
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, optimizer, kl_weight):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        recon, mean, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon, data, mean, logvar, kl_weight)

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

def validate(model, dataloader, kl_weight):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            recon, mean, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mean, logvar, kl_weight)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches

def main():
    print("="*60)
    print("Bullet Pattern VAE Training (PyTorch)")
    print("="*60)

    # Load data
    print(f"\nLoading dataset from: {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    print(f"Dataset shape: {data.shape}")
    print(f"Memory: {data.nbytes / 1024 / 1024:.2f} MB")

    # Train/val split
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"\nTrain samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    # Create datasets and dataloaders
    train_dataset = PatternDataset(train_data, augment=True)
    val_dataset = PatternDataset(val_data, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = PatternVAE(latent_dim=LATENT_DIM).to(device)

    # Print model info
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    total_params = encoder_params + decoder_params

    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,} (used at inference)")
    print(f"  Estimated decoder size: ~{decoder_params * 4 / 1024:.1f} KB")

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training loop
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_val_loss = float('inf')
    patience = 50  # Increased patience for longer training
    patience_counter = 0

    print("\n" + "="*60)
    print("Starting training...")
    print(f"KL annealing: 0 -> {MAX_KL_WEIGHT} over {WARMUP_EPOCHS} epochs")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        # KL annealing schedule: linearly increase from 0 to MAX_KL_WEIGHT
        if epoch <= WARMUP_EPOCHS:
            kl_weight = (epoch / WARMUP_EPOCHS) * MAX_KL_WEIGHT
        else:
            kl_weight = MAX_KL_WEIGHT

        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, kl_weight)
        val_loss, val_recon, val_kl = validate(model, val_loader, kl_weight)

        # Step scheduler
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | KL_w: {kl_weight:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train: {train_loss:.4f} (R: {train_recon:.4f}, KL: {train_kl:.4f}) | "
              f"Val: {val_loss:.4f} (R: {val_recon:.4f}, KL: {val_kl:.4f})")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save full model
            torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_PATH, f"vae_best_{timestamp}.pth"))

            # Save decoder separately
            torch.save(model.decoder.state_dict(), os.path.join(MODEL_OUTPUT_PATH, f"pattern_decoder_{timestamp}.pth"))

            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience exhausted)")
                break

    # Save final model
    torch.save(model.decoder.state_dict(), os.path.join(MODEL_OUTPUT_PATH, f"pattern_decoder_final_{timestamp}.pth"))

    # Save config
    config = {
        'latent_dim': LATENT_DIM,
        'patch_size': 32,
        'total_params': int(total_params),
        'decoder_params': int(decoder_params),
        'timestamp': timestamp,
        'final_train_loss': float(train_loss),
        'final_val_loss': float(val_loss),
        'best_val_loss': float(best_val_loss),
        'device': str(device)
    }

    with open(os.path.join(MODEL_OUTPUT_PATH, f"config_{timestamp}.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Decoder saved: pattern_decoder_{timestamp}.pth")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
