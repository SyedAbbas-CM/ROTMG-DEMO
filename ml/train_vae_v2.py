"""
VAE v2: 8-Channel Bullet Pattern Generator
==========================================
Outputs: spawn, direction, size, speed, acceleration, curve, wave_amp, wave_freq

Features:
- Expanded from 2 to 8 channels
- Contrast loss to prevent mode collapse
- Range loss to encourage full 0-1 usage
- Synthetic pattern augmentation
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

# ==============================================================================
# Configuration
# ==============================================================================
ORIGINAL_DATA_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/patterns_dataset/patterns_dataset.npy"
MODEL_OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/models"
PATTERN_OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/visualizations/pattern_library.json"

LATENT_DIM = 64          # Larger latent space for more info
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 20
MAX_KL_WEIGHT = 0.005    # Very low KL for good reconstruction
CONTRAST_WEIGHT = 0.3    # Weight for contrast loss
RANGE_WEIGHT = 0.3       # Weight for range loss

N_SYNTHETIC = 3000       # Number of synthetic patterns to add
N_OUTPUT_PATTERNS = 50   # Patterns to generate for game

CHANNEL_NAMES = ['spawn', 'direction', 'size', 'speed', 'accel', 'curve', 'wave_amp', 'wave_freq']

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# Data Expansion: 2 channels -> 8 channels
# ==============================================================================
def expand_training_data(original_data):
    """Expand 2-channel data to 8-channel with meaningful correlations."""
    N, H, W, C = original_data.shape
    expanded = np.zeros((N, H, W, 8), dtype=np.float32)

    print(f"Expanding {N} patterns from 2 to 8 channels...")

    for i in range(N):
        if i % 2000 == 0:
            print(f"  {i}/{N}...")

        intensity = original_data[i, :, :, 0]
        direction = original_data[i, :, :, 1]

        # Gradient for edge detection
        gy, gx = np.gradient(intensity)
        gradient_mag = np.sqrt(gx**2 + gy**2)
        grad_norm = gradient_mag / (gradient_mag.max() + 1e-6)

        # Distance from center
        cy, cx = H // 2, W // 2
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / (H // 2)

        # Channel 0: spawn - sharpened intensity
        spawn = np.clip((intensity - 0.05) / 0.9, 0, 1) ** 0.8
        expanded[i, :, :, 0] = spawn

        # Channel 1: direction - preserve
        expanded[i, :, :, 1] = direction

        # Channel 2: size - correlated with intensity
        expanded[i, :, :, 2] = np.clip(0.2 + intensity * 0.7 + np.random.randn(H, W) * 0.03, 0, 1)

        # Channel 3: speed - inverse of intensity (big = slow)
        expanded[i, :, :, 3] = np.clip(0.8 - intensity * 0.5 + np.random.randn(H, W) * 0.03, 0, 1)

        # Channel 4: acceleration - edges accelerate
        expanded[i, :, :, 4] = np.clip(0.5 + grad_norm * 0.3 + np.random.randn(H, W) * 0.05, 0, 1)

        # Channel 5: curve - based on distance (spirals)
        expanded[i, :, :, 5] = np.clip(0.5 + dist * intensity * 0.2 + np.random.randn(H, W) * 0.03, 0, 1)

        # Channel 6: wave amplitude - gradient areas wobble
        expanded[i, :, :, 6] = np.clip(grad_norm * 0.5 + np.random.randn(H, W) * 0.03, 0, 0.8)

        # Channel 7: wave frequency
        expanded[i, :, :, 7] = np.clip(0.3 + np.random.rand(H, W) * 0.4, 0, 1)

    return expanded

def generate_synthetic_patterns(n_patterns):
    """Generate synthetic patterns with clear distinct properties."""
    print(f"Generating {n_patterns} synthetic patterns...")

    patterns = []
    H, W = 32, 32
    cy, cx = H // 2, W // 2

    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    angle = np.arctan2(y_coords - cy, x_coords - cx)

    pattern_types = ['ring', 'spiral', 'cross', 'star', 'scatter', 'wave_wall', 'cone', 'double_ring']

    for i in range(n_patterns):
        if i % 1000 == 0:
            print(f"  {i}/{n_patterns}...")

        pattern = np.zeros((H, W, 8), dtype=np.float32)
        ptype = np.random.choice(pattern_types)

        if ptype == 'ring':
            radius = np.random.uniform(5, 13)
            thickness = np.random.uniform(1.5, 3)
            mask = np.abs(dist - radius) < thickness

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = (angle / (2 * np.pi) + 0.5) % 1.0
            pattern[:, :, 2] = np.random.uniform(0.4, 0.8)
            pattern[:, :, 3] = np.random.uniform(0.4, 0.7)
            pattern[:, :, 4] = 0.5
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = 0.0
            pattern[:, :, 7] = 0.0

        elif ptype == 'double_ring':
            r1 = np.random.uniform(4, 7)
            r2 = np.random.uniform(9, 13)
            mask = (np.abs(dist - r1) < 1.5) | (np.abs(dist - r2) < 1.5)

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = (angle / (2 * np.pi) + 0.5) % 1.0
            pattern[:, :, 2] = np.where(dist < (r1 + r2) / 2, 0.3, 0.7)
            pattern[:, :, 3] = np.where(dist < (r1 + r2) / 2, 0.8, 0.4)
            pattern[:, :, 4] = 0.5
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = 0.0
            pattern[:, :, 7] = 0.0

        elif ptype == 'spiral':
            n_arms = np.random.randint(2, 7)
            tightness = np.random.uniform(0.2, 0.5)
            spiral_angle = angle + dist * tightness
            mask = np.zeros((H, W), dtype=bool)
            for arm in range(n_arms):
                arm_angle = arm * 2 * np.pi / n_arms
                diff = (spiral_angle - arm_angle) % (2 * np.pi / n_arms)
                mask |= (diff < 0.3) & (dist > 3) & (dist < 14)

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = (angle / (2 * np.pi) + 0.5) % 1.0
            pattern[:, :, 2] = np.random.uniform(0.3, 0.5)
            pattern[:, :, 3] = np.random.uniform(0.5, 0.8)
            pattern[:, :, 4] = 0.5
            pattern[:, :, 5] = np.random.uniform(0.55, 0.75)
            pattern[:, :, 6] = 0.1
            pattern[:, :, 7] = 0.3

        elif ptype == 'cross':
            thickness = np.random.uniform(1.5, 3)
            mask = ((np.abs(x_coords - cx) < thickness) | (np.abs(y_coords - cy) < thickness)) & (dist < 14)

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = (angle / (2 * np.pi) + 0.5) % 1.0
            pattern[:, :, 2] = np.random.uniform(0.6, 0.9)
            pattern[:, :, 3] = np.random.uniform(0.2, 0.4)
            pattern[:, :, 4] = 0.5
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = 0.0
            pattern[:, :, 7] = 0.0

        elif ptype == 'star':
            n_points = np.random.randint(4, 9)
            mask = np.zeros((H, W), dtype=bool)
            for p in range(n_points):
                star_angle = p * 2 * np.pi / n_points
                diff = np.abs(angle - star_angle)
                diff = np.minimum(diff, 2 * np.pi - diff)
                mask |= (diff < 0.2) & (dist > 2) & (dist < 14)

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = (angle / (2 * np.pi) + 0.5) % 1.0
            pattern[:, :, 2] = np.random.uniform(0.4, 0.7)
            pattern[:, :, 3] = np.random.uniform(0.5, 0.7)
            pattern[:, :, 4] = np.random.uniform(0.5, 0.6)
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = 0.0
            pattern[:, :, 7] = 0.0

        elif ptype == 'scatter':
            density = np.random.uniform(0.15, 0.35)
            mask = np.random.rand(H, W) < density

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = np.random.rand(H, W)
            pattern[:, :, 2] = np.random.rand(H, W) * 0.5 + 0.25
            pattern[:, :, 3] = np.random.rand(H, W) * 0.4 + 0.4
            pattern[:, :, 4] = np.random.rand(H, W) * 0.2 + 0.4
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = np.random.rand(H, W) * 0.3
            pattern[:, :, 7] = np.random.rand(H, W) * 0.5

        elif ptype == 'wave_wall':
            horizontal = np.random.rand() > 0.5
            if horizontal:
                pos = np.random.randint(10, 22)
                mask = np.abs(y_coords - pos) < 2
                dir_val = 0.75 if np.random.rand() > 0.5 else 0.25
            else:
                pos = np.random.randint(10, 22)
                mask = np.abs(x_coords - pos) < 2
                dir_val = 0.5 if np.random.rand() > 0.5 else 0.0

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = dir_val
            pattern[:, :, 2] = np.random.uniform(0.4, 0.6)
            pattern[:, :, 3] = np.random.uniform(0.5, 0.7)
            pattern[:, :, 4] = 0.5
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = np.random.uniform(0.4, 0.7)
            pattern[:, :, 7] = np.random.uniform(0.4, 0.6)

        elif ptype == 'cone':
            aim = np.random.uniform(0, 2 * np.pi)
            width = np.random.uniform(0.4, 1.2)
            diff = np.abs(angle - aim)
            diff = np.minimum(diff, 2 * np.pi - diff)
            mask = (diff < width) & (dist > 3) & (dist < 14)

            pattern[:, :, 0] = mask.astype(float)
            pattern[:, :, 1] = (aim / (2 * np.pi)) % 1.0
            pattern[:, :, 2] = np.random.uniform(0.25, 0.45)
            pattern[:, :, 3] = np.random.uniform(0.7, 0.9)
            pattern[:, :, 4] = np.random.uniform(0.55, 0.7)
            pattern[:, :, 5] = 0.5
            pattern[:, :, 6] = 0.0
            pattern[:, :, 7] = 0.0

        patterns.append(pattern)

    return np.array(patterns, dtype=np.float32)

# ==============================================================================
# Dataset
# ==============================================================================
class PatternDataset(Dataset):
    def __init__(self, data, augment=False):
        self.data = torch.FloatTensor(data).permute(0, 3, 1, 2)  # [N,H,W,C] -> [N,C,H,W]
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.augment and np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                x = torch.flip(x, [2])
            if np.random.rand() > 0.5:
                x = torch.flip(x, [1])
            k = np.random.randint(0, 4)
            if k > 0:
                x = torch.rot90(x, k, [1, 2])
        return x

# ==============================================================================
# Model Architecture
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self, in_channels=8, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.global_pool(x).view(x.size(0), -1)
        x = F.leaky_relu(self.fc(x), 0.2)
        return self.fc_mean(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=8, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 256 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = x.view(x.size(0), 256, 2, 2)
        x = F.leaky_relu(self.bn1(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.deconv3(x)), 0.2)
        x = torch.sigmoid(self.deconv4(x))
        return x

class VAE(nn.Module):
    def __init__(self, channels=8, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(channels, latent_dim)
        self.decoder = Decoder(channels, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim).to(next(self.parameters()).device)
        return self.decoder(z)

# ==============================================================================
# Loss Function with Contrast and Range Penalties
# ==============================================================================
def vae_loss(recon, target, mean, logvar, kl_weight, contrast_weight, range_weight):
    # MSE reconstruction loss
    mse = F.mse_loss(recon, target, reduction='mean')

    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # Contrast loss: penalize low variance per channel
    variance = recon.var(dim=[2, 3])  # [batch, channels]
    min_variance = 0.05
    contrast_loss = F.relu(min_variance - variance).mean()

    # Range loss: penalize if not using full 0-1 range
    mins = recon.min(dim=2)[0].min(dim=2)[0]  # [batch, channels]
    maxs = recon.max(dim=2)[0].max(dim=2)[0]
    ranges = maxs - mins
    min_range = 0.5
    range_loss = F.relu(min_range - ranges).mean()

    total = mse + kl_weight * kl + contrast_weight * contrast_loss + range_weight * range_loss

    return total, mse, kl, contrast_loss, range_loss

# ==============================================================================
# Training
# ==============================================================================
def train_epoch(model, loader, optimizer, kl_w, contrast_w, range_w):
    model.train()
    total_loss, total_mse, total_kl, total_contrast, total_range = 0, 0, 0, 0, 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        recon, mean, logvar = model(data)
        loss, mse, kl, contrast, rng = vae_loss(recon, data, mean, logvar, kl_w, contrast_w, range_w)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_kl += kl.item()
        total_contrast += contrast.item()
        total_range += rng.item()

    n = len(loader)
    return total_loss/n, total_mse/n, total_kl/n, total_contrast/n, total_range/n

def validate(model, loader, kl_w, contrast_w, range_w):
    model.eval()
    total_loss, total_mse = 0, 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            recon, mean, logvar = model(data)
            loss, mse, _, _, _ = vae_loss(recon, data, mean, logvar, kl_w, contrast_w, range_w)
            total_loss += loss.item()
            total_mse += mse.item()

    n = len(loader)
    return total_loss/n, total_mse/n

def generate_pattern_library(model, n_patterns):
    """Generate patterns and save as JSON for game use."""
    model.eval()

    with torch.no_grad():
        patterns = model.generate(n_patterns).cpu().numpy()

    # Convert to game format
    pattern_list = []
    for i in range(n_patterns):
        p = patterns[i]  # [8, 32, 32]
        p = np.transpose(p, (1, 2, 0))  # [32, 32, 8]

        # Analyze pattern
        spawn = p[:, :, 0]
        density = (spawn > 0.3).sum() / (32 * 32)

        pattern_list.append({
            "id": i,
            "name": f"pattern_{i}",
            "channels": p.tolist(),
            "stats": {
                "density": float(density),
                "mean_spawn": float(spawn.mean()),
                "mean_size": float(p[:, :, 2].mean()),
                "mean_speed": float(p[:, :, 3].mean())
            }
        })

    output = {
        "version": 2,
        "channels": CHANNEL_NAMES,
        "pattern_count": n_patterns,
        "latent_dim": LATENT_DIM,
        "patterns": pattern_list
    }

    with open(PATTERN_OUTPUT_PATH, 'w') as f:
        json.dump(output, f)

    print(f"Saved {n_patterns} patterns to {PATTERN_OUTPUT_PATH}")

# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 70)
    print("VAE v2: 8-Channel Bullet Pattern Generator")
    print("=" * 70)

    # Load and expand data
    print(f"\nLoading original data: {ORIGINAL_DATA_PATH}")
    original = np.load(ORIGINAL_DATA_PATH)
    print(f"Original shape: {original.shape}")

    expanded = expand_training_data(original)
    synthetic = generate_synthetic_patterns(N_SYNTHETIC)

    combined = np.concatenate([expanded, synthetic], axis=0)
    np.random.shuffle(combined)

    print(f"\nCombined dataset: {combined.shape}")
    print(f"Memory: {combined.nbytes / 1024 / 1024:.1f} MB")

    # Per-channel stats
    print("\nPer-channel statistics:")
    for c, name in enumerate(CHANNEL_NAMES):
        ch = combined[:, :, :, c]
        print(f"  {c}: {name:12s} [{ch.min():.3f}, {ch.max():.3f}] mean={ch.mean():.3f} std={ch.std():.3f}")

    # Split
    split = int(0.9 * len(combined))
    train_data, val_data = combined[:split], combined[split:]

    train_loader = DataLoader(PatternDataset(train_data, augment=True), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PatternDataset(val_data), batch_size=BATCH_SIZE)

    # Model
    model = VAE(channels=8, latent_dim=LATENT_DIM).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Decoder size: ~{sum(p.numel() for p in model.decoder.parameters()) * 4 / 1024:.1f} KB")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # Training
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    best_val = float('inf')
    patience, patience_counter = 100, 0

    print("\n" + "=" * 70)
    print(f"Training for {EPOCHS} epochs...")
    print(f"KL warmup: 0 -> {MAX_KL_WEIGHT} over {WARMUP_EPOCHS} epochs")
    print(f"Contrast weight: {CONTRAST_WEIGHT}, Range weight: {RANGE_WEIGHT}")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # KL annealing
        kl_w = min((epoch / WARMUP_EPOCHS) * MAX_KL_WEIGHT, MAX_KL_WEIGHT)

        train_loss, train_mse, train_kl, train_c, train_r = train_epoch(
            model, train_loader, optimizer, kl_w, CONTRAST_WEIGHT, RANGE_WEIGHT
        )
        val_loss, val_mse = validate(model, val_loader, kl_w, CONTRAST_WEIGHT, RANGE_WEIGHT)

        scheduler.step()

        if epoch % 10 == 0 or epoch <= 5:
            print(f"E{epoch:4d} | Loss: {train_loss:.4f} MSE: {train_mse:.4f} KL: {train_kl:.4f} "
                  f"C: {train_c:.4f} R: {train_r:.4f} | Val: {val_loss:.4f}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{MODEL_OUTPUT_PATH}/vae_v2_best_{timestamp}.pth")
            torch.save(model.decoder.state_dict(), f"{MODEL_OUTPUT_PATH}/decoder_v2_{timestamp}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Generate patterns
    print("\n" + "=" * 70)
    print("Generating pattern library...")
    model.load_state_dict(torch.load(f"{MODEL_OUTPUT_PATH}/vae_v2_best_{timestamp}.pth"))
    generate_pattern_library(model, N_OUTPUT_PATTERNS)

    # Analyze output quality
    print("\nAnalyzing generated patterns...")
    with torch.no_grad():
        samples = model.generate(100).cpu().numpy()

    print("Generated pattern statistics:")
    for c, name in enumerate(CHANNEL_NAMES):
        ch = samples[:, c, :, :]
        print(f"  {c}: {name:12s} [{ch.min():.3f}, {ch.max():.3f}] mean={ch.mean():.3f} std={ch.std():.3f}")

    # Save config
    config = {
        "version": 2,
        "latent_dim": LATENT_DIM,
        "channels": 8,
        "channel_names": CHANNEL_NAMES,
        "timestamp": timestamp,
        "best_val_loss": float(best_val),
        "n_train": len(train_data),
        "n_synthetic": N_SYNTHETIC
    }
    with open(f"{MODEL_OUTPUT_PATH}/config_v2_{timestamp}.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Training complete! Best val loss: {best_val:.4f}")
    print(f"Model: {MODEL_OUTPUT_PATH}/vae_v2_best_{timestamp}.pth")
    print(f"Patterns: {PATTERN_OUTPUT_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
