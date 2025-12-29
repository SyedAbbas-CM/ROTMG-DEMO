"""
Proper VAE Evaluation Script

Tests:
1. Sample diversity - are generated patterns different?
2. Mode collapse - is the model outputting the same thing?
3. Output distributions - are channels using full range?
4. Reconstruction quality - can it recreate inputs?
5. Latent space quality - is it well-organized?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os

# Model architecture (copy from train_vae_v2.py)
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

    def forward(self, x):
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return self.decoder(z), mean, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def main():
    device = 'cpu'

    # Find and load model
    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('vae_v2_best')]
    if not model_files:
        print("No model found!")
        return

    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    print(f"Loading model: {model_path}")

    model = VAE(channels=8, latent_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print('='*60)
    print('VAE MODEL EVALUATION')
    print('='*60)

    # 1. Generate samples and check diversity
    print('\n1. SAMPLE DIVERSITY TEST')
    print('-'*40)

    n_samples = 100
    with torch.no_grad():
        z = torch.randn(n_samples, 64)
        samples = model.decode(z).numpy()  # [100, 8, 32, 32]

    channel_names = ['spawn', 'direction', 'size', 'speed', 'accel', 'curve', 'wave_amp', 'wave_freq']
    print('Cross-sample variance per channel (higher = more diverse):')
    for i, name in enumerate(channel_names):
        ch_samples = samples[:, i, :, :]
        var_per_pixel = ch_samples.var(axis=0)
        mean_var = var_per_pixel.mean()
        max_var = var_per_pixel.max()
        print(f'  {name:12}: mean_var={mean_var:.4f}, max_var={max_var:.4f}')

    # 2. Mode collapse test
    print('\n2. MODE COLLAPSE TEST')
    print('-'*40)

    similarities = []
    for i in range(0, 50, 2):
        s1 = samples[i].flatten()
        s2 = samples[i+1].flatten()
        sim = np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
        similarities.append(sim)

    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    print(f'Pairwise cosine similarity: {mean_sim:.4f} +/- {std_sim:.4f}')
    print(f'  (1.0 = identical, lower = more diverse)')

    if mean_sim > 0.95:
        print('  [FAIL] Severe mode collapse - all outputs nearly identical')
    elif mean_sim > 0.85:
        print('  [WARN] High similarity - limited diversity')
    elif mean_sim > 0.7:
        print('  [OK] Moderate diversity')
    else:
        print('  [GOOD] High diversity')

    # 3. Output distributions
    print('\n3. OUTPUT VALUE DISTRIBUTIONS')
    print('-'*40)

    for i, name in enumerate(channel_names):
        ch = samples[:, i, :, :].flatten()
        print(f'{name:12}: [{ch.min():.3f}, {ch.max():.3f}] mean={ch.mean():.3f} std={ch.std():.3f}')

    # 4. Spawn pattern analysis
    print('\n4. SPAWN PATTERN ANALYSIS')
    print('-'*40)

    spawn = samples[:, 0, :, :]
    print('Fraction of pixels above threshold:')
    for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
        frac = (spawn > thresh).mean()
        print(f'  spawn > {thresh}: {frac*100:.1f}%')

    # Check spatial distribution - are spawns concentrated in center?
    center_spawn = spawn[:, 12:20, 12:20].mean()
    edge_spawn = np.concatenate([
        spawn[:, :4, :].flatten(),
        spawn[:, -4:, :].flatten(),
        spawn[:, :, :4].flatten(),
        spawn[:, :, -4:].flatten()
    ]).mean()
    print(f'\nSpatial distribution:')
    print(f'  Center (12:20, 12:20) mean spawn: {center_spawn:.3f}')
    print(f'  Edges mean spawn: {edge_spawn:.3f}')
    if center_spawn > edge_spawn * 1.5:
        print('  [WARN] Spawns concentrated in center')

    # 5. Direction analysis
    print('\n5. DIRECTION CHANNEL ANALYSIS')
    print('-'*40)

    direction = samples[:, 1, :, :].flatten()
    # Check if directions point outward from center (radial)

    # For each sample, check correlation between pixel position and direction
    radial_correlations = []
    for s in range(min(20, n_samples)):
        dir_grid = samples[s, 1, :, :]  # [32, 32]
        for y in range(32):
            for x in range(32):
                # Expected radial direction from center
                dx, dy = x - 16, y - 16
                if dx == 0 and dy == 0:
                    continue
                expected_angle = np.arctan2(dy, dx)
                expected_norm = (expected_angle / (2 * np.pi)) % 1.0
                actual_norm = dir_grid[y, x]
                # Check if actual is close to expected (radial outward)
                diff = min(abs(actual_norm - expected_norm),
                          1 - abs(actual_norm - expected_norm))
                radial_correlations.append(1 - diff * 2)  # 1 = perfect radial, 0 = random

    radial_score = np.mean(radial_correlations)
    print(f'Radial direction score: {radial_score:.3f}')
    print(f'  (1.0 = all bullets point outward from center)')
    print(f'  (0.5 = random directions)')
    if radial_score > 0.7:
        print('  [PROBLEM] Directions are mostly radial - explains "burst" pattern')

    # 6. Latent space analysis
    print('\n6. LATENT SPACE ANALYSIS')
    print('-'*40)

    # Load training data and encode
    with open('visualizations/pattern_library.json') as f:
        lib = json.load(f)

    latents = []
    for p in lib['patterns']:
        channels = np.array(p['channels'])
        x = torch.tensor(channels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            mean, logvar = model.encode(x)
            latents.append(mean.numpy()[0])

    latents = np.array(latents)
    print(f'Encoded {len(latents)} training patterns')
    print(f'Latent mean: {latents.mean():.3f} (ideal: 0)')
    print(f'Latent std: {latents.std():.3f} (ideal: 1)')

    dim_stds = latents.std(axis=0)
    active_dims = (dim_stds > 0.1).sum()
    dead_dims = (dim_stds < 0.01).sum()
    print(f'Active dimensions (std > 0.1): {active_dims}/64')
    print(f'Dead dimensions (std < 0.01): {dead_dims}/64')

    # 7. Reconstruction test
    print('\n7. RECONSTRUCTION QUALITY')
    print('-'*40)

    recon_errors = []
    for p in lib['patterns'][:20]:
        channels = np.array(p['channels'])
        x = torch.tensor(channels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            recon, mean, logvar = model(x)
            mse = F.mse_loss(recon, x).item()
            recon_errors.append(mse)

    print(f'Mean reconstruction MSE: {np.mean(recon_errors):.4f}')
    print(f'  (lower is better)')

    # 8. Summary and diagnosis
    print('\n' + '='*60)
    print('DIAGNOSIS')
    print('='*60)

    issues = []

    if mean_sim > 0.85:
        issues.append("Mode collapse: outputs too similar")

    if radial_score > 0.7:
        issues.append("Radial bias: all directions point outward from center")

    if center_spawn > edge_spawn * 1.5:
        issues.append("Spatial bias: spawns concentrated in center")

    spawn_std = samples[:, 0, :, :].std()
    if spawn_std < 0.15:
        issues.append("Low spawn variance: not enough variation in spawn probability")

    curve_range = samples[:, 5, :, :].max() - samples[:, 5, :, :].min()
    if curve_range < 0.3:
        issues.append("Curve channel collapsed: no curved bullets")

    if dead_dims > 30:
        issues.append(f"Many dead latent dims: {dead_dims}/64 unused")

    if issues:
        print("\nPROBLEMS FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nROOT CAUSE ANALYSIS:")
        print("  The VAE learned to produce 'average' patterns because:")
        print("  1. Training data may all share similar structure (radial bursts)")
        print("  2. MSE loss encourages averaging, not diversity")
        print("  3. KL loss may be too strong, collapsing latent space")

        print("\nRECOMMENDED FIXES:")
        print("  1. Train with more diverse patterns (spirals, waves, lines)")
        print("  2. Use adversarial training (VAE-GAN) for sharper outputs")
        print("  3. Reduce KL weight to allow more latent variation")
        print("  4. Add perceptual/structural losses instead of just MSE")
    else:
        print("\nModel appears healthy!")


if __name__ == '__main__':
    main()
