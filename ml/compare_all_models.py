"""
Systematic Model Comparison
Compare ALL trained models on multiple metrics
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset
print("Loading dataset...")
data = np.load('patterns_dataset/patterns_dataset.npy')
split_idx = int(0.9 * len(data))
test_data = data[split_idx:]
print(f"Test samples: {len(test_data)}")

# Original architecture
class OriginalEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.fc_mean(x), self.fc_logvar(x)

class OriginalDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 2, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 64, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class OriginalVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = OriginalEncoder(latent_dim)
        self.decoder = OriginalDecoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

# Fixed architecture
class FixedEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 128)
        self.fc_mean = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.fc_mean(x), self.fc_logvar(x)

class FixedDecoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.deconv4 = nn.ConvTranspose2d(32, 2, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 2, 2)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        return x

class FixedVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = FixedEncoder(latent_dim)
        self.decoder = FixedDecoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

def evaluate_model(model, model_name, test_data, num_samples=50):
    """Evaluate model on multiple metrics"""
    model.eval()

    # Random sample of test data
    indices = np.random.choice(len(test_data), num_samples, replace=False)
    samples = test_data[indices]

    # Convert to torch
    x = torch.FloatTensor(samples).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        recon, mean, logvar = model(x)
        recon_np = recon.cpu().numpy()
        x_np = x.cpu().numpy()

    # Metrics
    mse_scores = []
    ssim_scores = []

    for i in range(num_samples):
        # MSE
        mse = np.mean((x_np[i] - recon_np[i]) ** 2)
        mse_scores.append(mse)

        # SSIM (per channel, then average)
        ssim_channels = []
        for c in range(2):
            s = ssim(x_np[i, c], recon_np[i, c], data_range=1.0)
            ssim_channels.append(s)
        ssim_scores.append(np.mean(ssim_channels))

    # Latent space statistics
    mean_np = mean.cpu().numpy()
    logvar_np = logvar.cpu().numpy()

    results = {
        'model_name': model_name,
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'latent_mean': np.mean(np.abs(mean_np)),
        'latent_std': np.std(mean_np),
        'logvar_mean': np.mean(logvar_np),
        'kl_divergence': -0.5 * np.mean(1 + logvar_np - mean_np**2 - np.exp(logvar_np))
    }

    return results, (x_np[:5], recon_np[:5])

# Find all model files
print("\nFinding models...")
model_files = sorted(glob.glob('models/vae_best_*.pth'), key=os.path.getmtime, reverse=True)
print(f"Found {len(model_files)} models")

results_list = []
visualizations = []

for model_path in model_files[:5]:  # Test top 5 most recent
    model_name = os.path.basename(model_path)
    size_kb = os.path.getsize(model_path) / 1024

    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"Size: {size_kb:.1f} KB")

    try:
        # Determine architecture
        if 'fixed' in model_name:
            model = FixedVAE(latent_dim=32).to(device)
        else:
            model = OriginalVAE(latent_dim=32).to(device)

        # Load weights
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)

        # Evaluate
        results, (originals, reconstructions) = evaluate_model(model, model_name, test_data)
        results_list.append(results)
        visualizations.append((model_name, originals, reconstructions))

        print(f"MSE:  {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")
        print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        print(f"KL Divergence: {results['kl_divergence']:.4f}")

    except Exception as e:
        print(f"Failed to load: {e}")

# Print comparison table
print(f"\n{'='*60}")
print("COMPARISON TABLE")
print(f"{'='*60}")
print(f"{'Model':<40} {'MSE':>10} {'SSIM':>10} {'KL':>10}")
print(f"{'-'*60}")

for r in sorted(results_list, key=lambda x: -x['ssim_mean']):
    name_short = r['model_name'][:38]
    print(f"{name_short:<40} {r['mse_mean']:>10.4f} {r['ssim_mean']:>10.4f} {r['kl_divergence']:>10.2f}")

# Find best model
best_ssim = max(results_list, key=lambda x: x['ssim_mean'])
best_mse = min(results_list, key=lambda x: x['mse_mean'])

print(f"\n{'='*60}")
print(f"BEST SSIM: {best_ssim['model_name']}")
print(f"  SSIM: {best_ssim['ssim_mean']:.4f}")
print(f"\nBEST MSE: {best_mse['model_name']}")
print(f"  MSE: {best_mse['mse_mean']:.4f}")
print(f"{'='*60}")

# Create visualization
fig, axes = plt.subplots(len(visualizations), 5, figsize=(15, 3*len(visualizations)))
if len(visualizations) == 1:
    axes = axes.reshape(1, -1)

for row, (name, orig, recon) in enumerate(visualizations):
    for col in range(5):
        if col < len(orig):
            # Show intensity channel only
            combined = np.concatenate([orig[col, 0], recon[col, 0]], axis=1)
            axes[row, col].imshow(combined, cmap='hot', vmin=0, vmax=1)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(f"{name[:20]}\nOrig | Recon", fontsize=8)
            else:
                axes[row, col].set_title("Orig | Recon", fontsize=8)

plt.tight_layout()
os.makedirs('test_results', exist_ok=True)
plt.savefig('test_results/model_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nSaved visualization: test_results/model_comparison.png")

# Analysis and recommendations
print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}")

avg_ssim = np.mean([r['ssim_mean'] for r in results_list])
print(f"Average SSIM across all models: {avg_ssim:.4f}")

if avg_ssim < 0.5:
    print("\n⚠ CRITICAL: All models have poor SSIM < 0.5")
    print("This suggests:")
    print("  1. VAE may not be suitable for this task")
    print("  2. Training data may have issues")
    print("  3. Hyperparameters need major changes")
    print("  4. Consider alternative architectures (U-Net, GAN, Diffusion)")
elif avg_ssim < 0.7:
    print("\n⚠ WARNING: Models have mediocre SSIM < 0.7")
    print("Recommendations:")
    print("  1. Try perceptual loss instead of MSE")
    print("  2. Reduce KL weight further")
    print("  3. Add skip connections (U-Net style)")
else:
    print("\n✓ Models show good structural similarity")

# Check if fixed architecture helped
if any('fixed' in r['model_name'] for r in results_list):
    fixed_results = [r for r in results_list if 'fixed' in r['model_name']]
    original_results = [r for r in results_list if 'fixed' not in r['model_name']]

    if fixed_results and original_results:
        fixed_avg_ssim = np.mean([r['ssim_mean'] for r in fixed_results])
        orig_avg_ssim = np.mean([r['ssim_mean'] for r in original_results])

        improvement = (fixed_avg_ssim - orig_avg_ssim) / orig_avg_ssim * 100

        print(f"\nFixed architecture SSIM: {fixed_avg_ssim:.4f}")
        print(f"Original architecture SSIM: {orig_avg_ssim:.4f}")
        print(f"Improvement: {improvement:+.1f}%")

        if improvement < 5:
            print("⚠ Architectural fix did NOT significantly improve SSIM")
            print("The problem may be elsewhere (data, loss function, task mismatch)")

print(f"\n{'='*60}")
print("NEXT STEPS")
print(f"{'='*60}")
print("1. If SSIM < 0.5: Consider completely different approach")
print("2. If SSIM 0.5-0.7: Try U-Net, perceptual loss, or more data")
print("3. If SSIM > 0.7: Optimize decoder size and deploy")
print("4. Regardless: Visualize latent space and interpolations")
