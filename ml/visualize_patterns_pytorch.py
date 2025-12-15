"""
Visualize generated bullet patterns from trained PyTorch decoder
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from glob import glob

# Same Decoder architecture as training
class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Decoder, self).__init__()
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

def find_latest_decoder():
    """Find most recent decoder model"""
    models = glob("models/pattern_decoder_*.pth")
    if not models:
        raise FileNotFoundError("No decoder models found!")

    # Filter out 'final' versions, prefer best checkpoints
    best_models = [m for m in models if 'final' not in m]
    if best_models:
        latest = sorted(best_models)[-1]
    else:
        latest = sorted(models)[-1]

    return latest

def generate_patterns(decoder, num_patterns=16, latent_dim=32):
    """Generate random patterns"""
    decoder.eval()

    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn(num_patterns, latent_dim)

        # Generate patterns
        patterns = decoder(z)

        # Convert to numpy [N, H, W, C]
        patterns = patterns.permute(0, 2, 3, 1).cpu().numpy()

    return patterns, z.numpy()

def visualize_pattern_grid(patterns, save_path=None):
    """Visualize grid of patterns"""
    n = len(patterns)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(16, rows * 2))
    fig.suptitle('Generated Bullet Patterns', fontsize=16, fontweight='bold')

    for i in range(n):
        row = i // cols
        col = (i % cols) * 2

        pattern = patterns[i]
        intensity = pattern[:, :, 0]
        direction = pattern[:, :, 1]

        # Intensity channel
        if rows > 1:
            ax1 = axes[row, col]
            ax2 = axes[row, col + 1]
        else:
            ax1 = axes[col]
            ax2 = axes[col + 1]

        im1 = ax1.imshow(intensity, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f'Pattern {i+1}\nIntensity', fontsize=10)
        ax1.axis('off')

        # Direction channel
        im2 = ax2.imshow(direction, cmap='hsv', vmin=0, vmax=1)
        ax2.set_title(f'Pattern {i+1}\nDirection', fontsize=10)
        ax2.axis('off')

    # Hide empty subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        if rows > 1:
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
        else:
            if col < len(axes):
                axes[col].axis('off')
            if col + 1 < len(axes):
                axes[col + 1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved grid to: {save_path}")

    return fig

def visualize_detailed_pattern(pattern, latent_vector=None, save_path=None):
    """Detailed visualization of single pattern"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    intensity = pattern[:, :, 0]
    direction = pattern[:, :, 1]

    # Intensity heatmap
    im1 = axes[0].imshow(intensity, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title('Spawn Intensity\n(Bullet Density)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, label='0=no spawn, 1=max spawn')

    # Direction heatmap
    im2 = axes[1].imshow(direction, cmap='hsv', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Direction Field\n(0-2π radians)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, label='0=0°, 0.5=180°, 1=360°')

    # Vector field (quiver plot)
    step = 2  # Sample every 2 pixels for clarity
    y, x = np.meshgrid(np.arange(0, 32, step), np.arange(0, 32, step))
    angles = direction[::step, ::step] * 2 * np.pi
    intensities = intensity[::step, ::step]

    u = np.cos(angles) * intensities
    v = np.sin(angles) * intensities

    axes[2].quiver(x, y, u, v, intensities, cmap='hot', scale=8, width=0.003)
    axes[2].set_title('Bullet Vector Field\n(Direction & Magnitude)', fontsize=14, fontweight='bold')
    axes[2].set_xlim(-1, 32)
    axes[2].set_ylim(-1, 32)
    axes[2].invert_yaxis()
    axes[2].set_aspect('equal')
    axes[2].grid(alpha=0.3)

    if latent_vector is not None:
        fig.suptitle(f'Latent seed (first 8): [{", ".join([f"{v:.2f}" for v in latent_vector[:8]])}...]',
                     fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved detailed pattern to: {save_path}")

    return fig

def export_patterns_json(patterns, latent_vectors, output_path):
    """Export patterns as JSON for game integration"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pattern_library = []

    for i, (pattern, latent) in enumerate(zip(patterns, latent_vectors)):
        intensity = pattern[:, :, 0]
        direction = pattern[:, :, 1]

        pattern_data = {
            'id': i,
            'latent_vector': latent.tolist(),
            'intensity': intensity.tolist(),
            'direction': direction.tolist(),
            'stats': {
                'mean_intensity': float(intensity.mean()),
                'max_intensity': float(intensity.max()),
                'std_intensity': float(intensity.std()),
                'density_30': float((intensity > 0.3).sum() / (32 * 32)),
                'density_50': float((intensity > 0.5).sum() / (32 * 32)),
                'density_70': float((intensity > 0.7).sum() / (32 * 32)),
            }
        }
        pattern_library.append(pattern_data)

    output_data = {
        'pattern_count': len(patterns),
        'resolution': [32, 32],
        'latent_dim': 32,
        'patterns': pattern_library
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Exported {len(patterns)} patterns to: {output_path}")

def main():
    print("="*60)
    print("Bullet Pattern Visualization (PyTorch)")
    print("="*60)

    # Load decoder
    decoder_path = find_latest_decoder()
    print(f"\nLoading decoder: {decoder_path}")

    decoder = Decoder(latent_dim=32)
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    decoder.eval()

    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {param_count:,}")

    # Generate patterns
    print(f"\nGenerating 16 patterns...")
    patterns, latent_vectors = generate_patterns(decoder, num_patterns=16)

    print(f"Generated patterns shape: {patterns.shape}")
    print(f"Intensity range: [{patterns[:,:,:,0].min():.3f}, {patterns[:,:,:,0].max():.3f}]")
    print(f"Direction range: [{patterns[:,:,:,1].min():.3f}, {patterns[:,:,:,1].max():.3f}]")

    # Create visualizations directory
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    # Visualize grid
    print(f"\nCreating pattern grid...")
    fig1 = visualize_pattern_grid(patterns, save_path=f"{vis_dir}/pattern_grid.png")

    # Visualize detailed patterns (first 3)
    print(f"\nCreating detailed visualizations...")
    for i in range(min(3, len(patterns))):
        visualize_detailed_pattern(
            patterns[i],
            latent_vectors[i],
            save_path=f"{vis_dir}/pattern_{i+1}_detailed.png"
        )

    # Export to JSON
    print(f"\nExporting patterns to JSON...")
    export_patterns_json(patterns, latent_vectors, f"{vis_dir}/pattern_library.json")

    print(f"\n{'='*60}")
    print(f"✓ Visualization complete!")
    print(f"  Output directory: {vis_dir}/")
    print(f"  - pattern_grid.png (overview)")
    print(f"  - pattern_X_detailed.png (detailed views)")
    print(f"  - pattern_library.json (game-ready data)")
    print(f"{'='*60}")

    # Show plots (optional)
    # plt.show()

if __name__ == "__main__":
    main()
