"""
Visualize generated bullet patterns
Test pattern generation before game integration
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json

MODEL_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/models"
OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/visualizations"

def find_latest_decoder():
    """Find the most recent decoder model"""
    models = [f for f in os.listdir(MODEL_PATH) if f.startswith("pattern_decoder") and f.endswith(".h5")]
    if not models:
        raise FileNotFoundError("No decoder models found! Train first.")

    latest = sorted(models)[-1]
    return os.path.join(MODEL_PATH, latest)

def generate_patterns(decoder, num_patterns=16):
    """Generate random patterns from latent space"""
    # Sample random latent vectors
    latent_vectors = np.random.randn(num_patterns, 32).astype(np.float32)

    # Generate patterns
    patterns = decoder.predict(latent_vectors, verbose=0)

    return patterns, latent_vectors

def visualize_pattern_grid(patterns, save_path=None):
    """Visualize a grid of patterns"""
    n = len(patterns)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(16, rows * 2))
    fig.suptitle('Generated Bullet Patterns', fontsize=16)

    for i in range(n):
        row = i // cols
        col = (i % cols) * 2

        pattern = patterns[i]
        intensity = pattern[:, :, 0]
        direction = pattern[:, :, 1]

        # Intensity channel
        ax1 = axes[row, col] if rows > 1 else axes[col]
        im1 = ax1.imshow(intensity, cmap='hot', vmin=0, vmax=1)
        ax1.set_title(f'Pattern {i+1} - Intensity')
        ax1.axis('off')

        # Direction channel
        ax2 = axes[row, col + 1] if rows > 1 else axes[col + 1]
        im2 = ax2.imshow(direction, cmap='hsv', vmin=0, vmax=1)
        ax2.set_title(f'Pattern {i+1} - Direction')
        ax2.axis('off')

    # Hide empty subplots
    for i in range(n, rows * cols):
        row = i // cols
        col = (i % cols) * 2
        if rows > 1:
            axes[row, col].axis('off')
            axes[row, col + 1].axis('off')
        else:
            axes[col].axis('off')
            axes[col + 1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")

    plt.show()

def visualize_single_pattern_detailed(pattern, latent_vector=None, save_path=None):
    """Detailed visualization of a single pattern"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    intensity = pattern[:, :, 0]
    direction = pattern[:, :, 1]

    # Intensity
    im1 = axes[0].imshow(intensity, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title('Spawn Intensity', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Direction
    im2 = axes[1].imshow(direction, cmap='hsv', vmin=0, vmax=1)
    axes[1].set_title('Direction (0-2π)', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Vector field visualization
    y, x = np.meshgrid(np.arange(0, 32, 2), np.arange(0, 32, 2))
    angles = direction[::2, ::2] * 2 * np.pi
    intensities = intensity[::2, ::2]

    u = np.cos(angles) * intensities
    v = np.sin(angles) * intensities

    axes[2].quiver(x, y, u, v, intensities, cmap='hot', scale=8)
    axes[2].set_title('Vector Field (Bullet Directions)', fontsize=14)
    axes[2].set_xlim(-1, 32)
    axes[2].set_ylim(-1, 32)
    axes[2].invert_yaxis()
    axes[2].set_aspect('equal')

    if latent_vector is not None:
        fig.suptitle(f'Latent vector (first 8): [{", ".join([f"{v:.2f}" for v in latent_vector[:8]])}...]', fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")

    plt.show()

def export_patterns_to_json(patterns, latent_vectors, output_path):
    """Export patterns as JSON for game integration"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pattern_library = []

    for i, (pattern, latent) in enumerate(zip(patterns, latent_vectors)):
        pattern_data = {
            'id': i,
            'latent_vector': latent.tolist(),
            'intensity': pattern[:, :, 0].tolist(),
            'direction': pattern[:, :, 1].tolist(),
            'stats': {
                'mean_intensity': float(pattern[:, :, 0].mean()),
                'max_intensity': float(pattern[:, :, 0].max()),
                'density': float((pattern[:, :, 0] > 0.3).sum() / (32 * 32))
            }
        }
        pattern_library.append(pattern_data)

    with open(output_path, 'w') as f:
        json.dump({'patterns': pattern_library}, f)

    print(f"✓ Exported {len(patterns)} patterns to: {output_path}")

def main():
    print("="*60)
    print("Bullet Pattern Visualization")
    print("="*60)

    # Load model
    decoder_path = find_latest_decoder()
    print(f"\nLoading decoder: {decoder_path}")
    decoder = tf.keras.models.load_model(decoder_path)

    # Generate patterns
    print(f"\nGenerating patterns...")
    patterns, latent_vectors = generate_patterns(decoder, num_patterns=16)

    print(f"Generated {len(patterns)} patterns")
    print(f"Shape: {patterns.shape}")

    # Visualize grid
    print(f"\nCreating visualizations...")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    visualize_pattern_grid(
        patterns,
        save_path=os.path.join(OUTPUT_PATH, "pattern_grid.png")
    )

    # Visualize detailed single pattern
    visualize_single_pattern_detailed(
        patterns[0],
        latent_vectors[0],
        save_path=os.path.join(OUTPUT_PATH, "pattern_detailed.png")
    )

    # Export to JSON
    json_path = os.path.join(OUTPUT_PATH, "pattern_library.json")
    export_patterns_to_json(patterns, latent_vectors, json_path)

    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
