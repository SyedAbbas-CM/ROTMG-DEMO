"""
Test 1.1: Reconstruction Quality
Tests if the VAE can accurately reconstruct training images
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import os

# Import model architecture
from train_pattern_vae_pytorch import Encoder, Decoder, PatternVAE

MODEL_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/models"
DATASET_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/patterns_dataset/patterns_dataset.npy"
OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/test_results"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_latest_model():
    """Load the most recent VAE model"""
    models = [f for f in os.listdir(MODEL_PATH) if f.startswith("vae_best") and f.endswith(".pth")]
    if not models:
        raise FileNotFoundError("No VAE models found!")

    latest = sorted(models)[-1]
    model_path = os.path.join(MODEL_PATH, latest)

    print(f"Loading model: {model_path}")
    model = PatternVAE(latent_dim=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def load_dataset():
    """Load training dataset"""
    data = np.load(DATASET_PATH)
    print(f"Dataset shape: {data.shape}")  # Should be (11581, 32, 32, 2)
    return data

def compute_reconstruction_metrics(original, reconstructed):
    """Compute MSE and SSIM for reconstruction"""
    # Ensure shapes match
    original = original.reshape(-1)
    reconstructed = reconstructed.reshape(-1)

    mse = mean_squared_error(original, reconstructed)

    # SSIM needs 2D images
    orig_2d = original.reshape(32, 32, 2)
    recon_2d = reconstructed.reshape(32, 32, 2)

    # Compute SSIM per channel and average
    ssim_intensity = ssim(orig_2d[:,:,0], recon_2d[:,:,0], data_range=1.0)
    ssim_direction = ssim(orig_2d[:,:,1], recon_2d[:,:,1], data_range=1.0)
    avg_ssim = (ssim_intensity + ssim_direction) / 2

    return mse, avg_ssim

def test_reconstruction(model, dataset, num_samples=10):
    """Test reconstruction on random samples"""
    print(f"\nTesting reconstruction on {num_samples} random samples...")

    # Select random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    results = []

    for idx in indices:
        # Get original pattern
        original = dataset[idx]  # Shape: (32, 32, 2)

        # Convert to torch tensor
        x = torch.FloatTensor(original).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 2, 32, 32]

        # Reconstruct
        with torch.no_grad():
            recon, mean, logvar = model(x)

        # Convert back to numpy
        recon_np = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [32, 32, 2]

        # Compute metrics
        mse, ssim_score = compute_reconstruction_metrics(original, recon_np)

        results.append({
            'index': idx,
            'original': original,
            'reconstructed': recon_np,
            'mse': mse,
            'ssim': ssim_score,
            'latent': mean.cpu().numpy()
        })

    return results

def visualize_reconstructions(results, save_path):
    """Visualize original vs reconstructed patterns"""
    num_samples = len(results)
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 3))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, result in enumerate(results):
        orig = result['original']
        recon = result['reconstructed']
        mse = result['mse']
        ssim_score = result['ssim']

        # Original intensity
        axes[i, 0].imshow(orig[:,:,0], cmap='hot', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original Intensity (Sample {i+1})')
        axes[i, 0].axis('off')

        # Reconstructed intensity
        axes[i, 1].imshow(recon[:,:,0], cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Reconstructed Intensity')
        axes[i, 1].axis('off')

        # Original direction
        axes[i, 2].imshow(orig[:,:,1], cmap='hsv', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Original Direction')
        axes[i, 2].axis('off')

        # Reconstructed direction
        axes[i, 3].imshow(recon[:,:,1], cmap='hsv', vmin=0, vmax=1)
        axes[i, 3].set_title(f'Reconstructed Direction\nMSE: {mse:.4f}, SSIM: {ssim_score:.3f}')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()

def print_summary(results):
    """Print summary statistics"""
    mses = [r['mse'] for r in results]
    ssims = [r['ssim'] for r in results]

    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY SUMMARY")
    print("="*60)
    print(f"\nSamples tested: {len(results)}")
    print(f"\nMSE (Mean Squared Error):")
    print(f"  Mean: {np.mean(mses):.4f}")
    print(f"  Std:  {np.std(mses):.4f}")
    print(f"  Min:  {np.min(mses):.4f}")
    print(f"  Max:  {np.max(mses):.4f}")

    print(f"\nSSIM (Structural Similarity):")
    print(f"  Mean: {np.mean(ssims):.3f}")
    print(f"  Std:  {np.std(ssims):.3f}")
    print(f"  Min:  {np.min(ssims):.3f}")
    print(f"  Max:  {np.max(ssims):.3f}")

    # Interpret results
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    print("="*60)

    avg_mse = np.mean(mses)
    avg_ssim = np.mean(ssims)

    if avg_mse < 0.02:
        print("✓ EXCELLENT reconstruction (MSE < 0.02)")
    elif avg_mse < 0.05:
        print("✓ GOOD reconstruction (MSE < 0.05)")
    elif avg_mse < 0.10:
        print("⚠ MODERATE reconstruction (MSE < 0.10)")
    else:
        print("✗ POOR reconstruction (MSE >= 0.10)")

    if avg_ssim > 0.9:
        print("✓ EXCELLENT structure preservation (SSIM > 0.9)")
    elif avg_ssim > 0.8:
        print("✓ GOOD structure preservation (SSIM > 0.8)")
    elif avg_ssim > 0.7:
        print("⚠ MODERATE structure preservation (SSIM > 0.7)")
    else:
        print("✗ POOR structure preservation (SSIM < 0.7)")

    print("\nConclusion:")
    if avg_mse < 0.05 and avg_ssim > 0.8:
        print("✓ Model has learned to reconstruct training data well!")
        print("  → Safe to proceed to latent space exploration")
    elif avg_mse < 0.10 and avg_ssim > 0.7:
        print("⚠ Model reconstruction is moderate")
        print("  → Can proceed but patterns may be noisy")
    else:
        print("✗ Model reconstruction is poor")
        print("  → Consider retraining with better hyperparameters")

    print("="*60)

def main():
    print("="*60)
    print("TEST 1.1: Reconstruction Quality")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Load model and dataset
    model = load_latest_model()
    dataset = load_dataset()

    # Test reconstruction
    results = test_reconstruction(model, dataset, num_samples=10)

    # Visualize
    viz_path = os.path.join(OUTPUT_PATH, "reconstruction_quality.png")
    visualize_reconstructions(results, viz_path)

    # Print summary
    print_summary(results)

    print(f"\n{'='*60}")
    print("Test complete! Check:")
    print(f"  - Visualizations: {viz_path}")
    print("="*60)

if __name__ == "__main__":
    main()
