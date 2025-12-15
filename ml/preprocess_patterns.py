"""
Bullet Pattern Preprocessing Pipeline
Converts texture images into pattern fields: [32, 32, 2]
  - Channel 0: Spawn intensity (0-1) from edge magnitude
  - Channel 1: Direction (0-1 mapped to 0-2π) from gradient angle
"""

import numpy as np
import cv2
import os
from pathlib import Path
import json

# Configuration
DATASET_PATH = "/Users/az/Desktop/Rotmg-Pservers/dtd/images"
OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/patterns_dataset"
PATCH_SIZE = 32
PATCHES_PER_IMAGE = 10  # Random crops per image
MIN_INTENSITY = 0.1  # Filter out boring patches

# Best categories for bullet patterns
PATTERN_CATEGORIES = [
    'cracked', 'spiralled', 'swirly', 'veined', 'zigzagged',
    'cobwebbed', 'woven', 'dotted', 'braided', 'marbled',
    'bumpy', 'crystalline', 'fibrous', 'flecked', 'frilly'
]

def extract_pattern_field(image):
    """
    Convert image to pattern field [H, W, 2]
    Returns: (intensity, direction) channels
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Compute gradients using Sobel
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Magnitude (intensity) - normalized to [0, 1]
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)

    # Direction (angle) - normalized to [0, 1]
    angle = np.arctan2(grad_y, grad_x)  # Range: [-π, π]
    direction = (angle + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Stack channels
    pattern_field = np.stack([magnitude, direction], axis=-1).astype(np.float32)

    return pattern_field

def extract_random_patch(pattern_field, size=32):
    """Extract random patch from pattern field"""
    h, w = pattern_field.shape[:2]

    if h < size or w < size:
        # Resize if image is too small
        pattern_field = cv2.resize(pattern_field, (size, size))
        return pattern_field

    # Random crop
    y = np.random.randint(0, h - size + 1)
    x = np.random.randint(0, w - size + 1)

    patch = pattern_field[y:y+size, x:x+size]
    return patch

def is_interesting_patch(patch, min_intensity=0.1):
    """Filter out boring/blank patches"""
    intensity_mean = patch[:, :, 0].mean()
    intensity_std = patch[:, :, 0].std()
    return intensity_mean > min_intensity and intensity_std > 0.05

def preprocess_dataset():
    """Main preprocessing function"""
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    all_patches = []
    metadata = []

    print(f"Starting preprocessing...")
    print(f"Categories: {PATTERN_CATEGORIES}")

    total_processed = 0
    total_accepted = 0

    for category in PATTERN_CATEGORIES:
        category_path = Path(DATASET_PATH) / category

        if not category_path.exists():
            print(f"⚠️  Category not found: {category}")
            continue

        image_files = list(category_path.glob("*.jpg"))
        print(f"\n📁 Processing {category}: {len(image_files)} images")

        category_patches = 0

        for img_path in image_files:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                # Extract pattern field
                pattern_field = extract_pattern_field(image)

                # Extract multiple random patches
                for _ in range(PATCHES_PER_IMAGE):
                    patch = extract_random_patch(pattern_field, size=PATCH_SIZE)

                    # Filter interesting patches
                    if is_interesting_patch(patch, min_intensity=MIN_INTENSITY):
                        all_patches.append(patch)
                        metadata.append({
                            'category': category,
                            'source_image': img_path.name,
                            'intensity_mean': float(patch[:, :, 0].mean()),
                            'intensity_std': float(patch[:, :, 0].std())
                        })
                        category_patches += 1
                        total_accepted += 1

                    total_processed += 1

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        print(f"  ✓ Extracted {category_patches} patches")

    # Convert to numpy array
    dataset = np.array(all_patches, dtype=np.float32)

    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Total processed: {total_processed}")
    print(f"Accepted patches: {total_accepted}")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Memory size: {dataset.nbytes / 1024 / 1024:.2f} MB")

    # Save dataset
    np.save(os.path.join(OUTPUT_PATH, "patterns_dataset.npy"), dataset)

    # Save metadata
    with open(os.path.join(OUTPUT_PATH, "metadata.json"), 'w') as f:
        json.dump({
            'total_samples': len(dataset),
            'patch_size': PATCH_SIZE,
            'categories': PATTERN_CATEGORIES,
            'samples': metadata
        }, f, indent=2)

    print(f"\n✓ Saved to: {OUTPUT_PATH}")

    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Mean intensity: {dataset[:, :, :, 0].mean():.3f}")
    print(f"  Std intensity: {dataset[:, :, :, 0].std():.3f}")
    print(f"  Min intensity: {dataset[:, :, :, 0].min():.3f}")
    print(f"  Max intensity: {dataset[:, :, :, 0].max():.3f}")

    return dataset, metadata

if __name__ == "__main__":
    dataset, metadata = preprocess_dataset()
