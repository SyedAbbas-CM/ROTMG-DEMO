"""
Create Extreme Test Patterns to Verify Visual Variety Pipeline
"""

import numpy as np
import json

print("Creating test patterns with EXTREME values...")

patterns = []

# Pattern 1: ALL LARGE SLOW bullets (high intensity everywhere)
print("\n1. Large Slow Pattern (high intensity)")
p1_intensity = np.ones((32, 32)) * 0.9  # Max intensity
p1_direction = np.random.rand(32, 32)
patterns.append({
    "id": 0,
    "name": "test_large_slow",
    "intensity": p1_intensity.tolist(),
    "direction": p1_direction.tolist(),
    "latent_vector": [0] * 32,
    "stats": {
        "density_50": 1.0,  # All pixels > 0.5
        "mean_intensity": 0.9,
        "std_intensity": 0.0
    }
})

# Pattern 2: ALL SMALL FAST bullets (low intensity everywhere)
print("2. Small Fast Pattern (low intensity)")
p2_intensity = np.ones((32, 32)) * 0.1  # Min intensity
p2_direction = np.random.rand(32, 32)
patterns.append({
    "id": 1,
    "name": "test_small_fast",
    "intensity": p2_intensity.tolist(),
    "direction": p2_direction.tolist(),
    "latent_vector": [0] * 32,
    "stats": {
        "density_50": 0.0,  # No pixels > 0.5
        "mean_intensity": 0.1,
        "std_intensity": 0.0
    }
})

# Pattern 3: HALF AND HALF (top=large/slow, bottom=small/fast)
print("3. Split Pattern (mixed)")
p3_intensity = np.zeros((32, 32))
p3_intensity[0:16, :] = 0.9  # Top half: large slow
p3_intensity[16:32, :] = 0.1  # Bottom half: small fast
p3_direction = np.random.rand(32, 32)
patterns.append({
    "id": 2,
    "name": "test_split",
    "intensity": p3_intensity.tolist(),
    "direction": p3_direction.tolist(),
    "latent_vector": [0] * 32,
    "stats": {
        "density_50": 0.5,  # Half pixels > 0.5
        "mean_intensity": 0.5,
        "std_intensity": 0.4
    }
})

# Pattern 4: RING (center low, edges high)
print("4. Ring Pattern (radial gradient)")
p4_intensity = np.zeros((32, 32))
center = np.array([16, 16])
for i in range(32):
    for j in range(32):
        dist = np.sqrt((i-16)**2 + (j-16)**2)
        # Ring at radius 10
        if 8 < dist < 14:
            p4_intensity[i, j] = 0.9  # High intensity ring
        else:
            p4_intensity[i, j] = 0.1  # Low intensity elsewhere
p4_direction = np.random.rand(32, 32)
patterns.append({
    "id": 3,
    "name": "test_ring",
    "intensity": p4_intensity.tolist(),
    "direction": p4_direction.tolist(),
    "latent_vector": [0] * 32,
    "stats": {
        "density_50": 0.3,  # Ring area
        "mean_intensity": 0.4,
        "std_intensity": 0.35
    }
})

# Pattern 5: CROSS (cardinal directions only)
print("5. Cross Pattern (4 directions)")
p5_intensity = np.ones((32, 32)) * 0.1  # Default low
p5_intensity[14:18, :] = 0.9  # Horizontal bar
p5_intensity[:, 14:18] = 0.9  # Vertical bar
p5_direction = np.zeros((32, 32))
# Set directions: up/down/left/right
p5_direction[14:18, 16:32] = 0.0    # Right
p5_direction[14:18, 0:16] = 0.5     # Left
p5_direction[0:16, 14:18] = 0.75    # Up
p5_direction[16:32, 14:18] = 0.25   # Down
patterns.append({
    "id": 4,
    "name": "test_cross",
    "intensity": p5_intensity.tolist(),
    "direction": p5_direction.tolist(),
    "latent_vector": [0] * 32,
    "stats": {
        "density_50": 0.25,  # Cross area
        "mean_intensity": 0.4,
        "std_intensity": 0.4
    }
})

# Save test patterns
output = {
    "pattern_count": len(patterns),
    "resolution": [32, 32],
    "latent_dim": 32,
    "patterns": patterns,
    "test_mode": True,
    "description": "Extreme test patterns to verify visual variety works"
}

with open('visualizations/pattern_library_TEST.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Created {len(patterns)} test patterns")
print("Saved to: visualizations/pattern_library_TEST.json")

# Print expected behaviors
print("\n" + "="*60)
print("EXPECTED BEHAVIORS:")
print("="*60)
print("1. Large Slow: All bullets should be BIG and SLOW")
print("2. Small Fast: All bullets should be TINY and FAST")
print("3. Split: Top bullets BIG/SLOW, bottom bullets TINY/FAST")
print("4. Ring: Ring of BIG/SLOW bullets, center/edges TINY/FAST")
print("5. Cross: BIG bullets in + shape, pointing outward")
print("="*60)
print("\nTO TEST:")
print("1. Backup current pattern library:")
print("   mv visualizations/pattern_library.json visualizations/pattern_library_BACKUP.json")
print("2. Use test patterns:")
print("   mv visualizations/pattern_library_TEST.json visualizations/pattern_library.json")
print("3. Restart server and test in-game")
print("4. If patterns look the SAME: Visual variety is BROKEN")
print("5. If patterns look DIFFERENT: Visual variety WORKS!")
