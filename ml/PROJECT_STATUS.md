# Bullet Pattern AI - Project Status

## Overview
We're building an AI system to generate organic, nature-inspired bullet patterns for a bullet-hell game (Realm of the Mad God). The system uses a lightweight VAE (Variational Autoencoder) trained on texture patterns to generate diverse bullet attack patterns that can be deployed on embedded hardware (Jetson Nano) or run locally.

## Hardware Tested
- **Jetson Nano**: NVIDIA Tegra X1, 4GB RAM, CUDA 10.2, TensorRT 8.2.1
  - Confirmed PyTorch 1.10.0 working with GPU acceleration
  - Can achieve 843 FPS on small CNNs (1,392 params)
  - Can handle 2,810 FPS with batch size 4
  - Memory limit: Keep models < 100K parameters
  - Status: ✅ Online and ready for deployment testing

- **M1 MacBook Pro**: Training environment
  - Will use TensorFlow Metal for GPU acceleration
  - Target model size: 50-100K parameters for Jetson compatibility

## Dataset
- **Source**: DTD (Describable Textures Dataset)
- **Location**: `/Users/az/Desktop/Rotmg-Pservers/dtd/images`
- **Total images**: 5,640 across 47 categories
- **Relevant categories for bullet patterns** (120 images each):
  - cracked, spiralled, swirly, veined, zigzagged
  - cobwebbed, woven, dotted, braided, marbled
  - bumpy, crystalline, fibrous, flecked, frilly

## Model Architecture

### Input/Output Format
- **Model Input**: Latent vector `[1, 32]` (random seed)
- **Model Output**: Pattern field `[32, 32, 2]`
  - Channel 0: Spawn intensity (0-1) - determines if/how strongly a bullet spawns
  - Channel 1: Direction (0-1 mapped to 0-2π) - defines bullet travel angle

### VAE Design
- **Encoder** (training only):
  - Conv2D layers: 16 → 32 → 64 filters
  - Downsamples 32×32 → 16×16 → 8×8 → 4×4
  - Dense to latent space: 32 dimensions

- **Decoder** (deployed for inference):
  - Dense from latent 32 → 4×4×64
  - Conv2DTranspose layers: 64 → 32 → 16 filters
  - Upsamples 4×4 → 8×8 → 16×16 → 32×32
  - Output: 2 channels (sigmoid activation)
  - **Target size**: ~50K parameters, <200KB model file

## Files Created

### Core Training Pipeline
1. **preprocess_patterns.py** - Extract pattern fields from textures
   - Applies Sobel edge detection
   - Extracts magnitude (intensity) and angle (direction)
   - Creates random 32×32 patches
   - Filters interesting patterns
   - Output: `patterns_dataset.npy`

2. **train_pattern_vae.py** - Train the VAE model
   - Lightweight architecture optimized for Jetson Nano
   - Adam optimizer, learning rate 1e-3
   - 100 epochs with early stopping
   - Saves decoder separately for deployment
   - Output: `pattern_decoder_TIMESTAMP.h5`

3. **visualize_patterns.py** - Visualize generated patterns
   - Creates pattern grids
   - Detailed single pattern visualization with vector fields
   - Exports patterns to JSON
   - Output: PNG visualizations + `pattern_library.json`

4. **export_model.py** - Export to ONNX for Jetson Nano
   - Converts Keras decoder to ONNX format
   - Tests inference
   - Output: `pattern_decoder.onnx`

### Utilities
- **requirements.txt** - Python dependencies
- **RUN_ME.sh** - Complete automated pipeline
- **PROJECT_STATUS.md** - This file

## Workflow

### Phase 1: Training (M1 MacBook) [CURRENT]
```bash
cd /Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml

# Install dependencies
pip3 install --user --break-system-packages tensorflow-macos tensorflow-metal numpy opencv-python matplotlib tf2onnx onnx onnxruntime

# Run pipeline
./RUN_ME.sh
```

Or manually:
```bash
python3 preprocess_patterns.py   # ~10-15K training samples
python3 train_pattern_vae.py     # 100 epochs, ~10-20 min
python3 visualize_patterns.py    # Generate sample patterns
python3 export_model.py          # Convert to ONNX
```

### Phase 2: Deployment (Jetson Nano) [TODO]
1. Copy `pattern_decoder.onnx` to Jetson Nano
2. Test inference with ONNXRuntime
3. Measure FPS and memory usage
4. Generate pattern library (500-1000 patterns offline)

### Phase 3: Game Integration [TODO]
1. Create Node.js adapter to convert pattern → bullets
2. Integrate with BulletManager
3. Add to boss AI system
4. Test in gameplay

## Pattern → Bullet Conversion Logic

For each cell (x,y) in the 32×32 pattern:

```javascript
if (intensity[y][x] < threshold) continue;

// World position
spawnX = boss.x + mapRange(x, 0, 31, -radius, radius);
spawnY = boss.y + mapRange(y, 0, 31, -radius, radius);

// Velocity from direction
angle = direction[y][x] * 2 * Math.PI;
speed = baseSpeed * intensityScale(intensity[y][x]);
vx = Math.cos(angle) * speed;
vy = Math.sin(angle) * speed;

BulletManager.addBullet({
  x: spawnX,
  y: spawnY,
  vx: vx,
  vy: vy,
  bulletType: BULLET_TYPES.NORMAL,
  damage: 10,
  ownerId: boss.id,
  worldId: boss.worldId
});
```

### Enhanced Properties (Inferred from Pattern)
- **Blob size** → Bigger bullet sprites
- **Dense clusters** → Lower damage, more bullets
- **Sparse areas** → High damage single shots
- **Turbulent directions** → Wobble/homing behavior
- **Radial patterns** → Acceleration effects

## Next Immediate Steps

1. **Run preprocessing** ✓ (script ready)
2. **Train model** ✓ (script ready)
3. **Visualize results** ✓ (script ready)
4. **Export ONNX** ✓ (script ready)
5. **Test on Jetson Nano** (after training)
6. **Create Node.js adapter** (after validation)
7. **Integrate with game** (final step)

## Technical Notes

### Why This Design Works
- Small model (50K params) = fast inference on Jetson
- VAE latent space = controllable generation
- Pattern field representation = flexible bullet properties
- Offline generation = no runtime ML overhead during gameplay
- Gradient-based patterns = natural-looking bullet formations

### Performance Expectations
- **Training**: ~10-20 minutes on M1
- **Inference (Jetson)**: ~500-1000 FPS expected
- **Pattern generation**: Can generate 1000 patterns in <1 second
- **Game integration**: Zero gameplay lag (pre-generated patterns)

## Questions for ChatGPT / Collaboration

1. Should we add data augmentation during training (rotation, flip)?
2. Do we need TensorRT quantization for Jetson, or is ONNX enough?
3. Should we test alternative architectures (smaller encoder)?
4. How to best parameterize pattern "style" (phase 1 vs phase 3)?

## Files Structure

```
ROTMG-DEMO/
├── ml/
│   ├── preprocess_patterns.py      ✅ Created
│   ├── train_pattern_vae.py        ✅ Created
│   ├── visualize_patterns.py       ✅ Created
│   ├── export_model.py             ✅ Created
│   ├── requirements.txt            ✅ Created
│   ├── RUN_ME.sh                   ✅ Created
│   ├── PROJECT_STATUS.md           ✅ Created (this file)
│   ├── patterns_dataset/           🔜 Will be created
│   ├── models/                     🔜 Will be created
│   ├── visualizations/             🔜 Will be created
│   └── exported/                   🔜 Will be created
├── dtd/                            ✅ Dataset present
│   └── images/                     ✅ 5,640 images
└── src/                            (game code)
    └── world/
        └── BulletManager.js        (integration target)
```

## Status: Ready to Execute Training Pipeline 🚀

All scripts are written and ready. Next command to run:

```bash
cd /Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml
./RUN_ME.sh
```

Or for ChatGPT to help with any improvements/testing before running.
