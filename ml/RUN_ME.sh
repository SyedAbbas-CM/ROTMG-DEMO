#!/bin/bash

echo "=========================================="
echo "Bullet Pattern AI - Training Pipeline"
echo "=========================================="

# Check if in correct directory
cd "$(dirname "$0")"

echo ""
echo "Step 1: Install dependencies"
echo "----------------------------"
pip3 install --user --break-system-packages tensorflow-macos==2.13.0 tensorflow-metal==1.0.1 numpy opencv-python matplotlib tf2onnx onnx onnxruntime || {
    echo "Failed to install dependencies. Try manually:"
    echo "pip3 install --user tensorflow-macos tensorflow-metal numpy opencv-python matplotlib"
    exit 1
}

echo ""
echo "Step 2: Preprocess dataset"
echo "----------------------------"
python3 preprocess_patterns.py || {
    echo "Preprocessing failed!"
    exit 1
}

echo ""
echo "Step 3: Train VAE model"
echo "----------------------------"
python3 train_pattern_vae.py || {
    echo "Training failed!"
    exit 1
}

echo ""
echo "Step 4: Visualize patterns"
echo "----------------------------"
python3 visualize_patterns.py || {
    echo "Visualization failed!"
    exit 1
}

echo ""
echo "Step 5: Export to ONNX"
echo "----------------------------"
python3 export_model.py || {
    echo "Export failed!"
    exit 1
}

echo ""
echo "=========================================="
echo "✓ Training pipeline complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check visualizations in: ./visualizations/"
echo "2. Deploy ONNX model to Jetson Nano: ./exported/pattern_decoder.onnx"
echo "3. Integrate with BulletManager (Node.js adapter coming next)"
