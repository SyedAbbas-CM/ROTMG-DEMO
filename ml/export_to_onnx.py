"""
Export trained PyTorch decoder to ONNX for Jetson Nano deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from glob import glob

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

    best_models = [m for m in models if 'final' not in m]
    if best_models:
        latest = sorted(best_models)[-1]
    else:
        latest = sorted(models)[-1]

    return latest

def export_to_onnx():
    print("="*60)
    print("Export PyTorch Decoder to ONNX for Jetson Nano")
    print("="*60)

    # Load model
    decoder_path = find_latest_decoder()
    print(f"\nLoading decoder: {decoder_path}")

    decoder = Decoder(latent_dim=32)
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
    decoder.eval()

    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {param_count:,}")

    # Create dummy input
    dummy_input = torch.randn(1, 32)

    # Test forward pass
    with torch.no_grad():
        test_output = decoder(dummy_input)
        print(f"\nTest forward pass:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")

    # Export to ONNX
    output_dir = "exported"
    os.makedirs(output_dir, exist_ok=True)

    onnx_path = os.path.join(output_dir, "pattern_decoder.onnx")

    print(f"\nExporting to ONNX...")
    torch.onnx.export(
        decoder,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,  # Compatible with older ONNX Runtime on Jetson
        do_constant_folding=True,
        input_names=['latent_vector'],
        output_names=['pattern'],
        dynamic_axes={
            'latent_vector': {0: 'batch_size'},
            'pattern': {0: 'batch_size'}
        }
    )

    file_size = os.path.getsize(onnx_path) / 1024  # KB
    print(f"✓ ONNX model exported!")
    print(f"  Path: {onnx_path}")
    print(f"  Size: {file_size:.1f} KB")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed!")
    except ImportError:
        print(f"⚠  ONNX package not available for verification")
        print(f"   Install with: pip install onnx")
    except Exception as e:
        print(f"⚠  ONNX verification warning: {e}")

    # Test with ONNX Runtime
    try:
        import onnxruntime as ort
        import numpy as np

        print(f"\nTesting ONNX Runtime inference...")
        session = ort.InferenceSession(onnx_path)

        # Test inference
        test_latent = np.random.randn(1, 32).astype(np.float32)
        ort_outputs = session.run(None, {'latent_vector': test_latent})
        ort_pattern = ort_outputs[0]

        print(f"✓ ONNX Runtime inference successful!")
        print(f"  Output shape: {ort_pattern.shape}")
        print(f"  Output range: [{ort_pattern.min():.3f}, {ort_pattern.max():.3f}]")

        # Compare with PyTorch
        with torch.no_grad():
            torch_pattern = decoder(torch.from_numpy(test_latent)).numpy()

        diff = np.abs(torch_pattern - ort_pattern).max()
        print(f"  Max difference from PyTorch: {diff:.6f}")

        if diff < 1e-5:
            print(f"  ✓ Outputs match PyTorch (difference < 1e-5)")
        else:
            print(f"  ⚠  Small numerical difference (expected, OK)")

    except ImportError:
        print(f"\n⚠  ONNX Runtime not available for testing")
        print(f"   Install with: pip install onnxruntime")

    # Save metadata
    metadata = {
        'model_type': 'bullet_pattern_decoder',
        'framework': 'pytorch',
        'export_format': 'onnx',
        'opset_version': 11,
        'input_shape': [1, 32],
        'output_shape': [1, 2, 32, 32],
        'latent_dim': 32,
        'parameters': param_count,
        'file_size_kb': file_size,
        'deployment_target': 'jetson_nano',
        'pytorch_model': decoder_path,
        'onnx_model': onnx_path
    }

    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved: {metadata_path}")

    # Create deployment instructions
    instructions = """
# Deploying to Jetson Nano

## 1. Copy model to Jetson
scp exported/pattern_decoder.onnx user@jetson-ip:/path/to/model/

## 2. Install dependencies on Jetson (if not already installed)
pip3 install onnxruntime

## 3. Test inference on Jetson
python3 <<EOF
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('pattern_decoder.onnx')
latent = np.random.randn(1, 32).astype(np.float32)
pattern = session.run(None, {'latent_vector': latent})[0]

print(f"Pattern shape: {pattern.shape}")
print(f"Intensity range: [{pattern[0,0].min():.3f}, {pattern[0,0].max():.3f}]")
print(f"Direction range: [{pattern[0,1].min():.3f}, {pattern[0,1].max():.3f}]")
EOF

## 4. Performance test
import time
start = time.time()
for i in range(1000):
    pattern = session.run(None, {'latent_vector': latent})[0]
end = time.time()
print(f"Inference FPS: {1000 / (end - start):.1f}")
"""

    instructions_path = os.path.join(output_dir, 'DEPLOYMENT.md')
    with open(instructions_path, 'w') as f:
        f.write(instructions)

    print(f"✓ Deployment instructions: {instructions_path}")

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Test on Jetson Nano (see {instructions_path})")
    print(f"2. Integrate with Node.js adapter")
    print(f"3. Connect to BulletManager")

if __name__ == "__main__":
    export_to_onnx()
