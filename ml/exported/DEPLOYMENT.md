
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
