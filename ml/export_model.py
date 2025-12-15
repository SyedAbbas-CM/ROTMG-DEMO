"""
Export trained model to ONNX for Jetson Nano deployment
"""

import tensorflow as tf
import tf2onnx
import onnx
import os
import json
import numpy as np

MODEL_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/models"
OUTPUT_PATH = "/Users/az/Desktop/Rotmg-Pservers/ROTMG-DEMO/ml/exported"

def find_latest_model():
    """Find the most recent decoder model"""
    models = [f for f in os.listdir(MODEL_PATH) if f.startswith("pattern_decoder") and f.endswith(".h5")]
    if not models:
        raise FileNotFoundError("No decoder models found!")

    latest = sorted(models)[-1]
    return os.path.join(MODEL_PATH, latest)

def export_to_onnx(model_path):
    """Convert Keras model to ONNX"""
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print("\nModel summary:")
    model.summary()

    # Get input spec
    input_signature = [tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='latent_input')]

    print(f"\nInput shape: {model.inputs[0].shape}")
    print(f"Output shape: {model.outputs[0].shape}")

    # Convert to ONNX
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    onnx_path = os.path.join(OUTPUT_PATH, "pattern_decoder.onnx")

    print(f"\nConverting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

    # Save ONNX model
    onnx.save(onnx_model, onnx_path)

    # Check file size
    file_size = os.path.getsize(onnx_path) / 1024  # KB
    print(f"\n✓ ONNX model saved!")
    print(f"  Path: {onnx_path}")
    print(f"  Size: {file_size:.1f} KB")

    # Test inference
    print(f"\nTesting ONNX inference...")
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)

    # Random latent vector
    test_input = np.random.randn(1, 32).astype(np.float32)

    # Run inference
    outputs = session.run(None, {'latent_input': test_input})
    pattern = outputs[0]

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {pattern.shape}")
    print(f"  Output range: [{pattern.min():.3f}, {pattern.max():.3f}]")
    print(f"✓ ONNX inference works!")

    # Save metadata
    metadata = {
        'model_path': model_path,
        'onnx_path': onnx_path,
        'file_size_kb': file_size,
        'input_shape': [1, 32],
        'output_shape': [1, 32, 32, 2],
        'latent_dim': 32,
        'deployment': 'jetson_nano'
    }

    with open(os.path.join(OUTPUT_PATH, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    return onnx_path

if __name__ == "__main__":
    print("="*60)
    print("Model Export to ONNX for Jetson Nano")
    print("="*60)

    try:
        model_path = find_latest_model()
        onnx_path = export_to_onnx(model_path)
        print(f"\n{'='*60}")
        print(f"Export complete! Ready for Jetson Nano deployment")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
