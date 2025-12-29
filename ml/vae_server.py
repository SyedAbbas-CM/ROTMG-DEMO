"""
VAE Pattern Server - Realtime pattern generation via HTTP

Exposes the trained VAE v2 decoder for live pattern generation.
The game client can call this to generate infinite unique patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ==============================================================================
# VAE Architecture (must match train_vae_v2.py exactly)
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, in_channels=8, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.global_pool(x).view(x.size(0), -1)
        x = F.leaky_relu(self.fc(x), 0.2)
        return self.fc_mean(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=8, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 256 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)

    def forward(self, z):
        x = F.leaky_relu(self.fc1(z), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = x.view(x.size(0), 256, 2, 2)
        x = F.leaky_relu(self.bn1(self.deconv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.deconv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.deconv3(x)), 0.2)
        x = torch.sigmoid(self.deconv4(x))
        return x


class VAE(nn.Module):
    def __init__(self, channels=8, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(channels, latent_dim)
        self.decoder = Decoder(channels, latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar

    def generate(self, n_samples, device):
        z = torch.randn(n_samples, self.latent_dim).to(device)
        return self.decoder(z)


# ==============================================================================
# Global State
# ==============================================================================
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64

# Store some latent vectors for interpolation
stored_latents = {}


def load_model():
    global model

    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_files = [f for f in os.listdir(model_dir) if f.startswith('vae_v2_best')]

    if not model_files:
        print("[VAE Server] No v2 model found!")
        return False

    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    print(f"[VAE Server] Loading model: {model_path}")

    model = VAE(channels=8, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"[VAE Server] Model loaded on {device}")
    return True


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None, 'latent_dim': LATENT_DIM})


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate a pattern from latent vector.

    POST body:
    {
        "latent": [64 floats] or null for random,
        "seed": optional int for reproducible random,
        "scale": optional float to scale latent magnitude (default 1.0)
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json or {}

    if 'latent' in data and data['latent'] is not None:
        z = torch.tensor(data['latent'], dtype=torch.float32).unsqueeze(0).to(device)
    else:
        if 'seed' in data:
            torch.manual_seed(data['seed'])
        z = torch.randn(1, LATENT_DIM).to(device)

    # Optional scaling
    scale = data.get('scale', 1.0)
    z = z * scale

    with torch.no_grad():
        pattern = model.decoder(z)

    pattern_np = pattern[0].cpu().numpy()  # [8, 32, 32]
    pattern_np = np.transpose(pattern_np, (1, 2, 0))  # [32, 32, 8]

    channel_names = ['spawn', 'direction', 'size', 'speed', 'accel', 'curve', 'wave_amp', 'wave_freq']
    stats = {}
    for i, name in enumerate(channel_names):
        ch = pattern_np[:, :, i]
        stats[name] = {
            'min': float(ch.min()),
            'max': float(ch.max()),
            'mean': float(ch.mean()),
            'std': float(ch.std())
        }

    return jsonify({
        'pattern': pattern_np.tolist(),
        'latent': z[0].cpu().tolist(),
        'stats': stats
    })


@app.route('/generate_simple', methods=['GET'])
def generate_simple():
    """Simple GET endpoint for quick testing. Returns just the pattern."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    seed = request.args.get('seed', type=int)
    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(1, LATENT_DIM).to(device)

    with torch.no_grad():
        pattern = model.decoder(z)

    pattern_np = pattern[0].cpu().numpy()
    pattern_np = np.transpose(pattern_np, (1, 2, 0))

    return jsonify({
        'pattern': pattern_np.tolist(),
        'latent': z[0].cpu().tolist()
    })


@app.route('/interpolate', methods=['POST'])
def interpolate():
    """
    Interpolate between two latent vectors.

    POST body:
    {
        "latent_a": [64 floats],
        "latent_b": [64 floats],
        "t": float 0-1 (0=A, 1=B)
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json

    z_a = torch.tensor(data['latent_a'], dtype=torch.float32).to(device)
    z_b = torch.tensor(data['latent_b'], dtype=torch.float32).to(device)
    t = data.get('t', 0.5)

    z = (1 - t) * z_a + t * z_b
    z = z.unsqueeze(0)

    with torch.no_grad():
        pattern = model.decoder(z)

    pattern_np = pattern[0].cpu().numpy()
    pattern_np = np.transpose(pattern_np, (1, 2, 0))

    return jsonify({
        'pattern': pattern_np.tolist(),
        'latent': z[0].cpu().tolist()
    })


@app.route('/store_latent', methods=['POST'])
def store_latent():
    """Store a latent vector with a name for later use."""
    data = request.json
    name = data.get('name', 'default')
    latent = data.get('latent')

    if latent is None:
        # Generate random and store
        latent = torch.randn(LATENT_DIM).tolist()

    stored_latents[name] = latent
    return jsonify({'stored': name, 'latent': latent})


@app.route('/get_latent/<name>', methods=['GET'])
def get_latent(name):
    """Get a stored latent vector."""
    if name in stored_latents:
        return jsonify({'name': name, 'latent': stored_latents[name]})
    return jsonify({'error': 'Not found'}), 404


@app.route('/random_batch', methods=['GET'])
def random_batch():
    """Generate a batch of random patterns."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    count = min(int(request.args.get('count', 10)), 50)

    with torch.no_grad():
        z = torch.randn(count, LATENT_DIM).to(device)
        patterns = model.decoder(z)

    patterns_np = patterns.cpu().numpy()
    patterns_np = np.transpose(patterns_np, (0, 2, 3, 1))

    results = []
    for i in range(count):
        results.append({
            'pattern': patterns_np[i].tolist(),
            'latent': z[i].cpu().tolist()
        })

    return jsonify({'patterns': results})


@app.route('/modify_latent', methods=['POST'])
def modify_latent():
    """
    Modify specific dimensions of a latent vector.

    POST body:
    {
        "latent": [64 floats] or null to use random base,
        "modifications": {
            "0": 2.0,  // Set dimension 0 to 2.0
            "5": -1.5  // Set dimension 5 to -1.5
        }
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.json

    if data.get('latent'):
        z = torch.tensor(data['latent'], dtype=torch.float32)
    else:
        z = torch.randn(LATENT_DIM)

    modifications = data.get('modifications', {})
    for dim_str, value in modifications.items():
        dim = int(dim_str)
        if 0 <= dim < LATENT_DIM:
            z[dim] = value

    z = z.unsqueeze(0).to(device)

    with torch.no_grad():
        pattern = model.decoder(z)

    pattern_np = pattern[0].cpu().numpy()
    pattern_np = np.transpose(pattern_np, (1, 2, 0))

    return jsonify({
        'pattern': pattern_np.tolist(),
        'latent': z[0].cpu().tolist()
    })


if __name__ == '__main__':
    if load_model():
        print("[VAE Server] Starting on http://localhost:5000")
        print("[VAE Server] Endpoints:")
        print("  GET  /health - Check server status")
        print("  POST /generate - Generate pattern from latent")
        print("  GET  /generate_simple?seed=N - Quick random pattern")
        print("  POST /interpolate - Interpolate between patterns")
        print("  POST /modify_latent - Modify latent dimensions")
        print("  GET  /random_batch?count=N - Generate batch")
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    else:
        print("[VAE Server] Failed to load model")
