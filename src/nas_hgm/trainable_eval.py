"""
Trainable Evaluation - Combines Compression + Zero-Shot Generalization
This is the TRUE objective: architectures that compress AND learn well!
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time


# ============================================================================
# Fast Data Loader (Cached)
# ============================================================================

_cached_loader = None

def get_trainability_data(batch_size=100):
    """Get cached small dataset for trainability tests"""
    global _cached_loader

    if _cached_loader is not None:
        return _cached_loader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )

    # Use 500 samples (fast but representative)
    indices = torch.randperm(len(full_dataset))[:500]
    subset = Subset(full_dataset, indices)

    _cached_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return _cached_loader


# ============================================================================
# Simple Classifier Wrapper
# ============================================================================

class QuickClassifier(nn.Module):
    """Minimal wrapper for trainability testing"""

    def __init__(self, backbone, input_size=784, num_classes=10, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.backbone = backbone
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_proj(x).unsqueeze(1)
        x = self.backbone(x).squeeze(1)
        return self.output(x)


# ============================================================================
# Fast Trainability Metrics
# ============================================================================

def measure_gradient_snr_fast(model, data_loader, device):
    """
    Gradient Signal-to-Noise Ratio (fast version)
    Higher = better trainability

    ~1 second for 5 batches
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    all_grads = []

    for i, (data, target) in enumerate(data_loader):
        if i >= 5:  # Only 5 batches for speed
            break

        data, target = data.to(device), target.to(device)

        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Collect gradients
        grad_vec = torch.cat([p.grad.flatten() for p in model.parameters()
                             if p.grad is not None])
        all_grads.append(grad_vec.cpu().detach().numpy())

    if len(all_grads) == 0:
        return 0.0

    all_grads = np.array(all_grads)

    # SNR = mean / std
    mean_grad = np.abs(all_grads.mean(axis=0)).mean()
    std_grad = all_grads.std(axis=0).mean()

    snr = mean_grad / (std_grad + 1e-8)

    return float(snr)


def measure_learning_speed(model, data_loader, device, lr=0.01):
    """
    How quickly does model learn after one gradient step?
    Higher improvement = better trainability

    ~0.5 seconds
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Get first batch
    try:
        data, target = next(iter(data_loader))
    except StopIteration:
        return 0.0, 0.0

    data, target = data.to(device), target.to(device)

    # Measure initial accuracy
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        acc_before = pred.eq(target).float().mean().item()

    # One gradient step
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Measure new accuracy
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1)
        acc_after = pred.eq(target).float().mean().item()

    improvement = acc_after - acc_before

    return acc_after, improvement


def measure_feature_separability_fast(model, data_loader, device):
    """
    Linear separability of learned features
    Higher = easier to learn

    ~0.5 seconds
    """
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= 5:  # Only 5 batches
                break

            data = data.to(device)

            # Get features before output layer
            x = data.view(data.size(0), -1)
            x = model.input_proj(x).unsqueeze(1)
            x = model.backbone(x).squeeze(1)

            features.append(x.cpu())
            labels.append(target)

    if len(features) == 0:
        return 0.0

    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    # Fisher criterion (simple version)
    unique_classes = np.unique(labels)

    within_var = 0
    between_var = 0
    global_mean = features.mean(axis=0)

    for c in unique_classes:
        class_features = features[labels == c]
        if len(class_features) == 0:
            continue

        class_mean = class_features.mean(axis=0)

        # Within-class variance
        within_var += ((class_features - class_mean) ** 2).sum()

        # Between-class variance
        between_var += len(class_features) * ((class_mean - global_mean) ** 2).sum()

    separability = between_var / (within_var + 1e-8)

    return float(separability)


# ============================================================================
# Combined Trainability Score
# ============================================================================

def evaluate_trainability(backbone, device='cuda', verbose=False):
    """
    Fast trainability evaluation (2-3 seconds total)

    Returns dict with trainability metrics
    """
    start_time = time.time()

    # Get data
    data_loader = get_trainability_data()

    # Wrap backbone with classifier
    model = QuickClassifier(backbone, hidden_dim=64).to(device)

    results = {}

    try:
        # Test 1: Gradient SNR (~1s)
        snr = measure_gradient_snr_fast(model, data_loader, device)
        results['gradient_snr'] = snr

        # Test 2: Learning speed (~0.5s)
        one_step_acc, improvement = measure_learning_speed(model, data_loader, device)
        results['one_step_accuracy'] = one_step_acc
        results['learning_speed'] = improvement

        # Test 3: Feature separability (~0.5s)
        separability = measure_feature_separability_fast(model, data_loader, device)
        results['feature_separability'] = separability

        # Combined trainability score
        trainability_score = (
            0.4 * np.tanh(snr) +                          # Gradient quality
            0.4 * improvement * 5 +                       # Learning speed (scaled)
            0.2 * np.tanh(separability / 100)            # Feature quality
        )

        results['trainability_score'] = float(np.clip(trainability_score, 0, 1))

    except Exception as e:
        if verbose:
            print(f"Trainability eval failed: {e}")
        results = {
            'gradient_snr': 0.0,
            'one_step_accuracy': 0.0,
            'learning_speed': 0.0,
            'feature_separability': 0.0,
            'trainability_score': 0.0,
        }

    results['trainability_time_ms'] = (time.time() - start_time) * 1000

    # Import sanitize function from compression_eval
    from .compression_eval import sanitize_metrics
    return sanitize_metrics(results)


# ============================================================================
# Main Function for Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Trainability Evaluation System...\n")

    from .architectures import create_transformer_baseline, ArchitectureWrapper

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Test transformer
    spec = create_transformer_baseline(dim=64, heads=4, layers=1)
    model = ArchitectureWrapper(spec).to(device)

    print("Evaluating Transformer baseline...")
    results = evaluate_trainability(model, device, verbose=True)

    print("\nResults:")
    print(f"  Gradient SNR: {results['gradient_snr']:.3f}")
    print(f"  Learning speed: {results['learning_speed']:.3f}")
    print(f"  Feature separability: {results['feature_separability']:.3f}")
    print(f"  Trainability score: {results['trainability_score']:.3f}")
    print(f"  Time: {results['trainability_time_ms']:.1f}ms")

    print("\n✅ Trainability evaluation ready!")
