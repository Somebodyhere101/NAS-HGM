"""
Zero-Training Architecture Evaluation via Compression Geometry
Measures architecture quality without any gradient descent.
Based on Paper 2: Generalization = Compression
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import hashlib


def sanitize_metrics(metrics):
    """
    Convert all metric values to JSON-serializable Python types.
    Handles torch tensors, numpy types, and other non-serializable objects.
    """
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            sanitized[key] = float(value.item() if value.numel() == 1 else value.mean().item())
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, np.ndarray):
            sanitized[key] = float(value.mean()) if value.size > 1 else float(value)
        elif isinstance(value, (int, float, str, bool, type(None))):
            sanitized[key] = value
        else:
            # Try to convert to float, fallback to string
            try:
                sanitized[key] = float(value)
            except:
                sanitized[key] = str(value)
    return sanitized


def spec_to_seed(spec):
    """
    Create deterministic seed from architecture spec.
    Same architecture → same seed → reproducible evaluation.
    """
    # Convert spec to JSON string (deterministic ordering)
    spec_str = json.dumps(spec, sort_keys=True)
    # Hash to integer seed
    hash_obj = hashlib.md5(spec_str.encode())
    seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)
    return seed


def generate_structured_data(batch_size=32, seq_len=16, dim=64, rank=4, seed=42):
    """
    Generate synthetic data with known low-rank structure.
    This simulates real-world data manifolds.
    """
    # Fix seed for deterministic evaluation
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Create rank-k data
    U = torch.randn(batch_size, seq_len, rank, generator=generator)
    V = torch.randn(rank, dim, generator=generator)

    # Low-rank data
    X = U @ V

    # Add small noise
    noise = 0.1 * torch.randn(batch_size, seq_len, dim, generator=generator)
    X = X + noise

    return X


def inject_compressed_patterns(model, rank=3, seed=42):
    """
    Inject low-rank patterns into model weights (Paper 2 method).
    This simulates the result of training without actually training.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # Get weight matrix
            W = param.data

            # SVD decomposition
            try:
                U, S, Vh = torch.svd(W)

                # Keep only top-k singular values
                S_compressed = torch.zeros_like(S)
                S_compressed[:rank] = S[:rank]

                # Reconstruct with low rank
                W_compressed = U @ torch.diag(S_compressed) @ Vh.t()

                # Inject compressed pattern
                param.data = W_compressed
            except:
                # If SVD fails, initialize low-rank directly
                m, n = W.shape
                k = min(rank, m, n)
                param.data = torch.randn(m, k, generator=generator) @ torch.randn(k, n, generator=generator) * 0.1


def measure_rank(tensor, threshold=0.95):
    """
    Compute effective rank using singular value energy.
    Returns: (effective_rank, compression_ratio, singular_values)
    """
    if tensor.dim() > 2:
        # Flatten to 2D for SVD
        orig_shape = tensor.shape
        tensor = tensor.reshape(orig_shape[0], -1)

    try:
        _, S, _ = torch.svd(tensor)

        # Effective rank: number of singular values capturing 95% energy
        total_energy = S.sum()
        cumulative = torch.cumsum(S, dim=0)
        effective_rank = (cumulative < threshold * total_energy).sum().item() + 1

        # Compression ratio
        compression = 1 - (effective_rank / len(S))

        return effective_rank, compression, S
    except:
        # Fallback if SVD fails
        return tensor.shape[-1], 0.0, torch.ones(tensor.shape[-1])


def measure_entropy(tensor):
    """
    Compute Shannon entropy of normalized tensor values.
    Lower entropy = more structured/compressed.
    """
    # Flatten and normalize to probability distribution
    flat = tensor.flatten().abs()
    probs = flat / (flat.sum() + 1e-10)

    # Shannon entropy
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

    # Normalize by max possible entropy
    max_entropy = np.log(len(flat))

    return entropy / max_entropy if max_entropy > 0 else 1.0


def check_bottleneck_exists(model):
    """
    Check if model has mandatory low-rank bottlenecks.
    Paper 2 finding: Bottlenecks are essential for compression.
    """
    bottleneck_count = 0
    total_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_layers += 1

            # Check if layer is a bottleneck (output_dim < input_dim)
            if module.out_features < module.in_features * 0.5:
                bottleneck_count += 1

    return bottleneck_count, total_layers


def trace_information_flow(model, x):
    """
    Track how information compresses through network layers.
    Returns compression at each layer.
    """
    compressions = []
    activations = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            rank, compression, _ = measure_rank(output)
            compressions.append(compression)
            activations.append(output.detach())

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        model(x)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return compressions, activations


def evaluate_architecture_fast(model, inject_rank=3, data_rank=4, seed=42, include_trainability=True):
    """
    Fast zero-training evaluation of architecture quality.

    Returns dict with:
        - compression_score: Overall compression capability
        - trainability_score: How well it will train (NEW!)
        - combined_score: Balanced objective (compression + trainability)
        - rank_reduction: Input rank → output rank ratio
        - bottleneck_score: Architectural bottleneck quality
        - flow_efficiency: How well compression propagates
        - speed: Inference time (ms)
    """
    start_time = time.time()

    # Get device from model
    device = next(model.parameters()).device

    # Set torch seed for deterministic initialization
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 1. Inject compressed patterns (simulate trained state)
    inject_compressed_patterns(model, rank=inject_rank, seed=seed)

    # 2. Generate structured data
    x = generate_structured_data(batch_size=8, rank=data_rank, seed=seed).to(device)

    # 3. Forward pass (no gradients)
    with torch.no_grad():
        output = model(x)

    # 4. Measure output compression
    input_rank, input_compression, _ = measure_rank(x)
    output_rank, output_compression, _ = measure_rank(output)

    # Rank reduction (clipped to [0, 1] - negative means expansion, not compression)
    rank_reduction = max(0.0, 1.0 - (output_rank / (input_rank + 1e-8)))

    # 5. Check architectural properties
    num_bottlenecks, total_layers = check_bottleneck_exists(model)
    bottleneck_score = num_bottlenecks / (total_layers + 1e-8)

    # 6. Trace compression flow
    compressions, _ = trace_information_flow(model, x)
    flow_efficiency = np.mean(compressions) if compressions else 0.0

    # 7. Measure entropy
    output_entropy = measure_entropy(output)

    # 8. Combined compression score
    compression_score = (
        0.3 * output_compression +
        0.3 * rank_reduction +
        0.2 * bottleneck_score +
        0.2 * (1 - output_entropy)
    )

    # 9. Measure trainability (NEW!)
    trainability_score = 0.0
    trainability_metrics = {}

    if include_trainability:
        try:
            from .trainable_eval import evaluate_trainability
            trainability_metrics = evaluate_trainability(model, device=device, verbose=False)
            trainability_score = trainability_metrics.get('trainability_score', 0.0)
        except Exception as e:
            # If trainability eval fails, skip it
            trainability_score = 0.5  # Neutral score

    # Combined score: weights favor trainability for better generalization
    combined_score = (
        0.4 * compression_score +
        0.6 * trainability_score
    )

    inference_time = (time.time() - start_time) * 1000  # ms

    result = {
        'combined_score': combined_score,
        'compression_score': compression_score,
        'trainability_score': trainability_score,
        'rank_reduction': rank_reduction,
        'output_compression': output_compression,
        'bottleneck_score': bottleneck_score,
        'flow_efficiency': flow_efficiency,
        'output_entropy': output_entropy,
        'output_rank': output_rank,
        'input_rank': input_rank,
        'inference_time_ms': inference_time,
    }

    # Add trainability details if available
    if trainability_metrics:
        result['gradient_snr'] = trainability_metrics.get('gradient_snr', 0.0)
        result['learning_speed'] = trainability_metrics.get('learning_speed', 0.0)
        result['feature_separability'] = trainability_metrics.get('feature_separability', 0.0)

    return sanitize_metrics(result)


def estimate_cmp_from_geometry(model):
    """
    Estimate Clade-Metaproductivity without training descendants.
    Uses geometric properties that correlate with future improvement.

    From Paper 1: CMP measures descendant success.
    We predict this from architecture geometry.
    """
    # Collect geometric features
    features = []

    # 1. Weight matrix properties
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            try:
                rank, compression, S = measure_rank(param.data)

                features.extend([
                    compression,
                    rank / max(param.shape[0], 1),  # Rank ratio (avoid div by 0)
                    (S[0] / (S.sum() + 1e-8)).item(),  # Top singular value ratio
                ])
            except:
                # Skip layers that fail SVD
                continue

    # 2. Architectural features
    num_bottlenecks, total_layers = check_bottleneck_exists(model)
    if total_layers > 0:
        features.append(num_bottlenecks / total_layers)

    # 3. Aggregate into CMP estimate (clip to reasonable range)
    if features and len(features) > 0:
        cmp_estimate = float(np.clip(np.mean(features), 0.0, 1.0))
    else:
        # Default to moderate CMP if no features extracted
        cmp_estimate = 0.3

    return cmp_estimate


def compare_architectures(models_dict, verbose=True):
    """
    Compare multiple architectures using zero-training evaluation.

    Args:
        models_dict: {name: model} dictionary
    Returns:
        results: {name: metrics} dictionary
    """
    results = {}

    for name, model in models_dict.items():
        if verbose:
            print(f"\nEvaluating {name}...")

        metrics = evaluate_architecture_fast(model)
        cmp = estimate_cmp_from_geometry(model)
        metrics['cmp_estimate'] = cmp

        results[name] = metrics

        if verbose:
            print(f"  Compression Score: {metrics['compression_score']:.3f}")
            print(f"  Rank Reduction: {metrics['rank_reduction']:.3f}")
            print(f"  CMP Estimate: {cmp:.3f}")
            print(f"  Time: {metrics['inference_time_ms']:.2f}ms")

    return results


if __name__ == "__main__":
    print("Testing Zero-Training Evaluation...\n")

    # Create simple test models
    class SimpleModel(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )

        def forward(self, x):
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
            out = self.layers(x_flat)
            return out.reshape(batch, seq, dim)

    class BottleneckModel(nn.Module):
        def __init__(self, dim=64, bottleneck=8):
            super().__init__()
            self.compress = nn.Linear(dim, bottleneck)
            self.expand = nn.Linear(bottleneck, dim)

        def forward(self, x):
            batch, seq, dim = x.shape
            x_flat = x.reshape(-1, dim)
            out = self.expand(torch.relu(self.compress(x_flat)))
            return out.reshape(batch, seq, dim)

    # Compare
    models = {
        'Simple': SimpleModel(),
        'Bottleneck': BottleneckModel(),
    }

    results = compare_architectures(models)

    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Overall Score: {metrics['compression_score']:.3f}")
        print(f"  CMP Estimate: {metrics['cmp_estimate']:.3f}")

    print("\nZero-training evaluation ready! 🚀")
