"""
Ultra-Fast Architecture Evaluation
Combines analytical scoring, learned prediction, and smart caching.
Target: 1000x speedup while maintaining accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import time


# ============================================================================
# 1. ANALYTICAL SCORING - No PyTorch needed!
# ============================================================================

def analytical_compression_score(spec):
    """
    Calculate compression score from architecture structure alone.
    ~1 microsecond vs ~50ms for full evaluation.

    Returns score in [0, 1] range.
    """
    if spec['type'] != 'sequential':
        return 0.0

    blocks = spec['blocks']
    if len(blocks) == 0:
        return 0.0

    # Count compression-inducing structures
    bottleneck_count = 0
    factorized_count = 0
    sparse_count = 0
    total_layers = 0

    # Track rank reduction through network
    rank_product = 1.0
    has_residual = False

    for block in blocks:
        block_type = block['type']

        if block_type == 'bottleneck':
            bottleneck_count += 1
            total_layers += 1
            # Rank reduction factor
            rank_product *= block['bottleneck'] / block['in']

        elif block_type == 'factorized':
            factorized_count += 1
            total_layers += 1
            rank_product *= block['rank'] / block['in']

        elif block_type == 'sparse_linear':
            sparse_count += 1
            total_layers += 1
            rank_product *= (1 - block['sparsity'])

        elif block_type == 'linear':
            total_layers += 1
            # Expansion is bad for compression
            if block['out'] > block['in']:
                rank_product *= block['in'] / block['out']

        elif block_type == 'residual':
            has_residual = True
            # Recursively analyze inner blocks
            inner_spec = {'type': 'sequential', 'blocks': block['blocks']}
            inner_score = analytical_compression_score(inner_spec)
            rank_product *= (1 - inner_score * 0.5)  # Residuals help

    if total_layers == 0:
        return 0.0

    # Combine factors
    bottleneck_score = bottleneck_count / (total_layers + 1)
    compression_score = 1 - rank_product  # Higher = more compressed
    structure_score = 0.2 if has_residual else 0.0

    # Weighted combination
    score = (
        0.4 * bottleneck_score +
        0.5 * compression_score +
        0.1 * structure_score
    )

    return np.clip(score, 0.0, 1.0)


def should_skip_mutation(spec, threshold=0.02):
    """
    Fast filter: skip obviously terrible architectures.
    Returns True if mutation should be skipped.
    Only rejects truly awful architectures.
    """
    score = analytical_compression_score(spec)
    return score < threshold


# ============================================================================
# 2. LEARNED PREDICTOR - Meta-Learning the Evaluation Function
# ============================================================================

class ArchitectureScorePredictor(nn.Module):
    """
    Neural network that learns to predict compression scores.
    Trains on real evaluations, then predicts new ones instantly.
    """

    def __init__(self, feature_dim=64, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Score in [0, 1]
        )

        # Training state
        self.is_trained = False
        self.training_data = []
        self.accuracy = 0.0

    def extract_features(self, spec):
        """
        Convert architecture spec to fixed-size feature vector.
        """
        features = np.zeros(self.feature_dim)

        if spec['type'] != 'sequential':
            return torch.tensor(features, dtype=torch.float32)

        blocks = spec['blocks']

        # Basic statistics
        features[0] = len(blocks)  # Total blocks

        # Block type counts (one-hot style)
        type_counts = defaultdict(int)
        for block in blocks:
            type_counts[block['type']] += 1

        features[1] = type_counts['bottleneck']
        features[2] = type_counts['linear']
        features[3] = type_counts['factorized']
        features[4] = type_counts['sparse_linear']
        features[5] = type_counts['guided_attention']
        features[6] = type_counts['activation']
        features[7] = type_counts['layernorm']
        features[8] = type_counts['residual']

        # Bottleneck sizes (histogram)
        bottleneck_sizes = []
        for block in blocks:
            if block['type'] == 'bottleneck':
                bottleneck_sizes.append(block['bottleneck'])

        if bottleneck_sizes:
            features[9] = np.mean(bottleneck_sizes)
            features[10] = np.min(bottleneck_sizes)
            features[11] = np.max(bottleneck_sizes)

        # Rank information
        ranks = []
        for block in blocks:
            if 'rank' in block:
                ranks.append(block['rank'])

        if ranks:
            features[12] = np.mean(ranks)

        # Dimension flow
        dims = []
        for block in blocks:
            if 'out' in block:
                dims.append(block['out'])
            elif 'dim' in block:
                dims.append(block['dim'])

        if dims:
            features[13] = np.mean(dims)
            features[14] = np.std(dims)

        # Analytical score as a feature
        features[15] = analytical_compression_score(spec)

        # Architecture depth patterns
        features[16] = len([b for b in blocks if b['type'] in ['linear', 'bottleneck']])
        features[17] = len([b for b in blocks if b['type'] == 'activation'])
        features[18] = len([b for b in blocks if b['type'] == 'layernorm'])

        return torch.tensor(features, dtype=torch.float32)

    def add_training_example(self, spec, true_score):
        """Add a real evaluation to training set"""
        features = self.extract_features(spec)
        self.training_data.append((features, true_score))

    def train_predictor(self, epochs=50, verbose=True):
        """Train on collected real evaluations"""
        if len(self.training_data) < 50:
            return False  # Need more data

        # Get device
        device = next(self.parameters()).device

        # Prepare dataset
        X = torch.stack([x[0] for x in self.training_data]).to(device)
        y = torch.tensor([x[1] for x in self.training_data], dtype=torch.float32).unsqueeze(1).to(device)

        # Train/val split
        n_train = int(0.8 * len(X))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        # Training
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()

        # Validation
        self.eval()
        with torch.no_grad():
            val_pred = self(X_val)
            val_loss = criterion(val_pred, y_val)
            # Accuracy: within 0.1 of true score
            accuracy = (torch.abs(val_pred - y_val) < 0.1).float().mean().item()

        self.accuracy = accuracy
        self.is_trained = True

        if verbose:
            print(f"\n🧠 Predictor trained on {len(self.training_data)} examples")
            print(f"   Validation loss: {val_loss.item():.4f}")
            print(f"   Accuracy (±0.1): {accuracy*100:.1f}%")

        return True

    def forward(self, features_or_spec):
        """Predict score from features or spec"""
        if isinstance(features_or_spec, dict):
            # It's a spec, extract features
            features = self.extract_features(features_or_spec)
        else:
            features = features_or_spec

        if features.dim() == 1:
            features = features.unsqueeze(0)

        return self.net(features).squeeze()

    def predict_score(self, spec):
        """Predict compression score for architecture"""
        if not self.is_trained:
            return None  # Not ready yet

        self.eval()
        with torch.no_grad():
            features = self.extract_features(spec)
            # Move to same device as model
            features = features.to(next(self.parameters()).device)
            score = self.net(features.unsqueeze(0)).squeeze().item()

        return score


# ============================================================================
# 3. SMART EVALUATION MANAGER
# ============================================================================

class FastEvaluationManager:
    """
    Manages evaluation with multiple strategies:
    1. Analytical pre-filter (instant)
    2. Learned prediction (0.1ms)
    3. Real evaluation (50ms) - ground truth
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.predictor = ArchitectureScorePredictor().to(device)

        # Statistics
        self.total_queries = 0
        self.analytical_rejects = 0
        self.predicted_evals = 0
        self.real_evals = 0

        # Training schedule
        self.train_interval = 50  # Retrain every N real evals
        self.last_train_size = 0
        self.min_training_examples = 50  # Minimum examples before first training

        self.start_time = time.time()

    def should_use_predictor(self):
        """Decide if predictor is good enough to use"""
        return self.predictor.is_trained and self.predictor.accuracy > 0.7

    def evaluate(self, spec, real_eval_fn, force_real=False):
        """
        Evaluate architecture using best available method.

        Args:
            spec: Architecture specification
            real_eval_fn: Function that does real evaluation
            force_real: Force real evaluation (for ground truth)

        Returns:
            score: Compression score
            method: 'analytical_reject', 'predicted', or 'real'
        """
        self.total_queries += 1

        # 1. Analytical pre-filter (only reject truly awful architectures)
        if should_skip_mutation(spec, threshold=0.02):
            self.analytical_rejects += 1
            return 0.0, 'analytical_reject'

        # 2. Use predictor if available and accurate
        if not force_real and self.should_use_predictor():
            # Use prediction with some probability, keep some real evals
            if np.random.random() < 0.9:  # 90% predicted, 10% real
                score = self.predictor.predict_score(spec)
                self.predicted_evals += 1
                return score, 'predicted'

        # 3. Real evaluation (ground truth)
        metrics = real_eval_fn(spec)
        score = metrics['compression_score']

        # Add to predictor training set
        self.predictor.add_training_example(spec, score)
        self.real_evals += 1

        # Retrain predictor periodically
        if (self.real_evals >= self.min_training_examples and
            self.real_evals % self.train_interval == 0 and
            self.real_evals > self.last_train_size):
            self.predictor.train_predictor(epochs=50, verbose=True)
            self.last_train_size = self.real_evals

        return score, 'real'

    def get_stats(self):
        """Get evaluation statistics"""
        if self.total_queries == 0:
            return {}

        elapsed = time.time() - self.start_time

        # Estimated time saved
        # Analytical: ~50ms saved per reject
        # Predicted: ~49.9ms saved per prediction
        time_saved = (
            self.analytical_rejects * 0.05 +  # 50ms saved
            self.predicted_evals * 0.0499      # 49.9ms saved
        )

        speedup = elapsed / max(0.001, elapsed - time_saved)

        return {
            'total_queries': self.total_queries,
            'analytical_rejects': self.analytical_rejects,
            'predicted_evals': self.predicted_evals,
            'real_evals': self.real_evals,
            'predictor_accuracy': self.predictor.accuracy if self.predictor.is_trained else 0.0,
            'effective_speedup': speedup,
            'time_saved_seconds': time_saved,
        }

    def print_stats(self):
        """Print statistics"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("⚡ FAST EVALUATION STATISTICS")
        print("=" * 60)
        print(f"Total queries: {stats['total_queries']}")
        print(f"  Analytical rejects: {stats['analytical_rejects']} ({stats['analytical_rejects']/stats['total_queries']*100:.1f}%)")
        print(f"  Predicted evals: {stats['predicted_evals']} ({stats['predicted_evals']/stats['total_queries']*100:.1f}%)")
        print(f"  Real evals: {stats['real_evals']} ({stats['real_evals']/stats['total_queries']*100:.1f}%)")

        if self.predictor.is_trained:
            print(f"\n🧠 Predictor accuracy: {stats['predictor_accuracy']*100:.1f}%")

        print(f"\n🚀 Effective speedup: {stats['effective_speedup']:.1f}x")
        print(f"   Time saved: {stats['time_saved_seconds']:.1f}s")
        print("=" * 60)


# ============================================================================
# 4. AGGRESSIVE PRUNING
# ============================================================================

def should_prune_node(node, min_evals=5, cmp_threshold=0.05):
    """
    Decide if a node's lineage should be pruned.

    Args:
        node: ArchNode to check
        min_evals: Minimum evaluations before pruning
        cmp_threshold: CMP below this = prune

    Returns:
        True if should prune
    """
    # Need enough data
    if node.num_evaluations < min_evals:
        return False

    # Check clade-level performance
    clade_cmp = node.get_clade_cmp()

    # Only prune truly terrible lineages
    if clade_cmp < cmp_threshold:
        return True

    # If node has many failed children, prune
    if len(node.children) >= 5:
        child_scores = [c.metrics['compression_score'] if c.metrics else 0.0
                       for c in node.children]
        if np.mean(child_scores) < 0.05:
            return True

    return False


if __name__ == "__main__":
    print("Testing Fast Evaluation System...\n")

    # Test analytical scoring
    from architectures import create_bottleneck_arch, create_transformer_baseline

    spec1 = create_bottleneck_arch(dim=64, bottleneck=8)
    spec2 = create_transformer_baseline(dim=64)

    score1 = analytical_compression_score(spec1)
    score2 = analytical_compression_score(spec2)

    print(f"Analytical scores:")
    print(f"  Bottleneck arch: {score1:.3f}")
    print(f"  Transformer arch: {score2:.3f}")

    # Test predictor
    print("\n🧠 Testing learned predictor...")
    predictor = ArchitectureScorePredictor()

    # Simulate some training data
    for i in range(100):
        spec = create_bottleneck_arch(dim=64, bottleneck=np.random.choice([4, 8, 16]))
        score = analytical_compression_score(spec) + np.random.normal(0, 0.05)
        predictor.add_training_example(spec, score)

    predictor.train_predictor(epochs=50)

    # Test prediction
    test_spec = create_bottleneck_arch(dim=64, bottleneck=8)
    pred_score = predictor.predict_score(test_spec)
    true_score = analytical_compression_score(test_spec)

    print(f"\nTest prediction:")
    print(f"  Predicted: {pred_score:.3f}")
    print(f"  Analytical: {true_score:.3f}")
    print(f"  Error: {abs(pred_score - true_score):.3f}")

    print("\n✅ Fast evaluation system ready!")
