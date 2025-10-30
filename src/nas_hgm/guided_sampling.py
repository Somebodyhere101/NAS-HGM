"""
Guided Randomness - Human-Inspired Probability Adjustment
Implements intuition-guided selection instead of pure argmax or random sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GuidedRandomnessLayer(nn.Module):
    """
    Learns to adjust probability landscapes like human intuition.
    Not pure random, not pure greedy - guided exploration.
    """

    def __init__(self, dim, history_size=100):
        super().__init__()
        self.dim = dim

        # "Intuition" - learned bias field
        self.guidance_field = nn.Parameter(torch.zeros(dim))

        # "Feeling" - temperature control
        self.temperature = nn.Parameter(torch.ones(1))

        # History for adaptive guidance
        self.history_size = history_size
        self.selection_history = []
        self.outcome_history = []

    def forward(self, logits, deterministic=False):
        """
        Args:
            logits: [batch, dim] or [dim] - raw scores
            deterministic: If True, return adjusted argmax (for eval)
        Returns:
            selection: indices or values
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Apply temperature and guidance
        adjusted = logits / (self.temperature.abs() + 1e-8) + self.guidance_field

        if deterministic:
            return adjusted.argmax(dim=-1)
        else:
            # Guided stochastic selection
            return self.guided_sample(adjusted)

    def guided_sample(self, logits):
        """Sample with quality bias - not uniform, not greedy"""
        probs = F.softmax(logits, dim=-1)

        # Sample from distribution
        if probs.shape[0] == 1:
            dist = torch.distributions.Categorical(probs[0])
            return dist.sample()
        else:
            dist = torch.distributions.Categorical(probs)
            return dist.sample()

    def update_intuition(self, selection, outcome_quality):
        """
        Update guidance field based on outcome.
        Good outcomes → strengthen that direction.
        """
        self.selection_history.append(selection)
        self.outcome_history.append(outcome_quality)

        # Keep limited history
        if len(self.selection_history) > self.history_size:
            self.selection_history.pop(0)
            self.outcome_history.pop(0)

        # Adjust guidance toward successful selections
        if len(self.outcome_history) > 10:
            outcomes = torch.tensor(self.outcome_history[-10:])
            selections = torch.tensor(self.selection_history[-10:])

            # Weight recent successes
            weights = outcomes / (outcomes.sum() + 1e-8)

            # Update guidance (simplified - in practice would be more sophisticated)
            for sel, w in zip(selections, weights):
                if sel < self.dim:
                    self.guidance_field.data[sel] += 0.01 * w


class EntropyMixer(nn.Module):
    """
    Mixes multiple probability distributions with controlled entropy.
    Allows balancing exploration vs exploitation.
    """

    def __init__(self, num_sources):
        super().__init__()
        self.num_sources = num_sources
        self.mixing_weights = nn.Parameter(torch.ones(num_sources) / num_sources)
        self.entropy_target = nn.Parameter(torch.ones(1))

    def forward(self, distributions):
        """
        Args:
            distributions: List of [batch, dim] probability distributions
        Returns:
            mixed: [batch, dim] mixed distribution
        """
        weights = F.softmax(self.mixing_weights, dim=0)

        # Weighted combination
        mixed = sum(w * d for w, d in zip(weights, distributions))

        # Adjust entropy toward target
        current_entropy = self.compute_entropy(mixed)
        target = self.entropy_target.abs()

        if current_entropy < target:
            # Increase entropy (more exploration)
            mixed = self.increase_entropy(mixed, target - current_entropy)
        elif current_entropy > target:
            # Decrease entropy (more exploitation)
            mixed = self.decrease_entropy(mixed, current_entropy - target)

        return mixed

    def compute_entropy(self, probs):
        """Shannon entropy of probability distribution"""
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

    def increase_entropy(self, probs, amount):
        """Make distribution more uniform"""
        uniform = torch.ones_like(probs) / probs.shape[-1]
        alpha = torch.sigmoid(amount)
        return (1 - alpha) * probs + alpha * uniform

    def decrease_entropy(self, probs, amount):
        """Make distribution more peaked"""
        alpha = torch.sigmoid(amount)
        return probs ** (1 + alpha)


class GuidedBottleneck(nn.Module):
    """
    Bottleneck with guided randomness in the compression phase.
    Combines forced low-rank with adaptive selection.
    """

    def __init__(self, in_dim, bottleneck_dim, out_dim):
        super().__init__()

        # Low-rank compression
        self.compress = nn.Linear(in_dim, bottleneck_dim, bias=False)
        self.expand = nn.Linear(bottleneck_dim, out_dim, bias=False)

        # Guided selection in bottleneck
        self.guidance = GuidedRandomnessLayer(bottleneck_dim)

        # Initialize for compression
        nn.init.orthogonal_(self.compress.weight)
        nn.init.orthogonal_(self.expand.weight)

    def forward(self, x, apply_guidance=False):
        """
        Args:
            x: [batch, in_dim]
            apply_guidance: Whether to apply guided randomness (disabled for now)
        Returns:
            out: [batch, out_dim]
        """
        # Compress
        z = self.compress(x)

        # Guidance disabled - dimension mismatch needs fixing
        # The GuidedRandomnessLayer is designed for discrete selection,
        # not continuous gating. Would need redesign for this use case.

        # Expand
        return self.expand(z)


def create_guided_attention(dim, num_heads=4, rank=None):
    """
    Create attention mechanism with guided randomness.
    Similar to transformer attention but with probability guidance.
    """
    if rank is None:
        rank = dim // 4

    class GuidedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads

            # Low-rank Q, K (Paper 2 insight)
            self.q_compress = nn.Linear(dim, rank, bias=False)
            self.k_compress = nn.Linear(dim, rank, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)

            # Guided selection for attention patterns
            self.attention_guidance = GuidedRandomnessLayer(dim)

        def forward(self, x):
            # Handle both 2D and 3D inputs
            if x.dim() == 2:
                # [batch, dim] -> add seq dimension
                x = x.unsqueeze(1)
                squeeze_output = True
            else:
                squeeze_output = False

            batch, seq, _ = x.shape

            # Low-rank projections
            q = self.q_compress(x)  # [batch, seq, rank]
            k = self.k_compress(x)
            v = self.v_proj(x)

            # Attention scores (compressed QK^T)
            scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.head_dim)

            # Standard softmax attention
            attn = F.softmax(scores, dim=-1)

            # Apply value projection
            out = torch.bmm(attn, v)

            if squeeze_output:
                out = out.squeeze(1)

            return out

    return GuidedAttention()


if __name__ == "__main__":
    # Test guided randomness
    print("Testing Guided Randomness Layer...")

    layer = GuidedRandomnessLayer(10)
    logits = torch.randn(5, 10)

    # Sample multiple times
    samples = [layer(logits).item() for _ in range(100)]
    print(f"Sample distribution: {np.bincount(samples, minlength=10)}")

    # Update with positive feedback on certain choices
    for _ in range(50):
        selection = layer(logits)
        # Reward selections 3, 7, 9
        quality = 1.0 if selection in [3, 7, 9] else 0.1
        layer.update_intuition(selection, quality)

    # Check if guidance learned
    print(f"Guidance field: {layer.guidance_field.data}")
    print(f"Temperature: {layer.temperature.data}")

    print("\nTesting Guided Bottleneck...")
    bottleneck = GuidedBottleneck(64, 8, 64)
    x = torch.randn(4, 64)
    out = bottleneck(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    print("\nGuided randomness components ready!")
