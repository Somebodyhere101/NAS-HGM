"""
Architecture Search Space and Mutation Operators
Defines modular building blocks and how to evolve them.
"""

import torch
import torch.nn as nn
import random
import copy
from .guided_sampling import GuidedBottleneck, create_guided_attention


# ============================================================================
# Architecture Specification Format
# ============================================================================

"""
Architecture = {
    'type': 'sequential',
    'blocks': [
        {'type': 'linear', 'in': 64, 'out': 64},
        {'type': 'bottleneck', 'in': 64, 'bottleneck': 8, 'out': 64},
        {'type': 'guided_attention', 'dim': 64, 'heads': 4, 'rank': 16},
        {'type': 'activation', 'fn': 'relu'},
    ],
    'input_shape': (batch, seq, 64),
    'output_shape': (batch, seq, 64),
}
"""


# ============================================================================
# Building Blocks
# ============================================================================

class BlockRegistry:
    """Registry of all available building blocks"""

    @staticmethod
    def create_block(spec):
        """Create a PyTorch module from spec dict"""
        block_type = spec['type']

        if block_type == 'linear':
            return nn.Linear(spec['in'], spec['out'], bias=spec.get('bias', False))

        elif block_type == 'bottleneck':
            return GuidedBottleneck(spec['in'], spec['bottleneck'], spec['out'])

        elif block_type == 'guided_attention':
            return create_guided_attention(
                spec['dim'],
                spec.get('heads', 4),
                spec.get('rank', spec['dim'] // 4)
            )

        elif block_type == 'activation':
            fn = spec.get('fn', 'relu')
            if fn == 'relu':
                return nn.ReLU()
            elif fn == 'gelu':
                return nn.GELU()
            elif fn == 'tanh':
                return nn.Tanh()
            else:
                return nn.ReLU()

        elif block_type == 'layernorm':
            return nn.LayerNorm(spec['dim'])

        elif block_type == 'residual':
            # Create residual connection
            inner = build_model_from_spec({'type': 'sequential', 'blocks': spec['blocks']})
            return ResidualBlock(inner)

        elif block_type == 'sparse_linear':
            return SparseLinear(spec['in'], spec['out'], spec.get('sparsity', 0.5))

        elif block_type == 'factorized':
            return FactorizedLinear(spec['in'], spec['out'], spec.get('rank', 16))

        elif block_type == 'learned_gate':
            return LearnedGate(spec['dim'])

        elif block_type == 'competitive_select':
            return CompetitiveSelection(spec['dim'], spec.get('k', spec['dim'] // 4))

        elif block_type == 'stochastic_path':
            return StochasticPath(spec['dim'], spec.get('num_paths', 2))

        elif block_type == 'confidence_gate':
            return ConfidenceGate(spec['dim'])

        elif block_type == 'dynamic_routing':
            return DynamicRouting(spec['in'], spec['out'], spec.get('num_experts', 4))

        elif block_type == 'extracted_rule':
            return ExtractedRule(spec['rule_matrix'], spec['dim'])

        elif block_type == 'composed':
            inner_blocks = [BlockRegistry.create_block(b) for b in spec['primitives']]
            return ComposedPrimitive(inner_blocks)

        elif block_type == 'synthesized':
            return SynthesizedPrimitive(spec['rule_matrix'], spec['dim'], spec['primitive_id'])

        else:
            raise ValueError(f"Unknown block type: {block_type}")


class ResidualBlock(nn.Module):
    """Residual connection wrapper"""

    def __init__(self, inner_module):
        super().__init__()
        self.inner = inner_module
        self.projection = None

    def forward(self, x):
        out = self.inner(x)

        # Handle dimension mismatch
        if x.shape[-1] != out.shape[-1]:
            # Project output to match input dimension
            if self.projection is None:
                self.projection = nn.Linear(out.shape[-1], x.shape[-1]).to(x.device)
            out = self.projection(out)

        # Now shapes should match
        if x.shape == out.shape:
            return x + out
        else:
            # If shapes still don't match, just return output
            return out


class SparseLinear(nn.Module):
    """Linear layer with enforced sparsity"""

    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.sparsity = sparsity

        # Create sparse mask
        mask = torch.rand(out_features, in_features) > sparsity
        self.register_buffer('mask', mask.float())

    def forward(self, x):
        # Apply mask to weights
        masked_weight = self.linear.weight * self.mask
        return nn.functional.linear(x, masked_weight)


class FactorizedLinear(nn.Module):
    """Low-rank factorized linear layer W = UV^T"""

    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        self.rank = rank
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.V(self.U(x))


class LearnedGate(nn.Module):
    """Learns to gate each dimension independently based on input context"""

    def __init__(self, dim):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gates = self.gate_network(x)
        return x * gates


class CompetitiveSelection(nn.Module):
    """
    Winner-take-all or k-winners selection mechanism.
    Only top-k features pass through, others zeroed.
    """

    def __init__(self, dim, k):
        super().__init__()
        self.dim = dim
        self.k = max(1, min(k, dim))
        self.importance = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        batch_size = x.shape[0]

        # Compute importance scores
        scores = torch.abs(x) * self.importance.abs()

        # Select top-k per batch element
        if x.dim() == 2:
            _, indices = torch.topk(scores, self.k, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, indices, 1.0)
        else:
            # Handle 3D tensors
            _, indices = torch.topk(scores, self.k, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, indices, 1.0)

        return x * mask


class StochasticPath(nn.Module):
    """
    Learns to route through multiple paths stochastically.
    During training, samples paths; during eval, uses expected value.
    """

    def __init__(self, dim, num_paths=2):
        super().__init__()
        self.num_paths = num_paths
        self.paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.Tanh()
            ) for _ in range(num_paths)
        ])
        self.router = nn.Linear(dim, num_paths)

    def forward(self, x):
        # Compute routing probabilities
        routing_logits = self.router(x.mean(dim=1) if x.dim() == 3 else x)
        routing_probs = torch.softmax(routing_logits, dim=-1)

        if self.training:
            # Sample a path during training
            path_idx = torch.multinomial(routing_probs, 1).squeeze(-1)
            outputs = []
            for i, path in enumerate(self.paths):
                mask = (path_idx == i).float().unsqueeze(-1)
                if x.dim() == 3:
                    mask = mask.unsqueeze(1)
                outputs.append(path(x) * mask)
            return sum(outputs)
        else:
            # Expected value during eval
            outputs = [path(x) for path in self.paths]
            stacked = torch.stack(outputs, dim=-1)
            if x.dim() == 3:
                weights = routing_probs.unsqueeze(1).unsqueeze(-2)
            else:
                weights = routing_probs.unsqueeze(-2)
            return (stacked * weights).sum(dim=-1)


class ConfidenceGate(nn.Module):
    """
    Learns confidence scores for each activation.
    High confidence = pass through, low confidence = suppress.
    """

    def __init__(self, dim):
        super().__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        confidence = self.confidence_net(x)
        # Soft thresholding
        gate = torch.sigmoid((confidence - self.threshold.sigmoid()) * 10)
        return x * gate


class DynamicRouting(nn.Module):
    """
    Mixture of experts style routing.
    Learns to route to different expert networks based on input.
    """

    def __init__(self, in_dim, out_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(in_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if x.dim() == 3:
            routing_weights = self.router(x.mean(dim=1))
        else:
            routing_weights = self.router(x)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        if x.dim() == 3:
            weights = routing_weights.unsqueeze(1).unsqueeze(-2)
        else:
            weights = routing_weights.unsqueeze(-2)

        return (expert_outputs * weights).sum(dim=-1)


class ExtractedRule(nn.Module):
    """
    Primitive generated from compressed QK^T rules.
    Stores decision logic extracted from successful architectures.
    """

    def __init__(self, rule_matrix, dim):
        super().__init__()
        self.dim = dim

        # Convert list back to tensor if needed
        if isinstance(rule_matrix, list):
            rule_matrix = torch.tensor(rule_matrix, dtype=torch.float32)

        U, S, Vh = torch.linalg.svd(rule_matrix, full_matrices=False)

        k = min(dim // 4, len(S))
        energy = (S[:k].sum() / S.sum()).item()
        while k < len(S) and energy < 0.95:
            k += 1
            energy = (S[:k].sum() / S.sum()).item()

        self.Q = nn.Parameter(U[:, :k] @ torch.diag(S[:k].sqrt()))
        self.K = nn.Parameter(Vh[:k, :].T @ torch.diag(S[:k].sqrt()))

    def forward(self, x):
        scores = (x @ self.Q) @ (x @ self.K).transpose(-1, -2) / (self.dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return attn @ x


class ComposedPrimitive(nn.Module):
    """
    Meta-primitive: composition of other primitives.
    System discovers these through evolution.
    """

    def __init__(self, primitives):
        super().__init__()
        self.primitives = nn.ModuleList(primitives)

    def forward(self, x):
        for p in self.primitives:
            x = p(x)
        return x


class SynthesizedPrimitive(nn.Module):
    """
    Primitive synthesized from extracted comparison rules.
    Generates new comparison operators from QK^T bottleneck.
    """

    def __init__(self, rule_matrix, dim, primitive_id):
        super().__init__()
        self.dim = dim
        self.primitive_id = primitive_id

        # Convert list back to tensor if needed
        if isinstance(rule_matrix, list):
            rule = torch.tensor(rule_matrix, dtype=torch.float32)
        else:
            rule = rule_matrix.float() if rule_matrix.dtype != torch.float32 else rule_matrix

        if rule.shape[0] != dim or rule.shape[1] != dim:
            if rule.shape[0] == rule.shape[1]:
                scale = dim / rule.shape[0]
                rule = nn.functional.interpolate(
                    rule.unsqueeze(0).unsqueeze(0),
                    size=(dim, dim),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                rule = torch.randn(dim, dim) * 0.02

        U, S, Vh = torch.linalg.svd(rule, full_matrices=False)

        scores = (S / (S.sum() + 1e-8)).cumsum(0)
        k = max(1, min(dim // 4, int((scores < 0.95).sum().item()) + 1))
        k = min(k, len(S))

        self.compare_left = nn.Parameter(U[:, :k] @ torch.diag(S[:k].sqrt()))
        self.compare_right = nn.Parameter(Vh[:k, :].T @ torch.diag(S[:k].sqrt()))

        self.threshold = nn.Parameter(torch.zeros(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        left = x @ self.compare_left
        right = x @ self.compare_right

        comparison = (left @ right.transpose(-1, -2)) / (self.dim ** 0.5)

        mask = (comparison > self.threshold).float()
        gated = torch.softmax(comparison + self.gate_bias, dim=-1)

        return (mask * gated) @ x


# ============================================================================
# Model Builder
# ============================================================================

def build_model_from_spec(spec):
    """Build PyTorch model from architecture specification"""

    if spec['type'] == 'sequential':
        blocks = []
        for block_spec in spec['blocks']:
            blocks.append(BlockRegistry.create_block(block_spec))

        return nn.Sequential(*blocks)

    else:
        raise ValueError(f"Unknown architecture type: {spec['type']}")


# ============================================================================
# Baseline Architectures
# ============================================================================

def create_transformer_baseline(dim=64, heads=4, layers=2):
    """Standard transformer as baseline"""
    return {
        'type': 'sequential',
        'blocks': [
            {'type': 'guided_attention', 'dim': dim, 'heads': heads, 'rank': dim // 4},
            {'type': 'layernorm', 'dim': dim},
            {'type': 'linear', 'in': dim, 'out': dim * 4},
            {'type': 'activation', 'fn': 'gelu'},
            {'type': 'linear', 'in': dim * 4, 'out': dim},
            {'type': 'layernorm', 'dim': dim},
        ] * layers
    }


def create_simple_mlp(dim=64, hidden=128):
    """Simple MLP baseline"""
    return {
        'type': 'sequential',
        'blocks': [
            {'type': 'linear', 'in': dim, 'out': hidden},
            {'type': 'activation', 'fn': 'relu'},
            {'type': 'linear', 'in': hidden, 'out': dim},
        ]
    }


def create_bottleneck_arch(dim=64, bottleneck=8):
    """Forced compression architecture"""
    return {
        'type': 'sequential',
        'blocks': [
            {'type': 'bottleneck', 'in': dim, 'bottleneck': bottleneck, 'out': dim},
            {'type': 'activation', 'fn': 'relu'},
            {'type': 'bottleneck', 'in': dim, 'bottleneck': bottleneck, 'out': dim},
        ]
    }


# ============================================================================
# Mutation Operators
# ============================================================================

def extract_rule_from_model(model, device='cpu'):
    """
    Extract compressed QK^T rule from attention layers.
    Returns rule matrix for creating new primitives.
    """
    rules = []

    for name, module in model.named_modules():
        try:
            if hasattr(module, 'q_compress') and hasattr(module, 'k_compress'):
                Q = module.q_compress.weight.detach().cpu()
                K = module.k_compress.weight.detach().cpu()
                rule = Q.T @ K
                rules.append(rule)
            elif hasattr(module, 'q_proj') and hasattr(module, 'k_proj'):
                Q = module.q_proj.weight.detach().cpu()
                K = module.k_proj.weight.detach().cpu()
                rule = Q @ K.T
                rules.append(rule)
        except:
            continue

    if rules:
        return torch.stack(rules).mean(dim=0)
    return None


def synthesize_primitives_from_rules(extracted_rules):
    """
    Synthesize NEW primitive operators from extracted QK^T comparison rules.
    True unbounded self-improvement: generates novel comparison operators.
    """
    synthesized = []

    for idx, rule in enumerate(extracted_rules):
        U, S, Vh = torch.linalg.svd(rule, full_matrices=False)

        rank = int((S.cumsum(0) / S.sum() < 0.95).sum().item()) + 1
        rank = max(1, rank)

        # Convert tensor to list for JSON serialization
        rule_list = rule.detach().cpu().tolist() if isinstance(rule, torch.Tensor) else rule

        synthesized.append({
            'mutation_name': f'add_synth_{idx}',
            'rule_matrix': rule_list,
            'primitive_id': idx,
            'rank': int(rank)
        })

    return synthesized


def generate_meta_mutations(successful_patterns):
    """
    Generate new mutation operators from discovered patterns.
    Analyzes successful architectures to create new mutation types.
    """
    new_mutations = []

    for pattern in successful_patterns:
        if 'bottleneck' in str(pattern).lower() and 'gate' in str(pattern).lower():
            new_mutations.append('add_gated_bottleneck')

        if 'competitive' in str(pattern).lower() and 'routing' in str(pattern).lower():
            new_mutations.append('add_competitive_routing')

        if len(pattern.get('blocks', [])) > 3:
            block_types = [b['type'] for b in pattern['blocks'] if 'type' in b]
            if len(set(block_types)) >= 3:
                new_mutations.append('add_discovered_composition')

    return list(set(new_mutations))


_discovered_mutations = []
_synthesized_primitives = []

def mutate_architecture(spec, mutation_type=None, mutation_probs=None, max_retries=10, extracted_rules=None):
    """
    Apply mutation to architecture spec with optional learned probabilities.
    Returns (new_spec, mutation_used) tuple.
    Retries if mutation creates dimension mismatches.

    Args:
        spec: Architecture specification
        mutation_type: Force specific mutation (overrides probabilities)
        mutation_probs: Dict of mutation -> probability (for learning)
        max_retries: Max attempts before giving up

    Returns:
        (mutated_spec, mutation_name): Tuple of new spec and which mutation was applied
    """
    base_mutations = [
        'add_bottleneck',
        'add_sparse_layer',
        'add_factorized',
        'reduce_rank',
        'increase_rank',
        'add_residual',
        'remove_layer',
        'add_layernorm',
        'change_activation',
        'adjust_bottleneck_size',
        'add_learned_gate',
        'add_competitive_select',
        'add_stochastic_path',
        'add_confidence_gate',
        'add_dynamic_routing',
    ]

    synth_mutations = [p['mutation_name'] for p in _synthesized_primitives]
    mutations = base_mutations + _discovered_mutations + synth_mutations

    for attempt in range(max_retries):
        spec_copy = copy.deepcopy(spec)

        if mutation_type is None:
            if mutation_probs:
                # Weighted sampling based on learned probabilities
                probs = [mutation_probs.get(m, 1.0) for m in mutations]
                total = sum(probs)
                probs = [p / total for p in probs]
                mut = random.choices(mutations, weights=probs, k=1)[0]
            else:
                # Uniform random (original behavior)
                mut = random.choice(mutations)
        else:
            mut = mutation_type

        blocks = spec_copy['blocks']

        # Apply mutation
        if mut == 'add_bottleneck':
            # Insert bottleneck at random position
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            bottleneck_size = random.choice([4, 8, 16, 32])
            blocks.insert(pos, {
                'type': 'bottleneck',
                'in': dim,
                'bottleneck': bottleneck_size,
                'out': dim
            })

        elif mut == 'add_sparse_layer':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            sparsity = random.choice([0.3, 0.5, 0.7, 0.9])
            blocks.insert(pos, {
                'type': 'sparse_linear',
                'in': dim,
                'out': dim,
                'sparsity': sparsity
            })

        elif mut == 'add_factorized':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            rank = random.choice([4, 8, 16, 32])
            blocks.insert(pos, {
                'type': 'factorized',
                'in': dim,
                'out': dim,
                'rank': rank
            })

        elif mut == 'reduce_rank':
            # Find attention or factorized blocks and reduce rank
            for block in blocks:
                if block['type'] in ['guided_attention', 'factorized']:
                    if 'rank' in block:
                        block['rank'] = max(2, block['rank'] // 2)

        elif mut == 'increase_rank':
            for block in blocks:
                if block['type'] in ['guided_attention', 'factorized']:
                    if 'rank' in block:
                        # Don't increase beyond dim/2 to maintain compression
                        max_rank = block.get('dim', 64) // 2 if block['type'] == 'guided_attention' else 32
                        block['rank'] = min(max_rank, block['rank'] * 2)

        elif mut == 'add_residual':
            # Wrap a sequence of blocks in residual
            if len(blocks) >= 2:
                start = random.randint(0, len(blocks) - 2)
                length = random.randint(1, min(3, len(blocks) - start))
                wrapped = blocks[start:start + length]
                residual_block = {
                    'type': 'residual',
                    'blocks': wrapped
                }
                blocks = blocks[:start] + [residual_block] + blocks[start + length:]
                spec_copy['blocks'] = blocks

        elif mut == 'remove_layer':
            if len(blocks) > 1:
                idx = random.randint(0, len(blocks) - 1)
                blocks.pop(idx)

        elif mut == 'add_layernorm':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            blocks.insert(pos, {'type': 'layernorm', 'dim': dim})

        elif mut == 'change_activation':
            for block in blocks:
                if block['type'] == 'activation':
                    block['fn'] = random.choice(['relu', 'gelu', 'tanh'])

        elif mut == 'adjust_bottleneck_size':
            for block in blocks:
                if block['type'] == 'bottleneck':
                    old_size = block['bottleneck']
                    block['bottleneck'] = random.choice([
                        max(2, old_size // 2),
                        old_size,
                        min(64, old_size * 2)
                    ])

        elif mut == 'add_learned_gate':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            blocks.insert(pos, {
                'type': 'learned_gate',
                'dim': dim
            })

        elif mut == 'add_competitive_select':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            k = random.choice([dim // 8, dim // 4, dim // 2])
            blocks.insert(pos, {
                'type': 'competitive_select',
                'dim': dim,
                'k': k
            })

        elif mut == 'add_stochastic_path':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            num_paths = random.choice([2, 3, 4])
            blocks.insert(pos, {
                'type': 'stochastic_path',
                'dim': dim,
                'num_paths': num_paths
            })

        elif mut == 'add_confidence_gate':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            blocks.insert(pos, {
                'type': 'confidence_gate',
                'dim': dim
            })

        elif mut == 'add_dynamic_routing':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            num_experts = random.choice([2, 4, 8])
            blocks.insert(pos, {
                'type': 'dynamic_routing',
                'in': dim,
                'out': dim,
                'num_experts': num_experts
            })

        elif mut == 'add_extracted_rule' and extracted_rules is not None:
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            rule_matrix = extracted_rules[random.randint(0, len(extracted_rules) - 1)]
            blocks.insert(pos, {
                'type': 'extracted_rule',
                'rule_matrix': rule_matrix,
                'dim': dim
            })

        elif mut == 'add_gated_bottleneck':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            bottleneck_size = random.choice([4, 8, 16])
            blocks.insert(pos, {
                'type': 'composed',
                'primitives': [
                    {'type': 'bottleneck', 'in': dim, 'bottleneck': bottleneck_size, 'out': dim},
                    {'type': 'learned_gate', 'dim': dim}
                ]
            })

        elif mut == 'add_competitive_routing':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            blocks.insert(pos, {
                'type': 'composed',
                'primitives': [
                    {'type': 'competitive_select', 'dim': dim, 'k': dim // 4},
                    {'type': 'dynamic_routing', 'in': dim, 'out': dim, 'num_experts': 4}
                ]
            })

        elif mut == 'add_discovered_composition':
            pos = random.randint(0, len(blocks))
            dim = _infer_dim(blocks, pos)
            composition_types = random.sample([
                'learned_gate', 'competitive_select', 'stochastic_path',
                'confidence_gate', 'bottleneck'
            ], k=random.randint(2, 3))

            composed_blocks = []
            for comp_type in composition_types:
                if comp_type == 'bottleneck':
                    composed_blocks.append({
                        'type': 'bottleneck',
                        'in': dim,
                        'bottleneck': random.choice([4, 8, 16]),
                        'out': dim
                    })
                else:
                    composed_blocks.append({'type': comp_type, 'dim': dim})

            blocks.insert(pos, {
                'type': 'composed',
                'primitives': composed_blocks
            })

        elif mut.startswith('add_synth_'):
            synth_info = next((p for p in _synthesized_primitives if p['mutation_name'] == mut), None)
            if synth_info:
                pos = random.randint(0, len(blocks))
                dim = _infer_dim(blocks, pos)
                blocks.insert(pos, {
                    'type': 'synthesized',
                    'rule_matrix': synth_info['rule_matrix'],
                    'dim': dim,
                    'primitive_id': synth_info['primitive_id']
                })

        # Validate architecture dimensions
        if validate_architecture_dims(spec_copy):
            return spec_copy, mut

    return spec, None


def _infer_dim(blocks, pos):
    """Infer dimension at position in block sequence"""
    # Look backwards for a block with known dimension
    for i in range(pos - 1, -1, -1):
        block = blocks[i]
        if 'out' in block:
            return block['out']
        if 'dim' in block:
            return block['dim']

    # Look forwards
    for i in range(pos, len(blocks)):
        block = blocks[i]
        if 'in' in block:
            return block['in']
        if 'dim' in block:
            return block['dim']

    # Default
    return 64


def validate_architecture_dims(spec):
    """
    Check if architecture has consistent dimensions.
    Returns True if valid, False if dimension mismatches exist.
    """
    if spec['type'] != 'sequential':
        return True

    blocks = spec['blocks']
    current_dim = 64  # Default input dimension

    for i, block in enumerate(blocks):
        block_type = block['type']

        if block_type == 'linear':
            if block['in'] != current_dim:
                return False
            current_dim = block['out']

        elif block_type in ['bottleneck', 'factorized']:
            if block['in'] != current_dim:
                return False
            current_dim = block['out']

        elif block_type == 'sparse_linear':
            if block['in'] != current_dim:
                return False
            current_dim = block['out']

        elif block_type in ['layernorm', 'guided_attention']:
            if block['dim'] != current_dim:
                return False
            # These preserve dimension

        elif block_type == 'activation':
            # Preserves dimension
            pass

        elif block_type == 'residual':
            # Validate inner blocks
            inner_spec = {'type': 'sequential', 'blocks': block['blocks']}
            if not validate_architecture_dims(inner_spec):
                return False
            # Residual preserves dimension (with projection if needed)

    return True


# ============================================================================
# Architecture Wrapper
# ============================================================================

class ArchitectureWrapper(nn.Module):
    """
    Wraps specification-built model with consistent interface.
    Handles shape transformations.
    """

    def __init__(self, spec, input_dim=64):
        super().__init__()
        self.spec = spec
        self.input_dim = input_dim
        self.model = build_model_from_spec(spec)

    def forward(self, x):
        """
        Args:
            x: [batch, seq, dim] tensor
        Returns:
            out: [batch, seq, dim] tensor
        """
        batch, seq, dim = x.shape
        x_flat = x.reshape(-1, dim)

        # Forward through model
        out_flat = self.model(x_flat)

        # Get output dimension
        out_dim = out_flat.shape[-1]

        # Add projection if dimension mismatch
        if out_dim != dim:
            if not hasattr(self, 'output_proj'):
                self.output_proj = nn.Linear(out_dim, dim).to(x.device)
            out_flat = self.output_proj(out_flat)

        # Reshape back to [batch, seq, dim]
        return out_flat.reshape(batch, seq, dim)


if __name__ == "__main__":
    print("Testing Architecture System...\n")

    # Create baseline
    spec = create_transformer_baseline(dim=64, heads=4, layers=1)
    print("Original spec:", spec)

    # Build model
    model = ArchitectureWrapper(spec)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward
    x = torch.randn(4, 16, 64)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")

    # Test mutations
    print("\nTesting mutations:")
    for mutation in ['add_bottleneck', 'add_sparse_layer', 'reduce_rank']:
        mutated = mutate_architecture(spec, mutation)
        print(f"\n{mutation}: {len(mutated['blocks'])} blocks")

    print("\n✅ Architecture system ready!")
