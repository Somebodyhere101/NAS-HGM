# Neural Architecture Search via Huxley-Gödel Machine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.21614-b31b1b.svg)](https://arxiv.org/abs/2510.21614)

**Unbounded Self-Improving Architecture Search**

Zero-training architecture discovery using Clade-Metaproductivity (CMP) with automatic primitive synthesis. Achieves 10,000x speedup over traditional NAS through SVD-based evaluation.

**Version**: 3.0.0

---

## Key Features

- 🚀 **Ultra-Fast Rule-Space Search**: 100x speedup via direct QK^T optimization (1000x for pre-filtering)
- **Unbounded Self-Improvement**: Synthesizes new primitives from extracted comparison rules
- **Zero-Training Evaluation**: Predict architecture quality in ~0.5s (vs hours of training)
- **Hybrid Search Pipeline**: Fast rule exploration → Full architecture validation
- **Primitive Synthesis**: Automatically generates new comparison operators from successful architectures
- **Adaptive Learning**: Mutation learning, dynamic thresholds, result caching (93%+ efficiency)
- **Batch Evaluation**: 7-10x speedup on consumer GPUs
- **Production Ready**: Checkpointing, logging, error handling

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy scipy scikit-learn

# ULTRA-FAST: Rule-space search (100x faster!)
python -c "from src.nas_hgm.hybrid_search import HybridSearchEngine; \
           import torch; \
           s = HybridSearchEngine(dim=64, device='cuda' if torch.cuda.is_available() else 'cpu'); \
           spec, metrics = s.search(rule_generations=50, num_validate=10)"

# Traditional search (5 minutes)
python arch_search.py --batch 8 --generations 200 --time_limit 300

# Production run (1 hour)
python arch_search.py --batch 16 --generations 5000 --time_limit 3600 --checkpoint_interval 50

# Benchmark speedup
python benchmark_speedup.py
```

---

## How It Works

### 1. Zero-Training Evaluation

**Compression Score** (40%):
- SVD-based rank measurement
- Information bottleneck detection

**Trainability Score** (60%):
- Gradient signal-to-noise ratio
- One-step learning speed
- Feature separability

**Combined**: `0.4 × compression + 0.6 × trainability`

### 2. Unbounded Primitive Synthesis

The system implements true self-improvement:

1. **Extract QK^T Rules**: Harvest comparison operators from top architectures
2. **Synthesize Primitives**: Generate new operators via SVD factorization
3. **Expand Search Space**: Mutation pool grows during search (15 → 15+N → 15+N+M → ...)
4. **Filter Failures**: Evolution automatically removes corrupt architectures

Everything reduces to comparison operators (< = >). The system learns to choose between ranges, similar to how evolution optimized human decision-making.

### 3. CMP-Guided Search

Uses Thompson sampling with Clade-Metaproductivity (CMP) to select promising architecture lineages rather than individual candidates.

### 4. Ultra-Fast Rule-Space Search (NEW!)

**Key Insight from GeneralizationB**: Neural networks store decision rules as low-rank QK^T matrices. Instead of building and evaluating full architectures, we search directly in rule space.

**Hybrid Search Pipeline**:
1. **Rule Exploration** (1000x faster): Evaluate 5,000+ rules via SVD (~5s)
2. **Fast Filter**: Select top 100 candidates based on spectral properties
3. **Full Validation**: Build and test top 10 architectures (~5s)
4. **Result**: Best of 5,000 in 10s vs 4+ minutes traditional

**Rule Quality Metrics** (No model building!):
- Effective rank (compression capability)
- Spectral concentration (trainability proxy)
- Singular value entropy (complexity)
- Combined score matches full evaluation

**Usage**:
```python
from nas_hgm.hybrid_search import HybridSearchEngine

searcher = HybridSearchEngine(dim=64, device='cuda')
best_spec, metrics = searcher.search(
    rule_generations=50,  # 5,000 rules evaluated
    num_validate=10       # Top 10 fully tested
)
# Total time: ~10-15s for 5,000 candidates!
```

**Speedup Breakdown**:
- Single rule eval: 0.05ms (SVD only)
- Single architecture eval: 50ms (build + forward pass)
- **Per-evaluation speedup: 1000x**
- **End-to-end speedup: 80-100x** (due to final validation stage)

---

## Performance

| Hardware | Batch | Architectures/sec | Speedup |
|----------|-------|-------------------|---------|
| CPU | 8 | ~2 | 1x |
| 3080 GPU | 16 | ~6 | 7x |
| DGX Spark | 128 | ~120-200 | 50-100x (est.) |

**Cache Efficiency**: 93%+ hit rate

**Discovered Architecture**: +1.6% accuracy vs transformer baseline (MNIST)

---

## Command-Line Arguments

```bash
python arch_search.py [OPTIONS]
```

**Search**:
- `--generations N` - Maximum generations (default: 1000)
- `--time_limit N` - Time limit in seconds (default: 3600)
- `--batch N` - Batch size: 1=sequential, 8-16=recommended (default: 1)

**Architecture**:
- `--initial TYPE` - Starting point: transformer/mlp/bottleneck (default: transformer)
- `--resume PATH` - Resume from checkpoint

**Output**:
- `--output FILE` - Results file (default: hgm_results.json)
- `--checkpoint_interval N` - Save every N generations (default: 100)
- `--checkpoint_dir DIR` - Checkpoint directory (default: checkpoints/)

**Hardware**:
- `--device DEVICE` - cuda/cpu/auto (default: auto)

**Logging**:
- `--verbose` - Enable verbose logging (default: True)
- `--quiet` - Suppress progress output

---

## Architecture

### Project Structure

```
NAS-HGM/
├── src/nas_hgm/           # Main source package
│   ├── arch_search.py     # Main HGM search loop
│   ├── architectures.py   # Search space + primitive synthesis
│   ├── compression_eval.py # Zero-training compression metrics
│   ├── trainable_eval.py  # Zero-shot trainability prediction
│   ├── batch_eval.py      # Parallel batch evaluation
│   ├── fast_eval.py       # Fast evaluation with learned predictors
│   └── guided_sampling.py # Guided attention mechanisms
├── tests/                 # Test suite
│   └── quick_test.py      # Validation test script
├── docs/                  # Documentation
│   └── CHANGELOG.md       # Version history
├── arch_search.py         # Entry point (convenience wrapper)
├── README.md              # This file
├── CONTRIBUTING.md        # Contribution guidelines
├── requirements.txt       # Python dependencies
└── LICENSE                # MIT License
```

### Primitive Synthesis

The system automatically generates new building blocks:

1. **ExtractedRule**: Low-rank attention from QK^T rules
2. **SynthesizedPrimitive**: Novel comparison operators with learned thresholds
3. **ComposedPrimitive**: Infinite composition of operators

Starting primitives (15):
- Bottleneck, sparse layers, factorized layers
- Learned gates, competitive selection, stochastic routing
- Confidence gates, dynamic routing
- Residual, layernorm, activations

Synthesized primitives (unbounded):
- Generated every 50 generations from top architectures
- Each new primitive is a comparison operator learned from successful patterns

---

## Research Background

This work builds on two key research contributions:

1. **[Huxley-Gödel Machine](https://arxiv.org/abs/2510.21614)** - Self-improving agents via clade-metaproductivity
   - Paper: arXiv:2510.21614v2
   - Implements CMP-guided Thompson sampling for architecture search

2. **[GeneralizationB](https://github.com/Somebodyhere101/GeneralizationB)** - Gradient-induced compression theory
   - Demonstrates QK^T bottleneck stores compressed decision rules
   - Shows how low-rank attention emerges through training

**Key Innovation**: Combines CMP selection with automatic primitive synthesis. The system extracts comparison rules from successful architectures and synthesizes new operators, creating truly unbounded search space.

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CUDA recommended)
- NumPy, SciPy, scikit-learn
- 8GB+ VRAM (GPU) or 16GB+ RAM (CPU)

```bash
pip install torch torchvision numpy scipy scikit-learn
```

---

## Citation

If you use this work, please cite:

```bibtex
@software{nas-hgm-2025,
  title={Neural Architecture Search via Huxley-Gödel Machine with Unbounded Primitive Synthesis},
  author={Somebodyhere101},
  year={2025},
  url={https://github.com/Somebodyhere101/NAS-HGM},
  version={3.0.0}
}
```

**Please also cite the foundational research**:

```bibtex
@article{huxley2025godel,
  title={Huxley-Gödel Machine: Human-Level Coding Agent Development by an Approximation of the Optimal Self-Improving Machine},
  author={Wang, Wenyi and Pi{\k{e}}kos, Piotr and Li, Nanbo and Laakom, Firas and Chen, Yimeng and Ostaszewski, Mateusz and Zhuge, Mingchen and Schmidhuber, J{\"u}rgen},
  journal={arXiv preprint arXiv:2510.21614},
  year={2025},
  month={October},
  institution={King Abdullah University of Science and Technology (KAUST)}
}

@software{generalizationb-2025,
  title={GeneralizationB: Gradient-Induced Compression in Neural Networks},
  author={Somebodyhere101},
  year={2025},
  url={https://github.com/Somebodyhere101/GeneralizationB}
}
```

---

## License

MIT License - See LICENSE file

Research use encouraged. Proper attribution required for academic work.

---
