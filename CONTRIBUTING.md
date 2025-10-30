# Contributing to NAS-HGM

Thank you for your interest in contributing to Neural Architecture Search via Huxley-Gödel Machine!

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Include minimal reproducible examples
- Specify your environment (Python version, PyTorch version, hardware)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-primitive`)
3. Make your changes
4. Add tests if applicable (`tests/` directory)
5. Ensure code runs without errors
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request with description

### Code Style

- Follow existing code style (no strict formatter required)
- Add docstrings for new functions/classes
- Keep functions focused and modular
- Avoid excessive comments - code should be self-documenting

### Areas for Contribution

**High Priority**:
- New primitive types for synthesis
- Improved rule extraction methods
- Better dimension handling in primitives
- Performance optimizations

**Medium Priority**:
- Additional evaluation metrics
- Visualization tools for search progress
- Documentation improvements
- More test cases

**Research**:
- Alternative synthesis strategies
- Different comparison operator types
- Meta-learning improvements
- Theoretical analysis

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/NAS-HGM.git
cd NAS-HGM

# Install dependencies
pip install -r requirements.txt

# Run tests
python tests/quick_test.py

# Run a quick search
python arch_search.py --batch 8 --generations 100 --time_limit 120
```

## Questions?

Open an issue for questions or discussions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
