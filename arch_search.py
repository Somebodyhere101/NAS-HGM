#!/usr/bin/env python3
"""
Convenience entry point for architecture search.
Allows running from project root: python arch_search.py [args]
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nas_hgm.arch_search import main

if __name__ == "__main__":
    main()
