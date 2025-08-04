#!/usr/bin/env python3
"""
Standalone script to run the business cycle analysis example.
This can be run without installing the package.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now import and run the example
from examples.business_cycle import main

if __name__ == "__main__":
    print("Running Business Cycle Analysis Example...")
    print("=" * 60)
    main()