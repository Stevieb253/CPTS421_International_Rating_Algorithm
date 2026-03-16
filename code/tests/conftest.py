import sys
import os

# Add the 'code' directory to sys.path so that 'services' is importable
# regardless of where pytest is launched from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))