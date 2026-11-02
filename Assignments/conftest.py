import sys
import os

# Ensure the Assignments directory is on the path so `helpers` is importable
# from any lab subdirectory when pytest is invoked from the Assignments root.
sys.path.insert(0, os.path.dirname(__file__))
