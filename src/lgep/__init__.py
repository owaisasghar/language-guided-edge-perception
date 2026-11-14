"""Language-Guided Multi-Camera Edge Perception (lgep)."""

import os as _os

# One OpenBLAS thread unless the user chose otherwise (effective only if NumPy is not imported yet; the entry
# scripts set it first thing). See docs/milestone8.md: OpenBLAS's thread pool spins on small matrix products.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "0.8.0"
