"""Extension B — build the temporal layer. Thin wrapper.

    python -m scripts.build_temporal_kg

Requires the core KG (`python -m scripts.build_kg`) so Product nodes exist to annotate.
"""
from thaigraphrag.extensions.temporal.temporal_kg import build

if __name__ == "__main__":
    build()
