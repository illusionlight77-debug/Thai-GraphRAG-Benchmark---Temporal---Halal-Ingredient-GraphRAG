"""Extension C — build the halal ingredient layer. Thin wrapper.

    python -m scripts.build_ingredient_kg

Requires the core KG (`python -m scripts.build_kg`) so Product nodes exist to link to.
"""
from thaigraphrag.extensions.halal_ingredient.ingredient_kg import build

if __name__ == "__main__":
    build()
