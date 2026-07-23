"""Build the KG + vector index. Thin wrapper.

    python -m scripts.build_kg [--reset] [--only mosque.csv] [--no-embed] [--limit N]
"""
from thaigraphrag.kg.build_kg import main

if __name__ == "__main__":
    main()
