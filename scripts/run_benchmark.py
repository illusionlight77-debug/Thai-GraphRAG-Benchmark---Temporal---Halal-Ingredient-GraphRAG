"""Run the GraphRAG-vs-vanilla benchmark. Thin wrapper.

    python -m scripts.run_benchmark --questions thai_eval.example.jsonl [--judge]
"""
from thaigraphrag.benchmark.run_benchmark import main

if __name__ == "__main__":
    main()
