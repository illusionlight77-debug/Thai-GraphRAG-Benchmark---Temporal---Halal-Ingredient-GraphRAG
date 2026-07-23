"""Extensions built on top of the core (A):

  temporal          — B: time-aware GraphRAG (answer "as of <year>")
  halal_ingredient  — C: explainable ingredient→source→ruling GraphRAG

Both reuse core clients, the Retriever interface, and the benchmark harness. Each
is separable: you can run A alone, or A+B, or A+C.
"""
