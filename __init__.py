"""
Questions Agent Platform (reference implementation).

This package provides a lightweight, production-oriented blueprint for:
- selecting daily questions (5 core + optional extra batches)
- scoring validated scales and tracking within-person baselines
- producing structured evidence for Anifold fusion ("projection", not "diagnosis")

The implementation uses Python standard library only (plus SQLite for persistence)
to keep the reference runnable in minimal environments.
"""
