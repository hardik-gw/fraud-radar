"""Fraud Radar — shared code for training and serving.

Everything the API needs at scoring time lives here rather than in a notebook,
so serving cannot drift away from training.
"""

__version__ = "0.1.0"
