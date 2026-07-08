"""Training pipeline for a local, self-trained part-embedding model.

Everything here is offline-testable except the actual network image fetch and
the actual torch training step, both of which are optional/lazy-imported and
injectable, matching the rest of the project's testing discipline.
"""
