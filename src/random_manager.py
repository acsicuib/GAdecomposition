"""
===============================================================================
File: random_manager.py
Description: Centralizes and manages random number generators (RNG) for reproducible simulation runs across different components.
Role: Class Definition (Utility)
Execution: Imported as a module
Inputs: Seed integers from configuration.
Outputs: Instances of isolated random and numpy.random generators.

Authors: Carlos Guerrero and Isaac Lera
Date: June 2026
License: MIT License
===============================================================================
"""
import random
import numpy as np

class RandomManager:
    """Gestiona un generador de números aleatorios independiente."""
    def __init__(self, seed: int):
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)