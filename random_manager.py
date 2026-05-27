import random
import numpy as np

class RandomManager:
    """Gestiona un generador de números aleatorios independiente."""
    def __init__(self, seed: int):
        self.py_rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)