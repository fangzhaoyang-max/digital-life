"""
Genetic encoding system for digital life
"""

import hashlib
import random
from typing import Dict


class GeneticEncoder:
    """Genetic encoding system for digital life parameters"""

    def __init__(self):
        self.gene_map = {
            'metabolism': (0, 32),
            'mutation_rate': (32, 64),
            'learning_rate': (64, 96),
            'exploration': (96, 128),
            'defense': (128, 160),
            # Strategy genes
            'pop_size': (160, 192),
            'crossover': (192, 224),
            'op_bias': (224, 256),
            'timeout': (256, 288)
        }

    def decode(self, dna: str) -> Dict:
        """Decode DNA to executable parameters"""
        params = {}
        for param, (start, end) in self.gene_map.items():
            segment = dna[start:end]
            if not segment:
                continue
            try:
                hash_val = int(hashlib.sha256(segment.encode()).hexdigest()[:8], 16)
                normalized = (hash_val % 10000) / 10000.0
                if param == 'metabolism':
                    params[param] = 0.5 + normalized  # 0.5-1.5
                elif param == 'mutation_rate':
                    params[param] = 0.01 + normalized * 0.1  # 0.01-0.11
                elif param == 'learning_rate':
                    params[param] = 0.001 + normalized * 0.01  # 0.001-0.011
                elif param == 'exploration':
                    params[param] = normalized  # 0-1
                elif param == 'defense':
                    params[param] = normalized * 2  # 0-2
                elif param == 'pop_size':
                    params[param] = int(4 + normalized * 28)  # 4-32
                elif param == 'crossover':
                    params[param] = 0.1 + normalized * 0.7   # 0.1-0.8
                elif param == 'op_bias':
                    params[param] = normalized               # 0-1
                elif param == 'timeout':
                    params[param] = int(400 + normalized * 1200)  # 400-1600 ms
            except Exception:
                params[param] = 0.5
        return params

    def encode(self, params: Dict) -> str:
        """Encode parameters to DNA fragments"""
        dna_segments = []
        for param, (start, end) in self.gene_map.items():
            value = params.get(param, 0.5)
            if param == 'metabolism':
                scaled = int((value - 0.5) * 10000)
            elif param == 'mutation_rate':
                scaled = int((value - 0.01) * 100000)
            elif param == 'learning_rate':
                scaled = int((value - 0.001) * 1000000)
            elif param == 'exploration':
                scaled = int(value * 10000)
            elif param == 'defense':
                scaled = int(value * 5000)
            elif param == 'pop_size':
                scaled = int(value)
            elif param == 'crossover':
                scaled = int(value * 10000)
            elif param == 'op_bias':
                scaled = int(value * 10000)
            elif param == 'timeout':
                scaled = int(value)
            else:
                scaled = int(value * 10000)
            segment = hashlib.sha256(str(scaled).encode()).hexdigest()[:32]
            dna_segments.append(segment)
        return ''.join(dna_segments)

    def recombine(self, dna1: str, dna2: str) -> str:
        """Recombine two DNA sequences"""
        new_dna = []
        for i in range(0, min(len(dna1), len(dna2)), 32):
            segment1 = dna1[i:i + 32]
            segment2 = dna2[i:i + 32]
            if random.random() < 0.5:
                new_dna.append(segment1)
            else:
                new_dna.append(segment2)
        if new_dna and random.random() < 0.1:
            pos = random.randint(0, len(new_dna) - 1)
            new_dna[pos] = hashlib.sha256(new_dna[pos].encode()).hexdigest()[:32]
        return ''.join(new_dna)