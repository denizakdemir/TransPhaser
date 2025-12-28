
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import os

class RealisticAgeDependentHLAGenerator:
    LOCI = ["HLA-A", "HLA-C", "HLA-B", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"]
    AGE_GROUPS = ['20-35', '36-50', '51-65', '66+']
    
    def __init__(self, seed: int = 42, age_effect_strength: float = 0.3):
        self.rng = np.random.RandomState(seed)
        self.age_effect_strength = age_effect_strength
        self.haplotypes = self._generate_haplotypes()
        self.base_frequencies = self._generate_base_frequencies()
        self.age_frequencies = self._generate_age_frequencies()

    def _generate_haplotypes(self, n=50):
        # Generate stable synthetic haplotypes
        haps = []
        for i in range(n):
            h = []
            for loc in self.LOCI:
                # Deterministic generation based on index to keep consistent across runs if needed
                # But here we use rng
                allele = f"{loc.split('-')[1]}*{self.rng.randint(1,99):02d}:{self.rng.randint(1,99):02d}"
                h.append(allele)
            haps.append(tuple(h))
        return haps

    def _generate_base_frequencies(self):
        # Power law (Zipf) to simulate realistic rarity
        n = len(self.haplotypes)
        ranks = np.arange(1, n + 1)
        weights = 1 / (ranks ** 1.2) # Slightly less extreme than 1.5 to have more medium freq
        return weights / weights.sum()

    def _generate_age_frequencies(self):
        # Apply age effect
        freqs = {}
        n_ages = len(self.AGE_GROUPS)
        
        # Assign random trends to haplotypes: -1 (decrease), 0 (neutral), 1 (increase)
        trends = self.rng.choice([-1, 0, 1], size=len(self.haplotypes), p=[0.4, 0.2, 0.4])
        
        for i, age_group in enumerate(self.AGE_GROUPS):
            # Linearly scale frequency: freq_age = base * (1 + strength * trend * (i - center))
            center = (n_ages - 1) / 2.0
            age_factor = (i - center) / center # -1 to 1
            
            # Modifier
            modifiers = 1.0 + (trends * age_factor * self.age_effect_strength)
            modifiers = np.maximum(modifiers, 0.001) # Ensure positive
            
            # Apply to base
            age_freqs = self.base_frequencies * modifiers
            age_freqs /= age_freqs.sum() # Renormalize
            freqs[age_group] = age_freqs
            
        return freqs

    def generate_dataset(self, n_per_age_group=2500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        samples = []
        true_freq_data = []

        for age_group in self.AGE_GROUPS:
            freqs = self.age_frequencies[age_group]
            
            # Record true frequencies
            for h, f in zip(self.haplotypes, freqs):
                h_str = '_'.join(h)
                true_freq_data.append({
                    'AgeGroup': age_group,
                    'Haplotype': h_str,
                    'TrueFrequency': f
                })

            # Sample haplotypes from the age-specific distribution
            indices = np.arange(len(self.haplotypes))
            
            # Generate n_per_age_group samples
            h1_indices = self.rng.choice(indices, size=n_per_age_group, p=freqs)
            h2_indices = self.rng.choice(indices, size=n_per_age_group, p=freqs)
            
            for i in range(n_per_age_group):
                h1 = self.haplotypes[h1_indices[i]]
                h2 = self.haplotypes[h2_indices[i]]
                
                sample = {
                    'IndividualID': f'SAMPLE_{age_group}_{i:05d}',
                    'Population': 'EUR',
                    'AgeGroup': age_group,
                    'Haplotype1': '_'.join(h1),
                    'Haplotype2': '_'.join(h2)
                }
                for idx, locus in enumerate(self.LOCI):
                    sample[locus] = '/'.join(sorted([h1[idx], h2[idx]]))
                samples.append(sample)

        # Create DataFrames
        all_data = pd.DataFrame(samples)
        
        unphased_cols = ['IndividualID'] + self.LOCI + ['Population', 'AgeGroup']
        unphased_df = all_data[unphased_cols].copy()
        
        phased_df = all_data[['IndividualID', 'Haplotype1', 'Haplotype2', 'Population', 'AgeGroup']].copy()
        
        true_freq_df = pd.DataFrame(true_freq_data)
        
        return unphased_df, phased_df, true_freq_df

if __name__ == "__main__":
    # Generate the dataset used by Experiment 2
    # Increased strength from default (probably) to 0.4
    print("Generating new age-dependent dataset...")
    gen = RealisticAgeDependentHLAGenerator(seed=123, age_effect_strength=1.5) 
    
    # Generate 2500 per age group -> 10000 total (consistent with other experiments)
    unphased, phased, true_freqs = gen.generate_dataset(n_per_age_group=2500)
    
    output_dir = 'examples/data_age_dependent_hard'
    os.makedirs(output_dir, exist_ok=True)
    
    unphased.to_csv(os.path.join(output_dir, 'unphased.csv'), index=False)
    phased.to_csv(os.path.join(output_dir, 'phased.csv'), index=False)
    true_freqs.to_csv(os.path.join(output_dir, 'true_frequencies.csv'), index=False)
    
    print(f"Saved {len(unphased)} samples to {output_dir}")
