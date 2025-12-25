"""
Realistic HLA data generator with linkage disequilibrium and population structure.

Simulates biologically realistic HLA haplotypes based on:
1. Real HLA allele frequencies from published data
2. Linkage disequilibrium (LD) between loci
3. Population-specific haplotype frequencies
4. Hardy-Weinberg equilibrium
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class RealisticHLAGenerator:
    """
    Generates realistic HLA genotypes with proper LD and population structure.
    
    Based on patterns from:
    - 1000 Genomes Project
    - dbMHC database
    - NMDP Haplotype Frequency database
    """
    
    LOCI = ["HLA-A", "HLA-C", "HLA-B", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"]
    
    # Expanded HLA haplotype frequencies for 6 loci
    COMMON_HAPLOTYPES = {
        'EUR': [
            (('A*01:01', 'C*07:01', 'B*08:01', 'DRB1*03:01', 'DQB1*02:01', 'DPB1*04:01'), 0.09),
            (('A*03:01', 'C*07:02', 'B*07:02', 'DRB1*15:01', 'DQB1*06:02', 'DPB1*04:01'), 0.07),
            (('A*02:01', 'C*07:01', 'B*08:01', 'DRB1*03:01', 'DQB1*02:01', 'DPB1*04:01'), 0.05),
            (('A*02:01', 'C*03:04', 'B*40:01', 'DRB1*07:01', 'DQB1*02:02', 'DPB1*04:01'), 0.04),
            (('A*01:01', 'C*06:02', 'B*57:01', 'DRB1*07:01', 'DQB1*03:03', 'DPB1*04:01'), 0.04),
            (('A*02:01', 'C*05:01', 'B*44:02', 'DRB1*04:01', 'DQB1*03:02', 'DPB1*04:01'), 0.04),
            (('A*03:01', 'C*04:01', 'B*35:01', 'DRB1*01:01', 'DQB1*05:01', 'DPB1*04:01'), 0.03),
            (('A*11:01', 'C*04:01', 'B*35:01', 'DRB1*01:01', 'DQB1*05:01', 'DPB1*04:01'), 0.03),
            (('A*29:02', 'C*16:01', 'B*44:03', 'DRB1*07:01', 'DQB1*02:02', 'DPB1*04:01'), 0.02),
            (('A*30:01', 'C*06:02', 'B*13:02', 'DRB1*07:01', 'DQB1*02:02', 'DPB1*04:01'), 0.02),
            (('A*02:01', 'C*07:02', 'B*07:02', 'DRB1*15:01', 'DQB1*06:02', 'DPB1*04:01'), 0.02),
            (('A*24:02', 'C*07:01', 'B*08:01', 'DRB1*03:01', 'DQB1*02:01', 'DPB1*04:01'), 0.01),
        ],
        'AFR': [
            (('A*30:02', 'C*17:01', 'B*42:01', 'DRB1*03:02', 'DQB1*04:02', 'DPB1*01:01'), 0.11),
            (('A*02:01', 'C*06:02', 'B*58:02', 'DRB1*03:01', 'DQB1*02:01', 'DPB1*01:01'), 0.09),
            (('A*23:01', 'C*07:01', 'B*49:01', 'DRB1*08:02', 'DQB1*04:02', 'DPB1*01:01'), 0.07),
            (('A*30:01', 'C*17:01', 'B*42:01', 'DRB1*11:02', 'DQB1*03:01', 'DPB1*01:01'), 0.06),
            (('A*02:05', 'C*04:01', 'B*53:01', 'DRB1*13:02', 'DQB1*06:04', 'DPB1*01:01'), 0.05),
            (('A*34:02', 'C*04:01', 'B*53:01', 'DRB1*13:01', 'DQB1*06:03', 'DPB1*01:01'), 0.04),
            (('A*68:02', 'C*02:02', 'B*15:03', 'DRB1*13:01', 'DQB1*06:03', 'DPB1*01:01'), 0.04),
            (('A*33:01', 'C*06:02', 'B*58:02', 'DRB1*03:02', 'DQB1*04:02', 'DPB1*01:01'), 0.03),
            (('A*29:02', 'C*06:02', 'B*45:01', 'DRB1*11:01', 'DQB1*03:01', 'DPB1*01:01'), 0.03),
            (('A*68:01', 'C*07:01', 'B*57:03', 'DRB1*11:01', 'DQB1*03:01', 'DPB1*01:01'), 0.02),
        ],
        'ASN': [
            (('A*02:07', 'C*01:02', 'B*46:01', 'DRB1*08:03', 'DQB1*06:01', 'DPB1*02:01'), 0.13),
            (('A*11:01', 'C*03:04', 'B*15:01', 'DRB1*12:02', 'DQB1*03:01', 'DPB1*04:01'), 0.10),
            (('A*24:02', 'C*01:02', 'B*54:01', 'DRB1*04:05', 'DQB1*03:02', 'DPB1*04:01'), 0.09),
            (('A*33:03', 'C*03:02', 'B*58:01', 'DRB1*03:01', 'DQB1*02:01', 'DPB1*04:01'), 0.08),
            (('A*02:01', 'C*07:02', 'B*40:01', 'DRB1*09:01', 'DQB1*03:03', 'DPB1*05:01'), 0.06),
            (('A*11:01', 'C*03:04', 'B*40:02', 'DRB1*04:03', 'DQB1*03:02', 'DPB1*05:01'), 0.05),
            (('A*24:02', 'C*07:02', 'B*52:01', 'DRB1*15:02', 'DQB1*06:01', 'DPB1*04:01'), 0.04),
            (('A*11:01', 'C*07:02', 'B*13:01', 'DRB1*07:01', 'DQB1*02:02', 'DPB1*04:01'), 0.03),
            (('A*02:03', 'C*07:02', 'B*38:02', 'DRB1*16:02', 'DQB1*05:02', 'DPB1*04:01'), 0.03),
            (('A*26:01', 'C*03:04', 'B*15:01', 'DRB1*15:01', 'DQB1*06:02', 'DPB1*04:01'), 0.02),
        ],
        'HIS': [
            (('A*02:01', 'C*04:01', 'B*35:01', 'DRB1*04:07', 'DQB1*03:02', 'DPB1*04:01'), 0.07),
            (('A*24:02', 'C*04:01', 'B*35:01', 'DRB1*04:07', 'DQB1*03:02', 'DPB1*04:01'), 0.06),
            (('A*02:01', 'C*16:01', 'B*44:03', 'DRB1*07:01', 'DQB1*02:02', 'DPB1*04:01'), 0.05),
            (('A*68:03', 'C*07:02', 'B*39:05', 'DRB1*04:04', 'DQB1*03:02', 'DPB1*04:01'), 0.04),
            (('A*02:06', 'C*07:02', 'B*39:02', 'DRB1*08:02', 'DQB1*04:02', 'DPB1*04:01'), 0.04),
            (('A*02:01', 'C*01:02', 'B*27:05', 'DRB1*01:01', 'DQB1*05:01', 'DPB1*04:01'), 0.03),
        ]
    }
    
    # Rare/random alleles expanded
    RARE_ALLELES = {
        'HLA-A': [
            'A*01:02', 'A*02:02', 'A*02:03', 'A*02:06', 'A*03:02', 'A*11:02', 'A*23:02', 
            'A*24:03', 'A*25:01', 'A*26:01', 'A*29:02', 'A*31:01', 'A*32:01', 'A*33:01', 
            'A*66:01', 'A*68:01', 'A*68:03', 'A*74:01', 'A*80:01', 'A*36:01', 'A*43:01',
            'A*02:05', 'A*02:07', 'A*02:11', 'A*11:01', 'A*24:02', 'A*30:01', 'A*30:02'
        ],
        'HLA-C': [
            'C*01:02', 'C*02:02', 'C*03:02', 'C*03:03', 'C*03:04', 'C*04:01', 'C*05:01',
            'C*06:02', 'C*07:01', 'C*07:02', 'C*08:01', 'C*12:02', 'C*12:03', 'C*14:02',
            'C*15:02', 'C*16:01', 'C*17:01', 'C*18:01', 'C*07:04', 'C*08:02'
        ],
        'HLA-B': [
            'B*07:05', 'B*08:02', 'B*13:02', 'B*14:01', 'B*14:02', 'B*18:01', 'B*27:05',
            'B*35:02', 'B*35:03', 'B*37:01', 'B*38:01', 'B*39:01', 'B*41:01', 'B*44:03',
            'B*45:01', 'B*47:01', 'B*48:01', 'B*50:01', 'B*51:01', 'B*55:01', 'B*56:01',
            'B*57:03', 'B*38:02', 'B*39:02', 'B*39:05', 'B*15:03', 'B*58:02', 'B*57:01',
            'B*07:02', 'B*08:01', 'B*13:01', 'B*15:01', 'B*40:01', 'B*40:02', 'B*44:02',
            'B*49:01', 'B*52:01', 'B*53:01', 'B*54:01', 'B*58:01', 'B*42:01'
        ],
        'HLA-DRB1': [
            'DRB1*01:02', 'DRB1*01:03', 'DRB1*03:03', 'DRB1*04:02', 'DRB1*04:04', 'DRB1*04:07',
            'DRB1*07:03', 'DRB1*08:01', 'DRB1*09:02', 'DRB1*10:01', 'DRB1*11:01', 'DRB1*11:03',
            'DRB1*11:04', 'DRB1*12:01', 'DRB1*13:03', 'DRB1*14:01', 'DRB1*15:03', 'DRB1*16:01',
            'DRB1*04:05', 'DRB1*08:04', 'DRB1*16:02', 'DRB1*08:03', 'DRB1*11:02', 'DRB1*08:02',
            'DRB1*01:01', 'DRB1*03:01', 'DRB1*03:02', 'DRB1*04:01', 'DRB1*04:03', 'DRB1*07:01',
            'DRB1*09:01', 'DRB1*11:01', 'DRB1*12:02', 'DRB1*13:01', 'DRB1*13:02', 'DRB1*15:01',
            'DRB1*15:02'
        ],
        'HLA-DQB1': [
            'DQB1*02:01', 'DQB1*02:02', 'DQB1*03:01', 'DQB1*03:02', 'DQB1*03:03', 'DQB1*04:02',
            'DQB1*05:01', 'DQB1*05:02', 'DQB1*06:01', 'DQB1*06:02', 'DQB1*06:03', 'DQB1*06:04',
            'DQB1*06:09', 'DQB1*04:01', 'DQB1*05:03'
        ],
        'HLA-DPB1': [
            'DPB1*01:01', 'DPB1*02:01', 'DPB1*03:01', 'DPB1*04:01', 'DPB1*04:02', 'DPB1*05:01',
            'DPB1*06:01', 'DPB1*09:01', 'DPB1*10:01', 'DPB1*11:01', 'DPB1*13:01', 'DPB1*14:01',
            'DPB1*17:01', 'DPB1*20:01', 'DPB1*02:02'
        ]
    }
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed."""
        self.rng = np.random.RandomState(seed)
        self._haplotype_pools = {}
        self._build_haplotype_pools()
    
    def _build_haplotype_pools(self):
        """Build weighted haplotype pools for each population."""
        for pop, haplotypes in self.COMMON_HAPLOTYPES.items():
            # Extract haplotypes and weights
            haps = [h for h, _ in haplotypes]
            weights = np.array([w for _, w in haplotypes])
            
            # Normalize to account for rare haplotypes (keep 40% for rare/background)
            # Increased rare fraction to increase allelic richness
            common_fraction = 0.60
            weights = weights / weights.sum() * common_fraction
            
            # Add rare haplotype probability
            rare_prob = 1.0 - weights.sum()
            
            self._haplotype_pools[pop] = {
                'common_haplotypes': haps,
                'common_weights': weights,
                'rare_probability': rare_prob
            }
    
    def _generate_rare_haplotype(self) -> Tuple[str, ...]:
        """Generate a rare/random haplotype using background allele frequencies."""
        # Simple random choice from allele pool for each locus
        return tuple(self.rng.choice(self.RARE_ALLELES[locus]) for locus in self.LOCI)
    
    def sample_haplotype(self, population: str) -> Tuple[str, ...]:
        """
        Sample a single haplotype from the population-specific distribution.
        
        Args:
            population: Population code ('EUR', 'AFR', 'ASN', 'HIS')
            
        Returns:
            Tuple of alleles (A, C, B, DRB1, DQB1, DPB1)
        """
        pool = self._haplotype_pools[population]
        
        # Decide if common or rare
        if self.rng.rand() < (1 - pool['rare_probability']):
            # Sample from common haplotypes
            idx = self.rng.choice(
                len(pool['common_haplotypes']),
                p=pool['common_weights'] / pool['common_weights'].sum()
            )
            return pool['common_haplotypes'][idx]
        else:
            # Generate rare haplotype
            # To increase structure even in 'rare' ones, we can occasionally 
            # recombine common haplotypes
            if self.rng.rand() < 0.3:
                # Recombinant: take half from one common, half from another
                idx1 = self.rng.choice(len(pool['common_haplotypes']))
                idx2 = self.rng.choice(len(pool['common_haplotypes']))
                h1 = pool['common_haplotypes'][idx1]
                h2 = pool['common_haplotypes'][idx2]
                split = self.rng.randint(1, len(self.LOCI))
                return h1[:split] + h2[split:]
            else:
                return self._generate_rare_haplotype()
    
    def generate_genotype(self, population: str) -> Tuple[Tuple, Tuple]:
        """Generate a realistic genotype (haplotype pair) for a population."""
        h1 = self.sample_haplotype(population)
        h2 = self.sample_haplotype(population)
        return (h1, h2)
    
    def generate_dataset(
        self,
        n_samples: int,
        populations: Optional[List[str]] = None,
        age_groups: Optional[List[str]] = None,
        output_dir: str = 'examples/data'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate realistic HLA dataset with LD and population structure."""
        if populations is None:
            populations = list(self.COMMON_HAPLOTYPES.keys())
        if age_groups is None:
            age_groups = ['0-18', '19-40', '41-65', '65+']
        
        samples = []
        
        for i in range(n_samples):
            # Sample population and age group
            population = self.rng.choice(populations)
            age_group = self.rng.choice(age_groups)
            
            # Generate genotype
            h1, h2 = self.generate_genotype(population)
            
            # Build sample dictionary
            sample = {
                'IndividualID': f'SAMPLE_{i+1:05d}',
                'Population': population,
                'AgeGroup': age_group,
                'Haplotype1': '_'.join(h1),
                'Haplotype2': '_'.join(h2)
            }
            
            # Add genotype columns for each locus
            for idx, locus in enumerate(self.LOCI):
                sample[locus] = '/'.join(sorted([h1[idx], h2[idx]]))
            
            samples.append(sample)
        
        # Create DataFrames
        all_data = pd.DataFrame(samples)
        
        # Unphased (genotypes only)
        unphased_cols = ['IndividualID'] + self.LOCI + ['Population', 'AgeGroup']
        unphased_df = all_data[unphased_cols].copy()
        
        # Phased (ground truth haplotypes)
        phased_df = all_data[['IndividualID', 'Haplotype1', 'Haplotype2',
                               'Population', 'AgeGroup']].copy()
        
        return unphased_df, phased_df
    
    def analyze_ld(self, phased_df: pd.DataFrame) -> Dict:
        """Analyze linkage disequilibrium and diversity in generated data."""
        haplotype_counts = defaultdict(int)
        allele_counts = {locus: defaultdict(int) for locus in self.LOCI}
        
        for _, row in phased_df.iterrows():
            h1_alleles = row['Haplotype1'].split('_')
            h2_alleles = row['Haplotype2'].split('_')
            
            haplotype_counts[row['Haplotype1']] += 1
            haplotype_counts[row['Haplotype2']] += 1
            
            for idx, locus in enumerate(self.LOCI):
                allele_counts[locus][h1_alleles[idx]] += 1
                allele_counts[locus][h2_alleles[idx]] += 1
        
        total_haps = sum(haplotype_counts.values())
        top_haplotypes = sorted(haplotype_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        locus_diversity = {
            locus: len(counts) for locus, counts in allele_counts.items()
        }
        
        return {
            'total_unique_haplotypes': len(haplotype_counts),
            'alleles_per_locus': locus_diversity,
            'top_15_haplotypes': [
                {'haplotype': h, 'frequency': c/total_haps} 
                for h, c in top_haplotypes
            ],
            'top_1_frequency': top_haplotypes[0][1] / total_haps,
            'top_5_cumulative': sum(c for _, c in top_haplotypes[:5]) / total_haps,
            'total_alleles': sum(locus_diversity.values())
        }


def main():
    """Generate realistic HLA dataset."""
    import os
    import json
    
    print("=" * 70)
    print("GENERATING ENHANCED REALISTIC HLA DATASET (6 LOCI)")
    print("=" * 70)
    
    generator = RealisticHLAGenerator(seed=42)
    
    # Generate dataset (increased to 10,000 for more structure/alleles)
    n_samples = 10000
    print(f"\nGenerating {n_samples} samples with improved LD and population structure...")
    unphased_df, phased_df = generator.generate_dataset(
        n_samples=n_samples,
        populations=['EUR', 'AFR', 'ASN', 'HIS'],
        output_dir='examples/data'
    )
    
    # Analyze LD
    print("\nAnalyzing linkage disequilibrium and diversity...")
    stats = generator.analyze_ld(phased_df)
    
    print(f"\n✅ Generated {len(unphased_df)} samples")
    print(f"   - Loci: {', '.join(generator.LOCI)}")
    print(f"   - Unique alleles: {stats['total_alleles']}")
    print(f"   - Unique haplotypes: {stats['total_unique_haplotypes']}")
    print(f"   - Top haplotype frequency: {stats['top_1_frequency']:.1%}")
    print(f"   - Top 5 cumulative: {stats['top_5_cumulative']:.1%}")
    
    print("\nAlleles per locus:")
    for locus, count in stats['alleles_per_locus'].items():
        print(f"  {locus:10}: {count} alleles")
    
    print("\nTop 10 most common haplotypes:")
    for i, item in enumerate(stats['top_15_haplotypes'][:10], 1):
        print(f"  {i}. {item['haplotype']}: {item['frequency']:.2%}")
    
    # Save files
    os.makedirs('examples/data', exist_ok=True)
    
    unphased_path = 'examples/data/realistic_genotypes_unphased.csv'
    phased_path = 'examples/data/realistic_haplotypes_phased.csv'
    stats_path = 'examples/data/realistic_data_stats.json'
    
    unphased_df.to_csv(unphased_path, index=False)
    phased_df.to_csv(phased_path, index=False)
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Saved:")
    print(f"   - {unphased_path}")
    print(f"   - {phased_path}")
    print(f"   - {stats_path}")
    
    # Show population distribution
    print("\nPopulation distribution:")
    print(unphased_df['Population'].value_counts())
    
    print("\n" + "=" * 70)
    print("ENHANCED DATASET READY")
    print("=" * 70)


if __name__ == '__main__':
    main()
