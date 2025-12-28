
import pandas as pd
import json
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from transphaser.em import EMHaplotypePhaser

def calculate_mae(true_freq_df, pred_df, age_groups):
    maes = {}
    for age in age_groups:
        t = true_freq_df[true_freq_df['AgeGroup'] == age]
        if 'AgeGroup' in pred_df.columns:
            p = pred_df[pred_df['AgeGroup'] == age]
        else:
            p = pred_df # Marginal used for all ages
        
        # We need to ensure we compare correctly.
        # true_freq_df has explicit 0s? No, it lists all haplotypes.
        # But predicted might miss some.
        
        # Merge on Haplotype
        merged = t.merge(p, on='Haplotype', how='outer', suffixes=('_true', '_pred'))
        merged['TrueFrequency'] = merged['TrueFrequency'].fillna(0)
        merged['PredictedFrequency'] = merged['PredictedFrequency'].fillna(0)
        
        mae = np.mean(np.abs(merged['TrueFrequency'] - merged['PredictedFrequency']))
        maes[age] = mae
    return maes

df_unphased = pd.read_csv('examples/data_age_dependent_hard/unphased.csv')
true_freq = pd.read_csv('examples/data_age_dependent_hard/true_frequencies.csv')
locus_columns = ['HLA-A', 'HLA-C', 'HLA-B', 'HLA-DRB1', 'HLA-DQB1', 'HLA-DPB1']

# Prepare genotypes
genotypes = []
for _, row in df_unphased.iterrows():
    g = []
    for loc in locus_columns:
        alleles = row[loc].split('/')
        g.append((alleles[0], alleles[1]))
    genotypes.append(g)

print("Running EM...")
em = EMHaplotypePhaser(tolerance=1e-5, max_iterations=50)
em.fit(genotypes)
preds = em.predict(genotypes)

all_haps = []
for p in preds:
    h1 = '_'.join(p[0]) if isinstance(p[0], (list, tuple)) else p[0]
    h2 = '_'.join(p[1]) if isinstance(p[1], (list, tuple)) else p[1]
    all_haps.extend([h1, h2])

counts = pd.Series(all_haps).value_counts()
freqs = counts / counts.sum()
pred_df = freqs.reset_index()
pred_df.columns = ['Haplotype', 'PredictedFrequency']

ages = ['20-35', '36-50', '51-65', '66+']
maes = calculate_mae(true_freq, pred_df, ages)
print(json.dumps(maes, indent=2))
