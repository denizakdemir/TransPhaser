"""
CORRECTED Frequency Prediction Experiment

Proper experimental design:
  Dataset 1 (NO age effect): Test ALL methods (Frequency, EM, Beagle, TransPhaser)
  Dataset 2 (WITH age effect): Test ONLY TransPhaser ±age (others can't use age info)
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from transphaser.config import TransPhaserConfig
from transphaser.em import EMHaplotypePhaser
from transphaser.runner import TransPhaserRunner
from transphaser.data_preprocessing import AlleleTokenizer
from transphaser.evaluation import HLAPhasingMetrics

try:
    from transphaser.beagle_runner import BeagleRunner
except ImportError:
    BeagleRunner = None

try:
    from generate_realistic_age_data import RealisticAgeDependentHLAGenerator
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from generate_realistic_age_data import RealisticAgeDependentHLAGenerator


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'frequency_experiment.log')),
            logging.StreamHandler()
        ]
    )


def calculate_frequency_from_predictions(predictions: List[Tuple], age_labels: List[str] = None) -> pd.DataFrame:
    """Calculate haplotype frequencies from predicted haplotype pairs."""
    all_haplotypes = []
    all_ages = []
    
    for idx, (h1, h2) in enumerate(predictions):
        h1_str = '_'.join(h1) if isinstance(h1, (list, tuple)) else h1
        h2_str = '_'.join(h2) if isinstance(h2, (list, tuple)) else h2
        
        age = age_labels[idx] if age_labels else 'All'
        
        all_haplotypes.extend([h1_str, h2_str])
        all_ages.extend([age, age])
    
    df = pd.DataFrame({'Haplotype': all_haplotypes, 'AgeGroup': all_ages})
    
    if age_labels is not None:
        freq_df = df.groupby(['AgeGroup', 'Haplotype']).size().reset_index(name='Count')
        total_by_age = df.groupby('AgeGroup').size().reset_index(name='Total')
        freq_df = freq_df.merge(total_by_age, on='AgeGroup')
        freq_df['PredictedFrequency'] = freq_df['Count'] / freq_df['Total']
        freq_df = freq_df[['AgeGroup', 'Haplotype', 'PredictedFrequency']]
    else:
        freq_df = df.groupby('Haplotype').size().reset_index(name='Count')
        freq_df['PredictedFrequency'] = freq_df['Count'] / freq_df['Count'].sum()
        freq_df = freq_df[['Haplotype', 'PredictedFrequency']]
    
    return freq_df


def chi_squared_test(true_freq: pd.DataFrame, pred_freq: pd.DataFrame, age_group: str = None) -> Dict:
    """Perform chi-squared goodness-of-fit test."""
    if age_group:
        true_subset = true_freq[true_freq['AgeGroup'] == age_group].copy()
        pred_subset = pred_freq[pred_freq['AgeGroup'] == age_group].copy()
    else:
        true_subset = true_freq.copy()
        pred_subset = pred_freq.copy()
    
    merged = true_subset.merge(pred_subset, on='Haplotype', how='outer', suffixes=('_true', '_pred'))
    merged = merged.fillna(1e-10)
    
    observed = merged['PredictedFrequency'].values
    expected = merged['TrueFrequency'].values
    
    # Normalize to ensure sums match exactly (avoid float precision errors)
    observed = observed / observed.sum() * 10000
    expected = expected / expected.sum() * 10000
    
    chi2, pval = stats.chisquare(observed, expected)
    mae = np.mean(np.abs(merged['PredictedFrequency'] - merged['TrueFrequency']))
    
    return {
        'chi_squared': chi2,
        'df': len(merged) - 1,
        'p_value': pval,
        'n_haplotypes': len(merged),
        'mean_absolute_error': mae
    }


def run_frequency_baseline(genotypes: List, age_labels: List[str]) -> pd.DataFrame:
    predictions = []
    for sample_genotypes in genotypes:
        h1 = [pair[0] for pair in sample_genotypes]
        h2 = [pair[1] for pair in sample_genotypes]
        predictions.append((h1, h2))
    return calculate_frequency_from_predictions(predictions, age_labels)


def run_em_frequency(genotypes: List, age_labels: List[str]) -> pd.DataFrame:
    em = EMHaplotypePhaser(tolerance=1e-8, max_iterations=100)
    frequencies = em.fit(genotypes)
    predictions = em.predict(genotypes)
    return calculate_frequency_from_predictions(predictions, age_labels)


def run_beagle_frequency(df_unphased: pd.DataFrame, locus_columns: List[str], age_labels: List[str], output_dir: str) -> pd.DataFrame:
    """Run Beagle and extract frequencies."""
    beagle_output = os.path.join(output_dir, 'beagle_temp')
    beagle_runner = BeagleRunner(output_dir=beagle_output)
    
    if not (beagle_runner.java_available and beagle_runner.jar_available):
        logging.warning("Beagle not available, skipping...")
        return None
    
    df_beagle_pred = beagle_runner.run(df_unphased, locus_columns)
    
    if df_beagle_pred is None:
        return None
    
    predictions = []
    for _, row in df_beagle_pred.iterrows():
        h1 = row['Haplotype1'].split('_')
        h2 = row['Haplotype2'].split('_')
        predictions.append((h1, h2))
    
    return calculate_frequency_from_predictions(predictions, age_labels)


def run_transphaser_frequency(
    config: TransPhaserConfig,
    df_unphased: pd.DataFrame,
    df_phased: pd.DataFrame,
    age_labels: List[str]
) -> pd.DataFrame:
    runner = TransPhaserRunner(config)
    results = runner.run(df_unphased, df_phased)
    
    predictions_df = runner.get_most_likely_haplotypes(df_unphased)
    
    predictions = []
    for _, row in predictions_df.iterrows():
        h1 = row['Haplotype1'].split('_')
        h2 = row['Haplotype2'].split('_')
        predictions.append((h1, h2))
    
    return calculate_frequency_from_predictions(predictions, age_labels)


def plot_frequency_comparison(
    true_freq: pd.DataFrame,
    predictions: Dict[str, pd.DataFrame],
    top_n: int,
    output_path: str,
    title: str
):
    """Create bar plot comparing true vs predicted frequencies."""
    if 'AgeGroup' in true_freq.columns:
        top_haplotypes = true_freq.groupby('Haplotype')['TrueFrequency'].mean().nlargest(top_n).index.tolist()
        age_groups = sorted(true_freq['AgeGroup'].unique())
        
        fig, axes = plt.subplots(1, len(age_groups), figsize=(5*len(age_groups), 6), sharey=True)
        if len(age_groups) == 1:
            axes = [axes]
        
        for ax_idx, age in enumerate(age_groups):
            ax = axes[ax_idx]
            
            true_age = true_freq[true_freq['AgeGroup'] == age]
            true_age = true_age[true_age['Haplotype'].isin(top_haplotypes)].set_index('Haplotype')
            
            plot_data = {'True': true_age['TrueFrequency']}
            
            for method_name, pred_df in predictions.items():
                if pred_df is None:
                    continue
                pred_age = pred_df[pred_df['AgeGroup'] == age]
                pred_age = pred_age[pred_age['Haplotype'].isin(top_haplotypes)].set_index('Haplotype')
                plot_data[method_name] = pred_age['PredictedFrequency']
            
            plot_df = pd.DataFrame(plot_data).fillna(0.0)
            plot_df = plot_df.loc[top_haplotypes]
            plot_df.index = [f"H{i+1}" for i in range(len(plot_df))]
            
            plot_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.8)
            ax.set_title(f'Age: {age}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Haplotype', fontsize=11)
            if ax_idx == 0:
                ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Marginal frequencies (no age stratification)
        top_haplotypes = true_freq.nlargest(top_n, 'TrueFrequency')['Haplotype'].tolist()
        
        plot_data = {'True': true_freq.set_index('Haplotype')['TrueFrequency']}
        for method_name, pred_df in predictions.items():
            if pred_df is None:
                continue
            plot_data[method_name] = pred_df.set_index('Haplotype')['PredictedFrequency']
        
        plot_df = pd.DataFrame(plot_data).fillna(0.0)
        plot_df = plot_df.loc[top_haplotypes]
        plot_df.index = [f"H{i+1}" for i in range(len(plot_df))]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        plot_df.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Haplotype', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def run_dataset1_no_age_effect(output_dir: str):
    """Dataset 1: NO age effect - test all methods fairly."""
    logging.info("="*80)
    logging.info("DATASET 1: NO AGE EFFECT - All Methods")
    logging.info("="*80)
    
    # Generate age-INDEPENDENT data (10,000 total for consistency)
    generator = RealisticAgeDependentHLAGenerator(seed=42, age_effect_strength=0.0)  # NO age effect!
    unphased_df, phased_df, true_freq = generator.generate_dataset(n_per_age_group=2500)
    
    # Save
    os.makedirs('examples/data_frequency_no_age', exist_ok=True)
    unphased_df.to_csv('examples/data_frequency_no_age/unphased.csv', index=False)
    phased_df.to_csv('examples/data_frequency_no_age/phased.csv', index=False)
    true_freq.to_csv('examples/data_frequency_no_age/true_frequencies.csv', index=False)
    
    locus_columns = ['HLA-A', 'HLA-C', 'HLA-B', 'HLA-DRB1', 'HLA-DQB1', 'HLA-DPB1']
    
    # Extract genotypes
    genotypes = []
    age_labels = []
    for _, row in unphased_df.iterrows():
        sample_genotypes = []
        for locus in locus_columns:
            alleles = row[locus].split('/')
            sample_genotypes.append((alleles[0].strip(), alleles[1].strip()))
        genotypes.append(sample_genotypes)
        age_labels.append(row['AgeGroup'])
    
    predictions = {}
    
    # Frequency Baseline
    logging.info("Running Frequency Baseline...")
    predictions['Frequency'] = run_frequency_baseline(genotypes, age_labels)
    
    # EM
    logging.info("Running EM...")
    predictions['EM'] = run_em_frequency(genotypes, age_labels)
    
    # Beagle
    logging.info("Running Beagle...")
    beagle_freq = run_beagle_frequency(unphased_df, locus_columns, age_labels, output_dir)
    if beagle_freq is not None:
        predictions['Beagle'] = beagle_freq
    
    # TransPhaser (no age - fair since data has no age effect anyway)
    logging.info("Running TransPhaser...")
    config = TransPhaserConfig(
        model_name="TransPhaser-Dataset1",
        seed=42,
        device='cpu',
        output_dir=os.path.join(output_dir, 'dataset1'),
        data={
            "unphased_data_path": 'examples/data_frequency_no_age/unphased.csv',
            "phased_data_path": 'examples/data_frequency_no_age/phased.csv',
            "locus_columns": locus_columns,
            "covariate_columns": [],
            "categorical_covariate_columns": [],
            "validation_split_ratio": 0.2
        },
        model={"embedding_dim": 128, "latent_dim": 16,
               "encoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256},
               "decoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256}},
        training={"batch_size": 32, "learning_rate": 1.23e-05, "epochs": 50,
                  "kl_annealing_epochs": 8, "early_stopping_patience": 15}
    )
    predictions['TransPhaser'] = run_transphaser_frequency(config, unphased_df, phased_df, age_labels)
    
    # Calculate statistics
    chi_sq_results = {}
    for age in true_freq['AgeGroup'].unique():
        chi_sq_results[age] = {}
        for method_name, pred_freq in predictions.items():
            if pred_freq is not None:
                chi_sq_results[age][method_name] = chi_squared_test(true_freq, pred_freq, age)
    
    # Save
    with open(os.path.join(output_dir, 'dataset1_results.json'), 'w') as f:
        json.dump(chi_sq_results, f, indent=2)
    
    # Plot
    plot_frequency_comparison(
        true_freq, predictions, top_n=8,
        output_path=os.path.join(output_dir, 'dataset1_comparison.png'),
        title='Dataset 1: All Methods (No Age Effect in Data)'
    )
    
    logging.info("Dataset 1 complete!")
    return chi_sq_results


def run_dataset2_with_age_effect(output_dir: str):
    """Dataset 2: WITH age effect - test only TransPhaser ±age."""
    logging.info("="*80)
    logging.info("DATASET 2: WITH AGE EFFECT - TransPhaser only")
    logging.info("="*80)
    
    # Use existing age-dependent data
    df_unphased = pd.read_csv('examples/data_age_dependent_hard/unphased.csv')
    df_phased = pd.read_csv('examples/data_age_dependent_hard/phased.csv')
    true_freq = pd.read_csv('examples/data_age_dependent_hard/true_frequencies.csv')
    
    locus_columns = ['HLA-A', 'HLA-C', 'HLA-B', 'HLA-DRB1', 'HLA-DQB1', 'HLA-DPB1']
    age_labels = df_unphased['AgeGroup'].tolist()
    
    predictions = {}
    
    # TransPhaser WITHOUT age
    logging.info("Running TransPhaser WITHOUT age covariate...")
    config_no_age = TransPhaserConfig(
        model_name="TransPhaser-NoAge",
        seed=42,
        device='cpu',
        output_dir=os.path.join(output_dir, 'dataset2_no_age'),
        data={
            "unphased_data_path": 'examples/data_age_dependent_hard/unphased.csv',
            "phased_data_path": 'examples/data_age_dependent_hard/phased.csv',
            "locus_columns": locus_columns,
            "covariate_columns": [],
            "categorical_covariate_columns": [],
            "validation_split_ratio": 0.2
        },
        model={"embedding_dim": 128, "latent_dim": 16,
               "encoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256},
               "decoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256}},
        training={"batch_size": 32, "learning_rate": 1.23e-05, "epochs": 50,
                  "kl_annealing_epochs": 8, "early_stopping_patience": 15}
    )
    predictions['TransPhaser (No Age)'] = run_transphaser_frequency(config_no_age, df_unphased, df_phased, age_labels)
    
    # TransPhaser WITH age
    logging.info("Running TransPhaser WITH age covariate...")
    config_with_age = TransPhaserConfig(
        model_name="TransPhaser-WithAge",
        seed=42,
        device='cpu',
        output_dir=os.path.join(output_dir, 'dataset2_with_age'),
        data={
            "unphased_data_path": 'examples/data_age_dependent_hard/unphased.csv',
            "phased_data_path": 'examples/data_age_dependent_hard/phased.csv',
            "locus_columns": locus_columns,
            "covariate_columns": ['AgeGroup'],
            "categorical_covariate_columns": ['AgeGroup'],
            "validation_split_ratio": 0.2
        },
        model={"embedding_dim": 128, "latent_dim": 16,
               "encoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256},
               "decoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256}},
        training={"batch_size": 32, "learning_rate": 1.23e-05, "epochs": 50,
                  "kl_annealing_epochs": 8, "early_stopping_patience": 15}
    )
    predictions['TransPhaser (With Age)'] = run_transphaser_frequency(config_with_age, df_unphased, df_phased, age_labels)
    
    # Calculate statistics
    chi_sq_results = {}
    for age in true_freq['AgeGroup'].unique():
        chi_sq_results[age] = {}
        for method_name, pred_freq in predictions.items():
            chi_sq_results[age][method_name] = chi_squared_test(true_freq, pred_freq, age)
    
    # Save
    with open(os.path.join(output_dir, 'dataset2_results.json'), 'w') as f:
        json.dump(chi_sq_results, f, indent=2)
    
    # Plot
    plot_frequency_comparison(
        true_freq, predictions, top_n=8,
        output_path=os.path.join(output_dir, 'dataset2_comparison.png'),
        title='Dataset 2: TransPhaser ±Age (Age-Dependent Data)'
    )
    
    logging.info("Dataset 2 complete!")
    return chi_sq_results


def main():
    output_dir = 'examples/output/frequency_experiment_corrected'
    setup_logging(output_dir)
    
    logging.info("Starting CORRECTED Frequency Prediction Experiment")
    logging.info("Dataset 1: No age effect → all methods")
    logging.info("Dataset 2: With age effect → TransPhaser ±age only")
    
    # Run both datasets
    chi1 = run_dataset1_no_age_effect(output_dir)
    chi2 = run_dataset2_with_age_effect(output_dir)
    
    logging.info("\n" + "="*80)
    logging.info("EXPERIMENT COMPLETE!")
    logging.info("="*80)
    logging.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
