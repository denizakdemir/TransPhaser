
import argparse
import logging
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Add parent dir to path to import transphaser
sys.path.insert(0, str(Path(__file__).parent.parent))

from transphaser.config import TransPhaserConfig
from transphaser.em import EMHaplotypePhaser
from transphaser.evaluation import HLAPhasingMetrics
from transphaser.runner import TransPhaserRunner
from transphaser.data_preprocessing import AlleleTokenizer

# Import local data generator
try:
    from generate_realistic_data import RealisticHLAGenerator
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from generate_realistic_data import RealisticHLAGenerator

def ensure_data_exists_one_pop(data_dir: str, seed: int = 42):
    """Checks if data exists, generates it if not. Generates ONLY 'EUR' population."""
    unphased_path = os.path.join(data_dir, "realistic_genotypes_unphased.csv")
    phased_path = os.path.join(data_dir, "realistic_haplotypes_phased.csv")
    
    if not os.path.exists(unphased_path) or not os.path.exists(phased_path):
        logging.info("Data files not found. Generating realistic HLA dataset (EUR ONLY)...")
        os.makedirs(data_dir, exist_ok=True)
        
        generator = RealisticHLAGenerator(seed=seed)
        # ONLY GENERATING EUR
        unphased_df, phased_df = generator.generate_dataset(
            n_samples=10000, # Consistent with other experiments
            populations=['EUR'],
            output_dir=data_dir
        )
        
        unphased_df.to_csv(unphased_path, index=False)
        phased_df.to_csv(phased_path, index=False)
        logging.info(f"Generated {len(unphased_df)} samples in {data_dir}")
    else:
        logging.info(f"Using existing data in {data_dir}")


def setup_logging(output_dir: str) -> None:
    """Configure logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class RandomGuessingBaseline:
    """Baseline that randomly assigns alleles to haplotypes."""
    
    def __init__(self, num_loci: int, seed: int = 42):
        self.num_loci = num_loci
        self.rng = np.random.RandomState(seed)
    
    def predict(self, genotypes: List[List[Tuple[str, str]]]) -> List[Tuple[List[str], List[str]]]:
        predictions = []
        for sample_genotypes in genotypes:
            hap1, hap2 = [], []
            for allele1, allele2 in sample_genotypes:
                # Randomly assign alleles to haplotypes
                if self.rng.rand() > 0.5:
                    hap1.append(allele1)
                    hap2.append(allele2)
                else:
                    hap1.append(allele2)
                    hap2.append(allele1)
            predictions.append((hap1, hap2))
        return predictions
    
    def evaluate(self, genotypes: List, ground_truth: List, tokenizer: AlleleTokenizer = None) -> Dict[str, float]:
        predictions = self.predict(genotypes)
        
        pred_pairs = []
        for p1, p2 in predictions:
            h1_str = '_'.join(p1)
            h2_str = '_'.join(p2)
            pred_pairs.append(tuple(sorted((h1_str, h2_str))))
            
        truth_pairs = []
        for t1, t2 in ground_truth:
            h1_str = '_'.join(t1)
            h2_str = '_'.join(t2)
            truth_pairs.append(tuple(sorted((h1_str, h2_str))))
            
        if tokenizer is None:
            tokenizer = AlleleTokenizer()
            
        metrics_calc = HLAPhasingMetrics(tokenizer=tokenizer)
        return metrics_calc.calculate_metrics(pred_pairs, truth_pairs)


class FrequencyBasedBaseline:
    """Baseline that predicts the most common VALID phasing for a given genotype."""
    
    def __init__(self, training_haplotypes: List[Tuple[List[str], List[str]]]):
        self.genotype_to_haplotypes = {}
        
        for hap1, hap2 in training_haplotypes:
            genotype_key = []
            for i in range(len(hap1)):
                alleles = sorted([hap1[i], hap2[i]])
                genotype_key.append(tuple(alleles))
            
            genotype_key = tuple(genotype_key)
            
            if genotype_key not in self.genotype_to_haplotypes:
                self.genotype_to_haplotypes[genotype_key] = {}
            
            h1_str = '_'.join(hap1)
            h2_str = '_'.join(hap2)
            pair_key = tuple(sorted([h1_str, h2_str]))
            
            self.genotype_to_haplotypes[genotype_key][pair_key] = \
                self.genotype_to_haplotypes[genotype_key].get(pair_key, 0) + 1
                
        self.best_guess_cache = {}
        for gt, counts in self.genotype_to_haplotypes.items():
            best_pair = max(counts.items(), key=lambda x: x[1])[0]
            self.best_guess_cache[gt] = (best_pair[0].split('_'), best_pair[1].split('_'))
            
    def predict(self, genotypes: List) -> List[Tuple[List[str], List[str]]]:
        predictions = []
        for sample_genotypes in genotypes:
            lookup_key = tuple(tuple(sorted(locus_pair)) for locus_pair in sample_genotypes)
            
            if lookup_key in self.best_guess_cache:
                predictions.append(self.best_guess_cache[lookup_key])
            else:
                h1 = [pair[0] for pair in sample_genotypes]
                h2 = [pair[1] for pair in sample_genotypes]
                predictions.append((h1, h2))
                
        return predictions
    
    def evaluate(self, genotypes: List, ground_truth: List, tokenizer: AlleleTokenizer = None) -> Dict[str, float]:
        predictions = self.predict(genotypes)
        
        pred_pairs = []
        for p1, p2 in predictions:
            h1_str = '_'.join(p1)
            h2_str = '_'.join(p2)
            pred_pairs.append(tuple(sorted((h1_str, h2_str))))
            
        truth_pairs = []
        for t1, t2 in ground_truth:
            h1_str = '_'.join(t1)
            h2_str = '_'.join(t2)
            truth_pairs.append(tuple(sorted((h1_str, h2_str))))
            
        if tokenizer is None:
            tokenizer = AlleleTokenizer()
            
        metrics_calc = HLAPhasingMetrics(tokenizer=tokenizer)
        return metrics_calc.calculate_metrics(pred_pairs, truth_pairs)


def run_baselines(config: TransPhaserConfig, output_dir: str) -> Dict[str, Dict]:
    logging.info("=" * 80)
    logging.info("RUNNING BASELINE COMPARISONS")
    logging.info("=" * 80)
    
    df_phased = pd.read_csv(config.data.phased_data_path)
    df_unphased = pd.read_csv(config.data.unphased_data_path)
    
    haplotypes = []
    for _, row in df_phased.iterrows():
        hap1 = row['Haplotype1'].split('_')
        hap2 = row['Haplotype2'].split('_')
        haplotypes.append((hap1, hap2))
    
    genotypes = []
    for _, row in df_unphased.iterrows():
        sample_genotypes = []
        for locus_col in config.data.locus_columns:
            genotype_str = str(row[locus_col])
            if '/' in genotype_str:
                alleles = genotype_str.split('/')
            elif ',' in genotype_str:
                alleles = genotype_str.split(',')
            else:
                alleles = [genotype_str, genotype_str]
            
            if len(alleles) >= 2:
                sample_genotypes.append((alleles[0].strip(), alleles[1].strip()))
            else:
                sample_genotypes.append((alleles[0].strip(), alleles[0].strip()))
        genotypes.append(sample_genotypes)
    
    split_idx = int(len(haplotypes) * (1 - config.data.validation_split_ratio))
    train_haps = haplotypes[:split_idx]
    val_haps = haplotypes[split_idx:]
    train_genotypes = genotypes[:split_idx]
    val_genotypes = genotypes[split_idx:]
    
    results = {}
    
    tokenizer = AlleleTokenizer()
    df_train_unphased = df_unphased.iloc[:split_idx]
    tokenizer.build_vocabulary_from_dataframe(df_train_unphased, config.data.locus_columns)

    logging.info("Running Random Guessing Baseline...")
    random_baseline = RandomGuessingBaseline(num_loci=len(config.data.locus_columns))
    random_results = random_baseline.evaluate(val_genotypes, val_haps, tokenizer=tokenizer)
    logging.info(f"Random Baseline - Accuracy: {random_results['phasing_accuracy']:.2%}")
    results['random_baseline'] = random_results
    
    logging.info("Running Frequency-Based Baseline...")
    freq_baseline = FrequencyBasedBaseline(train_haps)
    freq_results = freq_baseline.evaluate(val_genotypes, val_haps, tokenizer=tokenizer)
    logging.info(f"Frequency Baseline - Accuracy: {freq_results['phasing_accuracy']:.2%}")
    results['frequency_baseline'] = freq_results
    
    logging.info("Running EM (Expectation-Maximization) Baseline...")
    em_baseline = EMHaplotypePhaser(tolerance=1e-8, max_iterations=100)
    em_freqs = em_baseline.fit(train_genotypes)
    
    em_results = em_baseline.evaluate(val_genotypes, val_haps, tokenizer=tokenizer)
    em_results['log_likelihood_history'] = em_baseline.history_
    
    logging.info(f"EM Baseline - Accuracy: {em_results['phasing_accuracy']:.2%}")
    results['em_baseline'] = em_results
    
    # Beagle omitted for simplicity/reliability in this quick experiment unless needed if you have java
    # But let's try to include it if possible, if not just catch error
    try:
        from transphaser.beagle_runner import BeagleRunner
        output_sub = os.path.join(output_dir, "beagle_files")
        beagle_runner = BeagleRunner(output_dir=output_sub)
        
        if beagle_runner.java_available and beagle_runner.jar_available:
            logging.info("Running Beagle Baseline...")
            val_data = []
            for sample_gt in val_genotypes:
                row = {}
                for idx, (a1, a2) in enumerate(sample_gt):
                    col_name = config.data.locus_columns[idx]
                    row[col_name] = f"{a1}/{a2}"
                val_data.append(row)
            
            df_val_beagle_input = pd.DataFrame(val_data)
            df_beagle_pred = beagle_runner.run(df_val_beagle_input, config.data.locus_columns)
            
            if df_beagle_pred is not None:
                pred_pairs = []
                for _, row in df_beagle_pred.iterrows():
                    h1_str = row['Haplotype1']
                    h2_str = row['Haplotype2']
                    pred_pairs.append(tuple(sorted((h1_str, h2_str))))
                
                truth_pairs = []
                for t1, t2 in val_haps:
                    h1_str = '_'.join(t1)
                    h2_str = '_'.join(t2)
                    truth_pairs.append(tuple(sorted((h1_str, h2_str))))
                    
                metrics_calc = HLAPhasingMetrics(tokenizer=tokenizer)
                beagle_results = metrics_calc.calculate_metrics(pred_pairs, truth_pairs)
                logging.info(f"Beagle Baseline - Accuracy: {beagle_results['phasing_accuracy']:.2%}")
                results['beagle'] = beagle_results
        else:
            logging.warning("Skipping Beagle Baseline (Java or Jar missing).")
            
    except Exception as e:
        logging.warning(f"Error running Beagle Baseline: {e}")

    results_file = os.path.join(output_dir, 'baseline_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def train_transphaser(config: TransPhaserConfig, output_dir: str) -> Dict:
    logging.info("=" * 80)
    logging.info("TRAINING TransPhaser (No Covariates)")
    logging.info("=" * 80)
    
    start_time = time.time()
    
    df_unphased = pd.read_csv(config.data.unphased_data_path)
    df_phased = pd.read_csv(config.data.phased_data_path)
    
    runner = TransPhaserRunner(config)
    
    logging.info(f"Training TransPhaser for {config.training.epochs} epochs...")
    results = runner.run(df_unphased, df_phased)
    
    model_path = os.path.join(output_dir, 'transphaser_model.pt')
    runner.save(model_path)
    
    training_time = time.time() - start_time
    logging.info(f"TransPhaser training completed in {training_time:.2f} seconds")
    
    results['training_time_seconds'] = training_time
    results['epochs'] = config.training.epochs
    
    return results


def generate_comparison_report(baseline_results: Dict, output_dir: str, transphaser_results: Dict = None) -> None:
    logging.info("=" * 80)
    logging.info("FINAL COMPARISON REPORT")
    logging.info("=" * 80)
    
    methods = ['random_baseline', 'frequency_baseline', 'em_baseline', 'beagle', 'transphaser']
    metrics = ['phasing_accuracy', 'avg_hamming_distance', 'avg_switch_errors']
    
    report_data = {}
    
    def get_val(res, method, metric):
        if method == 'transphaser':
            if transphaser_results is None:
                return 0.0
            if 'evaluation_summary' in transphaser_results:
                return transphaser_results['evaluation_summary'].get(metric, 0.0)
            return transphaser_results.get(metric, 0.0)
        return res.get(method, {}).get(metric, 0.0)

    for method in methods:
        report_data[method] = {metric: get_val(baseline_results, method, metric) for metric in metrics}

    report_path = os.path.join(output_dir, 'comprehensive_evaluation.json')
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    logging.info("\n" + "=" * 80)
    logging.info(f"{'Method':<25} {'Accuracy':<12} {'Hamming':<12} {'Switch'}")
    logging.info("-" * 80)
    for method in methods:
        name = method.replace('_', ' ').title()
        acc = report_data[method]['phasing_accuracy']
        ham = report_data[method]['avg_hamming_distance']
        sw = report_data[method]['avg_switch_errors']
        logging.info(f"{name:<25} {acc:<12.2%} {ham:<12.2f} {sw:.2f}")
    logging.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='TransPhaser training ONE POPULATION without covariates')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (reduced for speed)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--data-dir', type=str, default='examples/data_one_pop', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='examples/output/one_pop_training', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    setup_logging(args.output_dir)
    
    logging.info("=" * 80)
    logging.info("TRANSPHASER ONE POPULATION TRAINING")
    logging.info("=" * 80)
    
    ensure_data_exists_one_pop(args.data_dir, args.seed)
    
    unphased_path = os.path.join(args.data_dir, "realistic_genotypes_unphased.csv")
    
    config = TransPhaserConfig(
        model_name=f"TransPhaser-OnePop",
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        data={
            "unphased_data_path": unphased_path,
            "phased_data_path": os.path.join(args.data_dir, "realistic_haplotypes_phased.csv"),
            "locus_columns": ["HLA-A", "HLA-C", "HLA-B", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"],
            "covariate_columns": [], # NO COVARIATES
            "categorical_covariate_columns": [], # NO COVARIATES
            "validation_split_ratio": 0.2
        },
        model={
            "embedding_dim": 128, 
            "latent_dim": 16,     
            "encoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256}, 
            "decoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256}, 
        },
        training={
            "batch_size": 32,
            "learning_rate": 1.23e-05,
            "epochs": args.epochs,
            "kl_annealing_type": "linear",
            "kl_annealing_epochs": max(1, int(args.epochs * 0.16)),
            "early_stopping_patience": 20,
            "reconstruction_weight": 1.0,
            "consistency_weight": 1.0, 
            "entropy_weight": 0.01,
        },
        reporting={
            "formats": ["json"],
            "base_filename": "final_report",
        }
    )
    
    config.save(os.path.join(args.output_dir, 'config.json'))
    
    try:
        baseline_results = run_baselines(config, args.output_dir)
    except Exception as e:
        logging.error(f"Baseline evaluation failed: {e}")
        baseline_results = {}
    
    try:
        transphaser_results = train_transphaser(config, args.output_dir)
    except Exception as e:
        logging.error(f"TransPhaser training failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        transphaser_results = {}
    
    generate_comparison_report(baseline_results, args.output_dir, transphaser_results)
    
    logging.info("\n" + "=" * 80)
    logging.info("DONE")
    logging.info("=" * 80)

if __name__ == '__main__':
    main()
