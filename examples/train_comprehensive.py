
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

def ensure_data_exists(data_dir: str, seed: int = 42):
    """Checks if data exists, generates it if not."""
    unphased_path = os.path.join(data_dir, "realistic_genotypes_unphased.csv")
    phased_path = os.path.join(data_dir, "realistic_haplotypes_phased.csv")
    
    if not os.path.exists(unphased_path) or not os.path.exists(phased_path):
        logging.info("Data files not found. Generating realistic HLA dataset...")
        os.makedirs(data_dir, exist_ok=True)
        
        generator = RealisticHLAGenerator(seed=seed)
        unphased_df, phased_df = generator.generate_dataset(
            n_samples=10000, 
            populations=['EUR', 'AFR', 'ASN', 'HIS'],
            output_dir=data_dir
        )
        
        unphased_df.to_csv(unphased_path, index=False)
        phased_df.to_csv(phased_path, index=False)
        logging.info(f"Generated {len(unphased_df)} samples in {data_dir}")
    else:
        logging.info(f"Using existing data in {data_dir}")



def create_missing_data_version(data_dir: str, input_path: str, missing_prob: float, seed: int) -> str:
    """
    Creates a copy of the input CSV with artificial missing values.
    Returns the path to the new CSV.
    """
    output_filename = f"realistic_genotypes_unphased_missing_{int(missing_prob*100)}pct.csv"
    output_path = os.path.join(data_dir, output_filename)
    
    logging.info(f"Creating dataset with {missing_prob:.1%} missing data at {output_path}...")
    
    df = pd.read_csv(input_path)
    rng = np.random.RandomState(seed)
    
    # Identify HLA columns
    hla_cols = [c for c in df.columns if c.startswith("HLA")]
    
    total_alleles = 0
    missing_alleles = 0
    
    for _, row in df.iterrows():
        for col in hla_cols:
            val = str(row[col])
            
            # Helper to process allele string
            def mask_val(v):
                if rng.rand() < missing_prob:
                    return "MISSING"
                return v

            if '/' in val:
                parts = val.split('/')
                new_parts = [mask_val(p) for p in parts]
                row[col] = '/'.join(new_parts)
                total_alleles += len(parts)
                missing_alleles += sum(1 for p in new_parts if p == "MISSING")
            elif ',' in val:
                parts = val.split(',')
                new_parts = [mask_val(p) for p in parts]
                row[col] = ','.join(new_parts)
                total_alleles += len(parts)
                missing_alleles += sum(1 for p in new_parts if p == "MISSING")
            else:
                # Homozygous or single
                if rng.rand() < missing_prob:
                    row[col] = "MISSING"
                    missing_alleles += 2 # Treating as both missing
                else:
                    # Keep as is, counts as 2 valid
                    pass
                total_alleles += 2

    df.to_csv(output_path, index=False)
    logging.info(f"Generated missing data: {missing_alleles}/{total_alleles} alleles masked ({missing_alleles/total_alleles:.2%})")
    
    return output_path


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
        """
        Randomly phase genotypes.
        
        Args:
            genotypes: List of samples, each sample is a list of (allele1, allele2) tuples per locus
            
        Returns:
            List of (haplotype1, haplotype2) tuples
        """
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
        """Evaluate random guessing accuracy and other phasing metrics."""
        predictions = self.predict(genotypes)
        
        # Convert to format expected by HLAPhasingMetrics
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
    """
    Baseline that predicts the most common VALID phasing for a given genotype.
    It builds a conditional probability map: P(HaplotypePair | Genotype).
    """
    
    def __init__(self, training_haplotypes: List[Tuple[List[str], List[str]]]):
        """
        Initialize from training data.
        
        Args:
            training_haplotypes: List of (haplotype1, haplotype2) tuples from training set
        """
        # Map: Genotype (sorted tuple of alleles) -> {HaplotypePair: count}
        self.genotype_to_haplotypes = {}
        
        for hap1, hap2 in training_haplotypes:
            # Reconstruct the genotype for this haplotype pair
            # Genotype is the multiset of alleles at each locus
            genotype_key = []
            for i in range(len(hap1)):
                # Locus i alleles
                alleles = sorted([hap1[i], hap2[i]])
                genotype_key.append(tuple(alleles))
            
            genotype_key = tuple(genotype_key) # efficient hashable key
            
            if genotype_key not in self.genotype_to_haplotypes:
                self.genotype_to_haplotypes[genotype_key] = {}
            
            # Store haplotype pair (sorted to handle phase ambiguity in counting)
            # We store the pair OF haplotypes strings
            h1_str = '_'.join(hap1)
            h2_str = '_'.join(hap2)
            pair_key = tuple(sorted([h1_str, h2_str]))
            
            self.genotype_to_haplotypes[genotype_key][pair_key] = \
                self.genotype_to_haplotypes[genotype_key].get(pair_key, 0) + 1
                
        # Pre-compute the best guess for each genotype
        self.best_guess_cache = {}
        for gt, counts in self.genotype_to_haplotypes.items():
            # Get phase pair with max count
            best_pair = max(counts.items(), key=lambda x: x[1])[0]
            # best_pair is (h1_str, h2_str)
            self.best_guess_cache[gt] = (best_pair[0].split('_'), best_pair[1].split('_'))
            
    def predict(self, genotypes: List) -> List[Tuple[List[str], List[str]]]:
        """
        Predict the most frequent phasing observed for this specific genotype.
        If genotype was never seen, fall back to random valid phasing.
        """
        predictions = []
        for sample_genotypes in genotypes:
            # specific genotype format for lookup: tuple of (allele1, allele2) tuples
            # sample_genotypes is already [(a1, a2), (b1, b2), ...]
            # but we need to ensure sorted order to match key
            lookup_key = tuple(tuple(sorted(locus_pair)) for locus_pair in sample_genotypes)
            
            if lookup_key in self.best_guess_cache:
                predictions.append(self.best_guess_cache[lookup_key])
            else:
                # Fallback: Just return the alleles split into two arbitrary haplotypes
                # (equivalent to random phasing, but valid)
                h1 = [pair[0] for pair in sample_genotypes]
                h2 = [pair[1] for pair in sample_genotypes]
                predictions.append((h1, h2))
                
        return predictions
    
    def evaluate(self, genotypes: List, ground_truth: List, tokenizer: AlleleTokenizer = None) -> Dict[str, float]:
        """Evaluate frequency-based accuracy and other phasing metrics."""
        predictions = self.predict(genotypes)
        
        # Convert to format expected by HLAPhasingMetrics
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
    """Run baseline methods and return results."""
    logging.info("=" * 80)
    logging.info("RUNNING BASELINE COMPARISONS")
    logging.info("=" * 80)
    
    # Load phased data (ground truth haplotypes)
    df_phased = pd.read_csv(config.data.phased_data_path)
    
    # Load unphased data (genotypes)
    df_unphased = pd.read_csv(config.data.unphased_data_path)
    
    # Extract haplotypes from phased data
    haplotypes = []
    for _, row in df_phased.iterrows():
        hap1 = row['Haplotype1'].split('_')
        hap2 = row['Haplotype2'].split('_')
        haplotypes.append((hap1, hap2))
    
    # Extract genotypes from unphased data
    # Parse format like "A*01/A*02" to get both alleles per locus
    genotypes = []
    for _, row in df_unphased.iterrows():
        sample_genotypes = []
        for locus_col in config.data.locus_columns:
            genotype_str = str(row[locus_col])
            # Handle both / and , separators
            if '/' in genotype_str:
                alleles = genotype_str.split('/')
            elif ',' in genotype_str:
                alleles = genotype_str.split(',')
            else:
                # Homozygous or single allele
                alleles = [genotype_str, genotype_str]
            
            if len(alleles) >= 2:
                sample_genotypes.append((alleles[0].strip(), alleles[1].strip()))
            else:
                # Fallback for unexpected format
                sample_genotypes.append((alleles[0].strip(), alleles[0].strip()))
        genotypes.append(sample_genotypes)
    
    # Split train/val (use same ratio as config)
    split_idx = int(len(haplotypes) * (1 - config.data.validation_split_ratio))
    train_haps = haplotypes[:split_idx]
    val_haps = haplotypes[split_idx:]
    train_genotypes = genotypes[:split_idx]
    val_genotypes = genotypes[split_idx:]
    
    results = {}
    
    # Build a tokenizer to satisfy HLAPhasingMetrics requirements
    tokenizer = AlleleTokenizer()
    df_train_unphased = df_unphased.iloc[:split_idx]
    tokenizer.build_vocabulary_from_dataframe(df_train_unphased, config.data.locus_columns)

    # Run random guessing baseline
    logging.info("Running Random Guessing Baseline...")
    random_baseline = RandomGuessingBaseline(num_loci=len(config.data.locus_columns))
    random_results = random_baseline.evaluate(val_genotypes, val_haps, tokenizer=tokenizer)
    logging.info(f"Random Baseline - Accuracy: {random_results['phasing_accuracy']:.2%}")
    results['random_baseline'] = random_results
    
    # Run frequency-based baseline
    logging.info("Running Frequency-Based Baseline...")
    freq_baseline = FrequencyBasedBaseline(train_haps)
    freq_results = freq_baseline.evaluate(val_genotypes, val_haps, tokenizer=tokenizer)
    logging.info(f"Frequency Baseline - Accuracy: {freq_results['phasing_accuracy']:.2%}")
    results['frequency_baseline'] = freq_results
    
    # Run EM baseline
    logging.info("Running EM (Expectation-Maximization) Baseline...")
    # EM learns from UNPHASED data (genotypes), so we fit on train_genotypes
    em_baseline = EMHaplotypePhaser(tolerance=1e-8, max_iterations=100)
    em_freqs = em_baseline.fit(train_genotypes)
    
    # Log top haplotypes
    logging.info("Top 5 Haplotypes found by EM:")
    for h, f in list(em_freqs.items())[:5]:
        logging.info(f"  {h}: {f:.4f}")
        
    em_results = em_baseline.evaluate(val_genotypes, val_haps, tokenizer=tokenizer)
    
    # Add history to results
    em_results['log_likelihood_history'] = em_baseline.history_
    em_results['top_haplotypes'] = {str(h): f for h, f in list(em_freqs.items())[:20]}
    
    # Plot likelihood history

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(em_baseline.history_) + 1), em_baseline.history_, marker='o')
    plt.title('EM Algorithm Convergence: Log-Likelihood History')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.grid(True)
    plot_path = os.path.join(output_dir, 'em_likelihood_history.png')
    plt.savefig(plot_path)
    plt.close()
    
    logging.info(f"EM Baseline - Accuracy: {em_results['phasing_accuracy']:.2%}")
    logging.info(f"EM likelihood plot saved to {plot_path}")
    results['em_baseline'] = em_results
    
    # Run Beagle Baseline
    # Import here to avoid top-level issues if dependency missing
    try:
        from transphaser.beagle_runner import BeagleRunner
        
        beagle_jar_default = "beagle.jar"
        output_sub = os.path.join(output_dir, "beagle_files")
        beagle_runner = BeagleRunner(output_dir=output_sub)
        
        if beagle_runner.java_available and beagle_runner.jar_available:
            logging.info("Running Beagle Baseline...")
            
            # Reconstruct validation dataframe for Beagle
            # val_genotypes is List[List[(a1, a2)]]
            # columns correspond to config.data.locus_columns
            val_data = []
            for sample_gt in val_genotypes:
                row = {}
                for idx, (a1, a2) in enumerate(sample_gt):
                    col_name = config.data.locus_columns[idx]
                    row[col_name] = f"{a1}/{a2}"
                val_data.append(row)
            
            df_val_beagle_input = pd.DataFrame(val_data)
            
            # Run Beagle
            df_beagle_pred = beagle_runner.run(df_val_beagle_input, config.data.locus_columns)
            
            if df_beagle_pred is not None:
                # Convert predictions to pairs for metrics
                pred_pairs = []
                for _, row in df_beagle_pred.iterrows():
                    h1_str = row['Haplotype1']
                    h2_str = row['Haplotype2']
                    # Sort to handle Phase Ambiguity
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
            
    except ImportError:
        logging.warning("Could not import BeagleRunner. Skipping Beagle Baseline.")
    except Exception as e:
        logging.error(f"Error running Beagle Baseline: {e}")

    # Save results
    results_file = os.path.join(output_dir, 'baseline_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Baseline results saved to {results_file}")
    return results


def train_transphaser(config: TransPhaserConfig, output_dir: str) -> Dict:
    """Train TransPhaser model and return results."""
    logging.info("=" * 80)
    logging.info("TRAINING TransPhaser")
    logging.info("=" * 80)
    
    start_time = time.time()
    
    # Load data
    df_unphased = pd.read_csv(config.data.unphased_data_path)
    df_phased = pd.read_csv(config.data.phased_data_path)
    
    # Initialize runner
    runner = TransPhaserRunner(config)
    
    # Train
    logging.info(f"Training TransPhaser for {config.training.epochs} epochs...")
    results = runner.run(df_unphased, df_phased)
    
    # Save model
    model_path = os.path.join(output_dir, 'transphaser_model.pt')
    runner.save(model_path)
    
    # Generate Plots and Extra Info
    try:
        # Plot Loss
        history = runner.extract_loss_history()
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('TransPhaser Training Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'))
        plt.close()
        
        # Calculate most likely haplotypes for a subset
        subset = df_unphased.iloc[:100]
        likely_haps = runner.get_most_likely_haplotypes(subset)
        likely_haps.to_csv(os.path.join(output_dir, 'sample_predictions.csv'), index=False)
        
        # Embeddings plot
        runner.plot_embeddings('HLA-A', output_path=os.path.join(output_dir, 'hla_a_embeddings.png'))
        
    except Exception as e:
        logging.warning(f"Could not generate extra plots: {e}")

    training_time = time.time() - start_time
    logging.info(f"TransPhaser training completed in {training_time:.2f} seconds")
    
    results['training_time_seconds'] = training_time
    results['epochs'] = config.training.epochs
    
    return results


def generate_comparison_report(baseline_results: Dict, output_dir: str, transphaser_results: Dict = None) -> None:
    """Generate comprehensive comparison report."""
    logging.info("=" * 80)
    logging.info("FINAL COMPARISON REPORT")
    logging.info("=" * 80)
    
    methods = ['random_baseline', 'frequency_baseline', 'em_baseline', 'beagle', 'transphaser']
    metrics = ['phasing_accuracy', 'avg_hamming_distance', 'avg_switch_errors']
    
    report_data = {}
    
    # helper to get metric
    def get_val(res, method, metric):
        if method == 'transphaser':
            if transphaser_results is None:
                return 0.0
            if 'evaluation_summary' in transphaser_results:
                return transphaser_results['evaluation_summary'].get(metric, 0.0)
            return transphaser_results.get(metric, 0.0)
        return res.get(method, {}).get(metric, 0.0)

    for method in methods:
        if method == 'transphaser':
            report_data[method] = {metric: get_val(transphaser_results, method, metric) for metric in metrics}
        else:
            report_data[method] = {metric: get_val(baseline_results, method, metric) for metric in metrics}

    # Save JSON report
    report = {
        'comparison_summary': report_data,
        'baseline_results': baseline_results,
        'transphaser_results': transphaser_results,
    }
    
    report_path = os.path.join(output_dir, 'comprehensive_evaluation.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary table
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
    
    transphaser_acc = report_data['transphaser']['phasing_accuracy']
    best_model_acc = transphaser_acc
    best_baseline_acc = max(report_data[m]['phasing_accuracy'] for m in ['random_baseline', 'frequency_baseline', 'em_baseline', 'beagle'])
    
    if best_model_acc < best_baseline_acc:
        logging.warning("⚠️  WARNING: TransPhaser did not beat the best baseline!")
    else:
        logging.info("✅ TransPhaser beats the best baseline!")
    
    logging.info(f"\nFull report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive TransPhaser training with baseline comparison')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument('--data-dir', type=str, default='examples/data', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='examples/output/comprehensive_training', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1.23e-05, help='Learning rate')
    parser.add_argument('--missing-prob', type=float, default=0.0, help='Probability of missing alleles in unphased data')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    
    logging.info("=" * 80)
    logging.info("TRANSPHASER COMPREHENSIVE TRAINING")
    logging.info("=" * 80)
    logging.info(f"Configuration:")
    logging.info(f"  Epochs: {args.epochs}")
    logging.info(f"  Device: {args.device}")
    logging.info(f"  Batch Size: {args.batch_size}")
    logging.info(f"  Learning Rate: {args.lr}")
    logging.info(f"  Seed: {args.seed}")
    logging.info(f"  Missing Prob: {args.missing_prob}")
    logging.info("=" * 80)
    
    # Create configuration
    ensure_data_exists(args.data_dir, args.seed)
    
    unphased_path = os.path.join(args.data_dir, "realistic_genotypes_unphased.csv")
    
    # Handle missing data generation if requested
    if args.missing_prob > 0:
        unphased_path = create_missing_data_version(args.data_dir, unphased_path, args.missing_prob, args.seed)
    
    config = TransPhaserConfig(
        model_name=f"TransPhaser-{args.epochs}epochs",
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        data={
            "unphased_data_path": unphased_path,
            "phased_data_path": os.path.join(args.data_dir, "realistic_haplotypes_phased.csv"),
            "locus_columns": ["HLA-A", "HLA-C", "HLA-B", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"],
            "covariate_columns": ["Population", "AgeGroup"],
            "categorical_covariate_columns": ["Population", "AgeGroup"],
            "validation_split_ratio": 0.2
        },
        model={
            "embedding_dim": 128,  # Tuned (Trial 7: 68.2% accuracy)
            "latent_dim": 16,      # Baseline value
            "encoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256},  # Tuned (Trial 7)
            "decoder": {"num_layers": 4, "num_heads": 8, "dropout": 0.3, "ff_dim": 256},  # Tuned (Trial 7)
        },
        training={
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "kl_annealing_type": "linear",
            "kl_annealing_epochs": max(1, int(args.epochs * 0.16)),  # Tuned (Trial 7): ~16% of training
            "early_stopping_patience": 20,
            "reconstruction_weight": 1.0,  # VAE reconstruction loss
            "consistency_weight": 1.0,     # 🔥 CRITICAL: Enforce H1 ⊕ H2 = Genotype
            "entropy_weight": 0.01,        # Encourage prediction diversity
        },
        reporting={
            "formats": ["json", "txt"],
            "base_filename": "final_report",
            "plot_filename": "training_loss_curves.png",
        }
    )
    
    # Save config
    config.save(os.path.join(args.output_dir, 'config.json'))
    
    # Run baselines
    try:
        baseline_results = run_baselines(config, args.output_dir)
    except Exception as e:
        logging.error(f"Baseline evaluation failed: {e}")
        logging.warning("Continuing with model training...")
        baseline_results = {}
    
    # Train TransPhaser FIRST for quick feedback
    try:
        transphaser_results = train_transphaser(config, args.output_dir)
    except Exception as e:
        logging.error(f"TransPhaser training failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        transphaser_results = {}
    
    
    # Generate comparison report
    generate_comparison_report(baseline_results, args.output_dir, transphaser_results)
    
    
    logging.info("\n" + "=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)


if __name__ == '__main__':
    main()
