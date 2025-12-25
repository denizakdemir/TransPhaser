# TransPhaser Examples

This directory contains example scripts for training, evaluating, and tuning the TransPhaser model.

## Main Scripts

### 1. `train_comprehensive.py`
**Primary training script** for TransPhaser with baseline comparisons.

Trains TransPhaser on the realistic 6-locus HLA dataset (10,000 samples, 4 populations) and compares against:
- EM (Expectation-Maximization) baseline
- Frequency-based baseline  
- Random guessing baseline

**Usage:**
```bash
python examples/train_comprehensive.py --epochs 100 --device cpu
```

**Key features:**
- 80/20 train/validation split
- Comprehensive metrics (phasing accuracy, Hamming distance, switch errors)
- Baseline comparisons
- Latent space visualization

### 2. `tune_hyperparameters.py`
**Hyperparameter optimization** using Bayesian optimization (Optuna).

Tunes hyperparameters for the exact same problem as `train_comprehensive.py`:
- Same 6-locus realistic HLA data (10K samples)
- Same 80/20 train/validation split
- Same evaluation metrics

**Usage:**
```bash
python examples/tune_hyperparameters.py --n-trials 30 --epochs 100
```

**Search space:**
- `embedding_dim`: [64, 128, 256]
- `latent_dim`: [16, 32, 64, 128]
- `num_layers`: [1, 2, 3, 4]
- `num_heads`: [2, 4, 8]
- `dropout`: [0.05 - 0.3]
- `learning_rate`: [1e-5 - 1e-3] (log scale)
- `batch_size`: [16, 32, 64]
- `kl_annealing_fraction`: [0.05 - 0.3]

### 3. `run_phaser_example.py`
**Simple demonstration** of basic TransPhaser functionality.

Quick example showing how to:
- Load HLA data
- Configure the model
- Train TransPhaser
- Make predictions
- Evaluate results

**Usage:**
```bash
python examples/run_phaser_example.py
```

### 4. `validate_unsupervised.py`
**Validation script** to verify. unsupervised training.

Ensures that:
- Ground truth phased data is NEVER used during training
- Only unphased genotypes are used for model training
- Phased data is only used for evaluation metrics

**Usage:**
```bash
python examples/validate_unsupervised.py
```

### 5. `generate_realistic_data.py`
**Data generation** for realistic HLA datasets.

Generates synthetic but biologically plausible HLA genotype/haplotype data with:
- Multiple populations (EUR, AFR, ASN, HIS)
- Realistic allele frequencies
- Linkage disequilibrium patterns
- Population structure

**Usage:**
```bash
python examples/generate_realistic_data.py
```

## Workflow

### Standard Training Workflow
1. Generate data (automatic if not present):
   ```bash
   python examples/generate_realistic_data.py
   ```

2. Train with baselines:
   ```bash
   python examples/train_comprehensive.py --epochs 100
   ```

3. Results are saved to `examples/output/comprehensive_training/`

### Hyperparameter Tuning Workflow
1. Run tuning (uses Bayesian optimization):
   ```bash
   python examples/tune_hyperparameters.py --n-trials 30 --epochs 100
   ```

2. Best hyperparameters are saved to `examples/output/tuning/tuning_summary.json`

3. Update `train_comprehensive.py` with best hyperparameters and retrain

## Data Directory

`examples/data/` contains:
- `realistic_genotypes_unphased.csv` - Unphased genotype data (10,000 samples)
- `realistic_haplotypes_phased.csv` - Ground truth phased haplotypes (for evaluation only)

## Archived Scripts

`examples/archive_scripts/` contains experimental scripts from development that are no longer part of the main workflow but kept for reference.

## Notes

- All scripts use **unsupervised training**: phased data is only used for evaluation, never for training
- Default configuration matches the manuscript: 6 loci (HLA-A, C, B, DRB1, DQB1, DPB1)
- GPU training is supported: use `--device cuda` for faster training
