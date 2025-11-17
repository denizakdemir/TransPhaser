# TransPhaser: Transformer-Based HLA Phasing Suite

TransPhaser is a deep learning framework for phasing HLA genotypes using transformer-based variational autoencoders. It predicts haplotype pairs from unphased genotype data, incorporating covariate information and handling missing data.

## Features

- **Transformer Architecture**: Encoder-decoder transformers for learning latent representations and generating haplotypes
- **Variational Inference**: ELBO-based training with KL annealing for robust learning
- **Multi-Locus Support**: Handle arbitrary numbers of HLA loci simultaneously
- **Covariate Integration**: Incorporate population, age, and other covariate information
- **Missing Data Handling**: Robust strategies for dealing with missing alleles
- **Modular Design**: Clean separation of preprocessing, training, evaluation, and persistence

## Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:denizakdemir/TransPhaser.git
   cd TransPhaser
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Generate Synthetic Data

```bash
python examples/generate_synthetic_data.py
```

This creates sample datasets in `examples/data/`:
- `synthetic_genotypes_unphased.csv` - Unphased genotype data
- `synthetic_haplotypes_phased.csv` - Ground truth haplotypes
- `synthetic_genotypes_unphased_missing.csv` - Data with missing alleles

### Run Example Workflow

```bash
cd examples
python run_phaser_example.py
```

This will:
1. Train a model on synthetic data (5 epochs)
2. Generate predictions on validation set
3. Evaluate phasing accuracy
4. Save trained model and results to `examples/output/`

## Input Data Format

TransPhaser expects CSV files with specific column structures:

### Unphased Genotype Data

| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| IndividualID | Unique sample identifier | String | `SAMPLE_0001` |
| HLA-* | HLA locus genotypes | `allele1/allele2` or `allele1,allele2` | `A*01/A*02` |
| Covariates | Population, age, etc. | String or numeric | `EUR`, `19-40` |

**Supported Genotype Formats:**
- **Slash-separated** (recommended): `A*01/A*02`, `B*07/B*08`
- **Comma-separated**: `A*01:01,A*02:01`, `B*07:02,B*08:01`
- **Homozygous**: Single allele automatically duplicated: `A*01` → `A*01/A*01`

**Example:**
```csv
IndividualID,HLA-A,HLA-B,HLA-DRB1,Population,AgeGroup
SAMPLE_0001,A*02/A*05,B*06/B*07,DRB1*02/DRB1*03,ASN,0-18
SAMPLE_0002,A*04/A*04,B*01/B*06,DRB1*01/DRB1*03,EUR,65+
```

### Phased Haplotype Data (Ground Truth)

| Column | Description | Format | Example |
|--------|-------------|--------|---------|
| IndividualID | Must match unphased data | String | `SAMPLE_0001` |
| Haplotype1 | First haplotype | `allele1_allele2_...` | `A*02_B*07_DRB1*02` |
| Haplotype2 | Second haplotype | `allele1_allele2_...` | `A*05_B*06_DRB1*03` |
| Covariates | Must match unphased data | Same as unphased | `ASN`, `0-18` |

**Example:**
```csv
IndividualID,Haplotype1,Haplotype2,Population,AgeGroup
SAMPLE_0001,A*02_B*07_DRB1*02,A*05_B*06_DRB1*03,ASN,0-18
SAMPLE_0002,A*04_B*06_DRB1*01,A*04_B*01_DRB1*03,EUR,65+
```

## Configuration

TransPhaser uses Pydantic-based configuration. See `examples/run_phaser_example.py` for a complete example:

```python
from transphaser.config import HLAPhasingConfig

config = HLAPhasingConfig(
    model_name="MyModel",
    seed=42,
    device="cpu",  # or "cuda"
    output_dir="output",
    data={
        "unphased_data_path": "data/unphased.csv",
        "phased_data_path": "data/phased.csv",
        "locus_columns": ["HLA-A", "HLA-B", "HLA-DRB1"],
        "covariate_columns": ["Population", "AgeGroup"],
        "categorical_covariate_columns": ["Population", "AgeGroup"],
        "validation_split_ratio": 0.2
    },
    model={
        "embedding_dim": 64,
        "latent_dim": 32,
        "encoder": {"num_layers": 2, "num_heads": 4, "dropout": 0.1, "ff_dim": 128},
        "decoder": {"num_layers": 2, "num_heads": 4, "dropout": 0.1, "ff_dim": 128}
    },
    training={
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 10,
        "kl_annealing_type": "linear"
    }
)
```

## Usage Example

### Training and Prediction

```python
from transphaser.runner import HLAPhasingRunner
from transphaser.config import HLAPhasingConfig

# Create configuration
config = HLAPhasingConfig(...)

# Initialize runner
runner = HLAPhasingRunner(config)

# Train model and generate predictions
runner.run()

# Save trained model
runner.save_model("my_model.pt")
```

### Loading and Using a Trained Model

```python
# Create new config for prediction
predict_config = HLAPhasingConfig(
    output_dir="predictions",
    data={
        "unphased_data_path": "new_data.csv",
        "locus_columns": ["HLA-A", "HLA-B", "HLA-DRB1"],
        "covariate_columns": ["Population", "AgeGroup"],
        "categorical_covariate_columns": ["Population", "AgeGroup"]
    }
)

# Initialize new runner
predict_runner = HLAPhasingRunner(predict_config)

# Load model and predict
predict_runner.predict(model_path="my_model.pt")
```

## Output Files

TransPhaser generates the following outputs in the specified output directory:

| File | Description |
|------|-------------|
| `predictions.csv` | Predicted haplotype pairs with sample IDs |
| `trained_model.pt` | Complete model checkpoint with weights, tokenizer, and config |
| `checkpoints/` | Epoch checkpoints and best model during training |
| `final_report_*.json` | Evaluation metrics (accuracy, Hamming distance, switch errors) |
| `final_report_*.txt` | Human-readable report |
| `training_loss_curves.png` | Training and validation loss visualization |

## Testing

Run the full test suite (101 tests including integration tests):

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
# Unit tests only
pytest tests/ -v -k "not integration"

# Integration tests only
pytest tests/test_integration.py -v

# Specific component
pytest tests/test_data_preprocessing.py -v
```

## Project Structure

```
TransPhaser/
├── transphaser/            # Core package
│   ├── config.py          # Configuration management
│   ├── data_preprocessing.py  # Data parsing and tokenization
│   ├── embeddings.py      # Allele and locus embeddings
│   ├── encoder.py         # Transformer encoder
│   ├── decoder.py         # Transformer decoder
│   ├── model.py           # Main HLA phasing model
│   ├── trainer.py         # Training loop
│   ├── runner.py          # High-level workflow orchestration
│   ├── evaluation.py      # Metrics and evaluation
│   ├── persistence.py     # Model save/load
│   └── ...
├── examples/              # Example scripts and data
│   ├── generate_synthetic_data.py
│   ├── run_phaser_example.py
│   └── data/             # Synthetic datasets
├── tests/                # Comprehensive test suite (101 tests)
│   ├── test_integration.py   # End-to-end tests
│   ├── test_data_preprocessing.py
│   └── ...
├── README.md
└── requirements.txt
```

## Performance

On synthetic data (5000 samples, 3 loci, 5 epochs):
- **Phasing Accuracy**: ~45% (significantly better than random guessing)
- **Avg Hamming Distance**: ~1.1 alleles per haplotype
- **Training Time**: ~8 seconds on CPU
- **Model Size**: ~625 KB (147K parameters)

Performance improves with:
- More training epochs
- Larger and more diverse datasets
- GPU acceleration for larger models
- Fine-tuning hyperparameters

## Known Issues and Limitations

1. **Modest accuracy on synthetic data**: The model achieves ~45% accuracy on synthetic data after minimal training. This is expected for:
   - Small synthetic datasets with random haplotype frequencies
   - Limited training epochs (5)
   - Simple model architecture

   Real-world performance will depend on dataset quality and training.

2. **Data format**: Input data must follow the specified CSV format. Validation errors will be raised for:
   - Missing required columns
   - Invalid genotype formats
   - Mismatched sample IDs between unphased/phased data

## Troubleshooting

### Common Issues

**Problem**: All predictions are "PAD_PAD_PAD"
- **Cause**: Genotype parser format mismatch (fixed in v0.2.0)
- **Solution**: Ensure genotypes use slash (`/`) or comma (`,`) separators, not other delimiters

**Problem**: Low phasing accuracy
- **Cause**: Insufficient training, small dataset, or suboptimal hyperparameters
- **Solution**:
  - Increase training epochs
  - Use larger, more diverse training data
  - Tune model architecture (embedding_dim, num_layers, etc.)
  - Enable GPU for faster iteration

**Problem**: Out of memory during training
- **Solution**:
  - Reduce batch size
  - Reduce model dimensions (embedding_dim, latent_dim)
  - Use gradient accumulation

## Recent Changes

### Version 0.2.0 (October 2025)
- **Critical Fix**: Fixed genotype parser to support slash-separated format (`A*01/A*02`)
- Added comprehensive integration tests (5 new tests)
- Improved documentation with clear input format specifications
- Added test for format validation (both slash and comma formats)
- Phasing accuracy improved from 0% to 45% after parser fix

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## Citation

If you use TransPhaser in your research, please cite:

```
@software{transphaser2025,
  title={TransPhaser: Transformer-Based HLA Phasing Suite},
  author={Akdemir, Deniz},
  year={2025},
  url={https://github.com/denizakdemir/TransPhaser}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Using Custom Agents in VS Code

VS Code detects `.agent.md` files under `.github/agents` and shows them in the agents dropdown. Select an agent to apply its role‑specific instructions and tools; use handoff buttons to move between agents with context.

- Codebase Finisher: `.github/agents/finisher.agent.md`
- Test Pilot: `.github/agents/test-pilot.agent.md`
- Statistical Methods Auditor: `.github/agents/stats-auditor.agent.md`
- Data Validator: `.github/agents/data-validator.agent.md`
- Performance/Hardware Tuner: `.github/agents/perf-tuner.agent.md`
- Packaging/Release Manager: `.github/agents/release-manager.agent.md`
- Docs/Examples Doctor: `.github/agents/docs-doctor.agent.md`
- Security/Dependency Auditor: `.github/agents/security-auditor.agent.md`

Three‑Step Handoff Workflow (example)
1) Plan → Finisher: Produce a Project Snapshot (purpose, stack, core API, current issues) and a 5–7 step roadmap to reach a v0.1.0‑quality state. Assume CPU unless CUDA is available based on environment.
2) Tests → Test Pilot: Generate focused, runnable pytest tests for targeted modules (1 happy path + 1–2 critical edges each). Mock heavy Torch ops and file I/O.
3) Methods → Stats Auditor: Review splitting/preprocessing order, metrics, and seed handling; return actionable fixes. Optionally hand back to Finisher to apply minimal code changes.

Starter Prompts
- Finisher: “Scan the TransPhaser repo and produce a Project Snapshot: purpose, stack, core public API, and current issues (tests, structure, docs). Check env constraints from `pyproject.toml` and `requirements.txt`, assume CPU unless CUDA is available. Propose a 5–7 step, low‑risk roadmap to reach a v0.1.0‑quality state (tests passing, minimal docs), naming first target modules and tests to add. Ask one clarifying question, then wait for confirmation; do not edit code yet.”
- Test Pilot: “Generate focused pytest tests for `transphaser/config.py`, `transphaser/runner.py`, and `transphaser/data_preprocessing.py`: 1 happy path + 1–2 critical edges each. Assume CPU; mock heavy Torch ops and file I/O. Output runnable test files only.”
- Stats Auditor: “Audit splitting, preprocessing order, metrics, and seed handling across `runner.py`, `trainer.py`, and `evaluation.py`. Deliver a Methodology Audit Report with Critical/Medium/Low issues and concrete fixes.”
- Data Validator: “Validate datasets at `examples/data/*` (or given paths): required loci columns, allele/allele format, covariate levels, missing policies. Produce a Data Validation Report with suggested fixes.”
- Perf/Hardware Tuner: “Propose low‑risk runtime tuning (device selection, batch sizes, DataLoader workers/pin_memory, optional AMP on CUDA) with acceptance criteria; keep reproducibility.”
- Release Manager: “Draft a release checklist (semver bump, `pyproject` sanity, sdist/wheel build + clean install smoke, README/test instructions). No code edits yet.”
- Docs/Examples Doctor: “Propose docstrings for key public APIs and a README quickstart showing `HLAPhasingConfig` + `HLAPhasingRunner` on a tiny in‑memory dataset.”
- Security/Dependency Auditor: “Review dependency and license risks (torch/numpy/pandas/sklearn); recommend safe pins and wheel channels; note any code hotspots (I/O, dynamic imports).”
