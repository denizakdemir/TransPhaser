# TransPhaser: Neural Expectation-Maximization for HLA Phasing


TransPhaser is a deep learning framework for phasing HLA genotypes using transformer-based Neural Expectation-Maximization. It combines the flexibility of neural networks with the structural constraints of probabilistic graphical models.

---

## 🎯 What Makes TransPhaser Unique

### **Neural EM Architecture**
TransPhaser uses a **Neural Proposal Network** to amortize the expensive E-step of the Expectation-Maximization algorithm. This allows it to:
- ✅ **Respect Genotype Constraints**: Enforces that predicted haplotype pairs must reconstruct the observed genotypes.
- ✅ **Learn from Unlabeled Data**: Uses genotype reconstruction likelihood.
- ✅ **Scale Efficiently**: Avoids enumerating all possible phasings.
- ✅ **Leverage Phased Data**: Can opportunistically use phased reference panels (semi-supervised) for improved accuracy.

---

## 🚀 Key Features

- **Transformer Architecture**: Captures complex linkage disequilibrium patterns across loci.
- **Probabilistic Foundation**: Rigorous likelihood-based objective (ELBO).
- **Embedded Priors**: Learns conditional haplotype priors (`P(h_k | h_{<k})`) and allele embeddings.
- **Robustness**: Handles missing data and genotype ambiguity.
- **Easy-to-Use API**: Simple python interface for training and inference.
- **Persistence**: Save and load full model states including tokenizers.

---

## 📊 Performance

TransPhaser outperforms classical baselines on realistic 6-locus HLA data:

| Method | Phasing Accuracy | Hamming Distance | Switch Errors |
|--------|------------------|------------------|---------------|
| **TransPhaser** | **83.55%** | **0.55** | **0.34** |
| Beagle 5.4 | 79.35% | 0.70 | 0.37 |
| EM Baseline | 54.25% | 1.78 | 1.06 |
| Frequency Baseline | 48.50% | 1.97 | 1.18 |
| Random Baseline | 14.85% | 3.43 | 1.98 |

*Results based on 10-epoch training run (`examples/train_comprehensive.py`) on 10,000 synthetic samples.*

---

## 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone git@github.com:denizakdemir/TransPhaser.git
   cd TransPhaser
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -e .
   ```

---

## ⚡ Quick Start

### 1. Generate Realistic Data
Generate synthetic yet biologically realistic HLA data (with linkage disequilibrium):
```bash
python examples/generate_realistic_data.py
```

### 2. Run Comprehensive Training
Train TransPhaser and compare against baselines (EM, Frequency, Beagle):
```bash
python examples/train_comprehensive.py --epochs 100 --device cpu
```

---

## 🛠️ Configuration & Usage

TransPhaser uses `TransPhaserConfig` for flexible configuration:

```python
from transphaser.config import TransPhaserConfig
from transphaser.runner import TransPhaserRunner

# Simple configuration
config = TransPhaserConfig(
    data={
        "unphased_data_path": "examples/data/realistic_genotypes_unphased.csv",
        "phased_data_path": "examples/data/realistic_haplotypes_phased.csv",  # Optional (for semi-supervised / eval)
        "locus_columns": ["HLA-A", "HLA-C", "HLA-B", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"],
    },
    training={"epochs": 100}
)

runner = TransPhaserRunner(config)
runner.run(df_unphased, df_phased=None) # Train
runner.save("my_model.pt")

# Inference
predictions = runner.get_most_likely_haplotypes(new_data)
print(predictions.head())
```

---

## 📂 Project Structure

```
TransPhaser/
├── transphaser/          # Core package
│   ├── model.py          # TransPhaser Model & Loss
│   ├── runner.py         # Main API (Runner)
│   ├── config.py         # Configuration classes
│   ├── data_preprocessing.py  # Data handling
│   ├── em.py             # EM baseline
│   ├── evaluation.py     # Metrics
│   └── beagle_runner.py  # Beagle wrapper
├── examples/             # Example scripts
│   ├── train_comprehensive.py       # Full benchmarking script
│   └── generate_realistic_data.py   # Data generation
├── tests/                # Comprehensive test suite
├── Manuscript/           # LaTeX manuscript
└── README.md
```

---

## 🧪 Testing

Run the full test suite:
```bash
python -m unittest discover tests
```

---

## 📄 Citation

If you use TransPhaser in your research, please cite:

```bibtex
@software{transphaser2025,
  title={TransPhaser: Neural Expectation-Maximization for HLA Phasing},
  author={Akdemir, Deniz},
  year={2025},
  url={https://github.com/denizakdemir/TransPhaser}
}
```

---
