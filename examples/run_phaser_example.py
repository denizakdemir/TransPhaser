import torch
import os
import logging
from transphaser.config import HLAPhasingConfig
from transphaser.runner import HLAPhasingRunner

# --- Configuration ---
DATA_DIR = "examples/data"
UNPHASED_DATA_FILE = os.path.join(DATA_DIR, "synthetic_genotypes_unphased.csv")
PHASED_DATA_FILE = os.path.join(DATA_DIR, "synthetic_haplotypes_phased.csv")
OUTPUT_DIR = "examples/output"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Workflow ---
if __name__ == "__main__":
    logging.info("Starting TransPhaser Example Workflow...")

    # 1. Create Configuration
    config = HLAPhasingConfig(
        model_name="TransPhaser-Example",
        seed=SEED,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        data={
            "unphased_data_path": UNPHASED_DATA_FILE,
            "phased_data_path": PHASED_DATA_FILE,
            "locus_columns": ["HLA-A", "HLA-B", "HLA-DRB1"],
            "covariate_columns": ["Population", "AgeGroup"],
            "categorical_covariate_columns": ["Population", "AgeGroup"],
        },
        model={
            "embedding_dim": 64,
            "latent_dim": 32,
            "encoder": {"num_layers": 2, "num_heads": 4, "dropout": 0.1},
            "decoder": {"num_layers": 2, "num_heads": 4, "dropout": 0.1},
        },
        training={
            "batch_size": 32,
            "learning_rate": 1e-4,
            "epochs": 5,
            "kl_annealing_type": "linear",
            "kl_annealing_epochs": 1,
        },
        reporting={
            "formats": ["json", "txt"],
            "base_filename": "final_report",
            "plot_filename": "training_loss_curves.png",
        }
    )

    # 2. Initialize and Run the Workflow
    runner = HLAPhasingRunner(config)
    runner.run()

    logging.info("TransPhaser Example Workflow Finished.")
