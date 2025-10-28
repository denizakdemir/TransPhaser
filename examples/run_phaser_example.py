import torch
import os
import logging
import pandas as pd
from transphaser.config import HLAPhasingConfig
from transphaser.runner import HLAPhasingRunner

# --- Configuration ---
DATA_DIR = "examples/data"
UNPHASED_DATA_FILE = os.path.join(DATA_DIR, "synthetic_genotypes_unphased.csv")
PHASED_DATA_FILE = os.path.join(DATA_DIR, "synthetic_haplotypes_phased.csv")
OUTPUT_DIR = "examples/output"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "trained_model.pt")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Workflow ---
if __name__ == "__main__":
    logging.info("Starting TransPhaser Example Workflow...")

    # 1. Create Configuration for Training
    config_train = HLAPhasingConfig(
        model_name="TransPhaser-Example-Train",
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
            "encoder": {"num_layers": 2, "num_heads": 4, "dropout": 0.1, "ff_dim": 128},
            "decoder": {"num_layers": 2, "num_heads": 4, "dropout": 0.1, "ff_dim": 128},
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
            "base_filename": "final_report_train",
            "plot_filename": "training_loss_curves.png",
        }
    )

    # 2. Initialize and Run the Training Workflow
    logging.info("--- Training and Saving the Model ---")
    runner_train = HLAPhasingRunner(config_train)
    runner_train.run()
    runner_train.save_model(MODEL_SAVE_PATH)
    logging.info(f"Model trained and saved to {MODEL_SAVE_PATH}")

    # 3. Demonstrate Loading and Predicting with the Saved Model
    logging.info("\n--- Loading and Predicting with a New Runner ---")

    # Create some new data for prediction
    new_data = {
        'IndividualID': ['new_sample_1', 'new_sample_2'],
        'HLA-A': ['A*01:01/A*02:01', 'A*03:01/A*04:01'],
        'HLA-B': ['B*07:01/B*08:01', 'B*09:01/B*10:01'],
        'HLA-DRB1': ['DRB1*01:01/DRB1*02:01', 'DRB1*03:01/DRB1*04:01'],
        'Population': ['CEU', 'YRI'],
        'AgeGroup': ['Young', 'Old']
    }
    new_data_df = pd.DataFrame(new_data)
    new_data_path = os.path.join(DATA_DIR, "new_unphased_data.csv")
    new_data_df.to_csv(new_data_path, index=False)

    # Create a new configuration for prediction.
    config_predict = config_train.copy(deep=True)
    config_predict.data.unphased_data_path = new_data_path
    config_predict.data.phased_data_path = None # No ground truth for prediction
    config_predict.reporting.base_filename = "final_report_predict"

    runner_predict = HLAPhasingRunner(config_predict)
    runner_predict.predict(model_path=MODEL_SAVE_PATH)

    logging.info("Prediction complete on new data with the loaded model.")
    logging.info(f"Prediction results saved in: {config_predict.output_dir}")

    logging.info("\nTransPhaser Example Workflow Finished.")
