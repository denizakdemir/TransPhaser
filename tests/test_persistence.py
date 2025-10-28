import unittest
import os
import tempfile
import shutil
import torch
import pandas as pd
from transphaser.runner import HLAPhasingRunner
from transphaser.config import HLAPhasingConfig

class TestModelPersistence(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dummy_unphased_path = os.path.join(self.test_dir, "unphased.csv")
        self.dummy_phased_path = os.path.join(self.test_dir, "phased.csv")

        pd.DataFrame({
            'IndividualID': [f'S{i}' for i in range(1, 41)],
            'HLA-A': [f'A*01:0{i%2 + 1}/A*02:0{i%2 + 1}' for i in range(1, 41)],
            'HLA-B': [f'B*07:0{i%2 + 1}/B*08:0{i%2 + 1}' for i in range(1, 41)],
            'HLA-DRB1': [f'DRB1*01:0{i%2 + 1}/DRB1*03:0{i%2 + 1}' for i in range(1, 41)],
            'Population': ['CEU' if i % 2 == 0 else 'YRI' for i in range(1, 41)],
            'AgeGroup': ['Young' if i < 20 else 'Old' for i in range(1, 41)]
        }).to_csv(self.dummy_unphased_path, index=False)

        pd.DataFrame({
            'IndividualID': [f'S{i}' for i in range(1, 41)],
            'Haplotype1': [f'A*01:0{i%2 + 1}_B*07:0{i%2 + 1}_DRB1*01:0{i%2 + 1}' for i in range(1, 41)],
            'Haplotype2': [f'A*02:0{i%2 + 1}_B*08:0{i%2 + 1}_DRB1*03:0{i%2 + 1}' for i in range(1, 41)]
        }).to_csv(self.dummy_phased_path, index=False)

        self.config = HLAPhasingConfig(
            model_name="TestPersistenceModel",
            seed=42,
            device="cpu",
            output_dir=self.test_dir,
            data={
                "unphased_data_path": self.dummy_unphased_path,
                "phased_data_path": self.dummy_phased_path,
                "locus_columns": ["HLA-A", "HLA-B", "HLA-DRB1"],
                "covariate_columns": ["Population", "AgeGroup"],
                "categorical_covariate_columns": ["Population", "AgeGroup"],
            },
            model={
                "embedding_dim": 8,
                "latent_dim": 4,
                "encoder": {"num_layers": 1, "num_heads": 1, "dropout": 0.1, "ff_dim": 16},
                "decoder": {"num_layers": 1, "num_heads": 1, "dropout": 0.1, "ff_dim": 16},
            },
            training={
                "batch_size": 2,
                "learning_rate": 1e-4,
                "epochs": 1,
                "kl_annealing_type": "linear",
                "kl_annealing_epochs": 1,
            },
            reporting={
                "formats": ["json"],
                "base_filename": "test_report",
                "plot_filename": "test_plot.png",
            }
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_and_load_model(self):
        # --- 1. Train and Save a Model ---
        runner_train = HLAPhasingRunner(self.config)
        runner_train._set_seeds()
        df_unphased, df_phased_truth = runner_train._load_data()
        runner_train._preprocess_data(df_unphased, df_phased_truth)
        runner_train._build_model()

        # Manually save the initial, untrained model
        initial_model_path = os.path.join(self.test_dir, "initial_model.pt")
        runner_train.save_model(initial_model_path)

        # --- 2. Load the Model into a New Runner ---
        runner_load = HLAPhasingRunner(self.config)
        runner_load._set_seeds()
        df_unphased_load, _ = runner_load._load_data()
        runner_load._preprocess_data(df_unphased_load, None)
        runner_load._build_model()
        runner_load.load_model(initial_model_path)

        # --- 3. Verify Model State ---
        # Check that the loaded model's state dict matches the original
        original_state = runner_train.model.state_dict()
        loaded_state = runner_load.model.state_dict()

        for key in original_state:
            self.assertTrue(torch.equal(original_state[key], loaded_state[key]))

        # --- 4. Verify Prediction Consistency ---
        # Get a batch of data for prediction
        runner_load.model.eval()
        with torch.no_grad():
            for batch in runner_load.val_loader:
                # Prepare batch for prediction
                pred_batch, sample_ids = runner_load._prepare_prediction_batch(batch)
                if pred_batch is None:
                    continue

                # Get prediction from the loaded model
                predictions_loaded = runner_load.model.predict_haplotypes(pred_batch)

                # Get prediction from the original model for the same batch
                predictions_original = runner_train.model.predict_haplotypes(pred_batch)

                # Check that predictions are identical
                self.assertTrue(torch.equal(predictions_loaded, predictions_original))
                break # Only need to check one batch

if __name__ == '__main__':
    unittest.main()
