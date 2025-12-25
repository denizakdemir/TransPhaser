
import unittest
import torch
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from transphaser.runner import TransPhaserRunner
from transphaser.config import HLAPhasingConfig
from transphaser.model import TransPhaser

class TestTransPhaserRunner(unittest.TestCase):
    
    def setUp(self):
        # Create temp dir
        self.test_dir = tempfile.mkdtemp()
        
        # Create dummy data
        self.locus_cols = ["HLA-A", "HLA-B"]
        data = {
            "HLA-A": ["A*01:01/A*02:01", "A*03:01/A*03:01"], # Het, Hom
            "HLA-B": ["B*07:02/B*08:01", "B*44:02/B*44:02"],
            "Population": ["EUR", "AFR"],
            "AgeGroup": ["Adult", "Child"]
        }
        self.df_unphased = pd.DataFrame(data)
        
        # Dummy phased data
        phased_data = {
            "Haplotype1": ["A*01:01_B*07:02", "A*03:01_B*44:02"],
            "Haplotype2": ["A*02:01_B*08:01", "A*03:01_B*44:02"]
        }
        self.df_phased = pd.DataFrame(phased_data)
        
        # Config
        self.config = HLAPhasingConfig(
            model_name="test_model",
            output_dir=self.test_dir,
            data={
                "locus_columns": self.locus_cols,
                "covariate_columns": ["Population"],
                "categorical_covariate_columns": ["Population"],
                "validation_split_ratio": 0.5
            },
            model={
                "embedding_dim": 4,
                "latent_dim": 2,
            },
            training={
                "epochs": 2,
                "batch_size": 2,
                "early_stopping_patience": 5  # Added to prevent None value error
            }
        )
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_end_to_end_training(self):
        """Test full training loop basic execution."""
        runner = TransPhaserRunner(self.config)
        results = runner.run(self.df_unphased, self.df_phased)
        
        self.assertIn("training_history", results)
        self.assertIn("final_metrics", results)
        self.assertTrue(len(results["training_history"]["train_loss"]) > 0)
        
    def test_persistence(self):
        """Test save and load."""
        runner = TransPhaserRunner(self.config)
        # Train briefly to build model
        runner.run(self.df_unphased, self.df_phased)
        
        save_path = os.path.join(self.test_dir, "model.pt")
        runner.save(save_path)
        
        # New runner
        runner2 = TransPhaserRunner(self.config)
        runner2.load(save_path)
        
        self.assertIsNotNone(runner2.model)
        self.assertIsNotNone(runner2.tokenizer)
        
        # Check if weights match
        w1 = runner.model.proposal.phasing_head.weight
        w2 = runner2.model.proposal.phasing_head.weight
        self.assertTrue(torch.allclose(w1, w2))
        
    def test_helper_methods(self):
        """Test helper inference methods."""
        runner = TransPhaserRunner(self.config)
        runner.run(self.df_unphased, self.df_phased)
        
        # Most likely
        preds = runner.get_most_likely_haplotypes(self.df_unphased)
        self.assertEqual(len(preds), len(self.df_unphased))
        self.assertIn("Haplotype1", preds.columns)
        self.assertIn("Probability", preds.columns)
        
        # Frequencies
        freqs = runner.get_haplotype_frequencies(self.df_unphased)
        self.assertIn("Frequency", freqs.columns)
        
        # Embeddings
        embeds = runner.get_embeddings()
        self.assertIn("HLA-A", embeds)
        self.assertEqual(embeds["HLA-A"].shape[1], 4) # dim
        
        # Plotting (smoke test)
        plot_path = os.path.join(self.test_dir, "plot.png")
        runner.plot_embeddings("HLA-A", output_path=plot_path)
        self.assertTrue(os.path.exists(plot_path))

if __name__ == '__main__':
    unittest.main()
