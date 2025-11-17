"""
Integration and end-to-end tests for TransPhaser.

These tests verify that the entire pipeline works correctly from
data loading through prediction and evaluation.
"""

import unittest
import os
import tempfile
import shutil
import pandas as pd
import torch

from transphaser.config import HLAPhasingConfig
from transphaser.runner import HLAPhasingRunner


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests for the complete TransPhaser pipeline."""

    def setUp(self):
        """Set up test fixtures for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.data_dir, exist_ok=True)

        # Create minimal synthetic data with slash-separated format
        unphased_data = {
            'IndividualID': [f'S{i:03d}' for i in range(50)],
            'HLA-A': ['A*01/A*02', 'A*02/A*03', 'A*01/A*01'] * 16 + ['A*02/A*03', 'A*01/A*02'],
            'HLA-B': ['B*07/B*08', 'B*08/B*08', 'B*07/B*15'] * 16 + ['B*08/B*15', 'B*07/B*07'],
            'Population': ['EUR', 'ASN', 'AFR'] * 16 + ['EUR', 'ASN'],
            'AgeGroup': ['0-18', '19-40', '41-65', '65+'] * 12 + ['0-18', '19-40']
        }
        self.unphased_path = os.path.join(self.data_dir, "unphased.csv")
        pd.DataFrame(unphased_data).to_csv(self.unphased_path, index=False)

        # Create corresponding phased ground truth
        phased_data = {
            'IndividualID': [f'S{i:03d}' for i in range(50)],
            'Haplotype1': ['A*01_B*07', 'A*02_B*08', 'A*01_B*07'] * 16 + ['A*02_B*08', 'A*01_B*07'],
            'Haplotype2': ['A*02_B*08', 'A*03_B*08', 'A*01_B*15'] * 16 + ['A*03_B*15', 'A*02_B*07'],
            'Population': ['EUR', 'ASN', 'AFR'] * 16 + ['EUR', 'ASN'],
            'AgeGroup': ['0-18', '19-40', '41-65', '65+'] * 12 + ['0-18', '19-40']
        }
        self.phased_path = os.path.join(self.data_dir, "phased.csv")
        pd.DataFrame(phased_data).to_csv(self.phased_path, index=False)

    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_pipeline_with_training(self):
        """Test the complete pipeline: training, prediction, and evaluation."""
        # Create configuration
        config = HLAPhasingConfig(
            model_name="IntegrationTest",
            seed=42,
            device="cpu",
            output_dir=self.output_dir,
            data={
                "unphased_data_path": self.unphased_path,
                "phased_data_path": self.phased_path,
                "locus_columns": ["HLA-A", "HLA-B"],
                "covariate_columns": ["Population", "AgeGroup"],
                "categorical_covariate_columns": ["Population", "AgeGroup"],
                "validation_split_ratio": 0.2
            },
            model={
                "embedding_dim": 32,
                "latent_dim": 16,
                "encoder": {"num_layers": 1, "num_heads": 2, "dropout": 0.1, "ff_dim": 64},
                "decoder": {"num_layers": 1, "num_heads": 2, "dropout": 0.1, "ff_dim": 64},
            },
            training={
                "batch_size": 8,
                "learning_rate": 1e-3,
                "epochs": 2,  # Minimal training for speed
                "kl_annealing_type": "linear",
                "kl_annealing_epochs": 1,
            },
            reporting={
                "formats": ["json"],
                "base_filename": "integration_test_report",
            }
        )

        # Initialize runner and run full workflow
        runner = HLAPhasingRunner(config)
        runner.run()

        # Verify outputs exist
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "predictions.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "integration_test_report.json")))

        # Verify predictions file has correct format
        predictions_df = pd.read_csv(os.path.join(self.output_dir, "predictions.csv"))
        self.assertIn('IndividualID', predictions_df.columns)
        self.assertIn('Predicted_Haplotype1', predictions_df.columns)
        self.assertIn('Predicted_Haplotype2', predictions_df.columns)
        self.assertGreater(len(predictions_df), 0)

        # Verify predictions are not all PAD tokens (the bug we fixed)
        predictions_list = predictions_df['Predicted_Haplotype1'].tolist()
        non_pad_predictions = [p for p in predictions_list if p != 'PAD_PAD']
        self.assertGreater(len(non_pad_predictions), 0,
                          "All predictions are PAD tokens - parser bug may have returned")

    def test_model_save_and_load(self):
        """Test model persistence: save, load, and predict."""
        # Create minimal config
        config = HLAPhasingConfig(
            model_name="SaveLoadTest",
            seed=42,
            device="cpu",
            output_dir=self.output_dir,
            data={
                "unphased_data_path": self.unphased_path,
                "phased_data_path": self.phased_path,
                "locus_columns": ["HLA-A", "HLA-B"],
                "covariate_columns": ["Population", "AgeGroup"],
                "categorical_covariate_columns": ["Population", "AgeGroup"],
            },
            model={
                "embedding_dim": 32,
                "latent_dim": 16,
                "encoder": {"num_layers": 1, "num_heads": 2, "dropout": 0.1, "ff_dim": 64},
                "decoder": {"num_layers": 1, "num_heads": 2, "dropout": 0.1, "ff_dim": 64},
            },
            training={
                "batch_size": 8,
                "learning_rate": 1e-3,
                "epochs": 1,
            },
            reporting={"formats": ["json"], "base_filename": "save_load_report"}
        )

        # Train and save model
        runner = HLAPhasingRunner(config)
        runner.run()
        model_path = os.path.join(self.output_dir, "test_model.pt")
        runner.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        # Create new runner and load model
        config2 = config.copy(deep=True)
        config2.output_dir = os.path.join(self.test_dir, "output2")
        config2.reporting.base_filename = "loaded_model_report"
        runner2 = HLAPhasingRunner(config2)
        runner2.predict(model_path=model_path)

        # Verify predictions from loaded model
        predictions_path = os.path.join(config2.output_dir, "predictions.csv")
        self.assertTrue(os.path.exists(predictions_path))
        predictions_df = pd.read_csv(predictions_path)
        self.assertGreater(len(predictions_df), 0)

    def test_input_format_validation(self):
        """Test that both slash and comma-separated formats work."""
        # Test with comma-separated format
        comma_data = {
            'IndividualID': ['S001', 'S002'],
            'HLA-A': ['A*01:01,A*02:01', 'A*03:01,A*03:01'],
            'HLA-B': ['B*07:02,B*08:01', 'B*15:01,B*15:01'],
            'Population': ['EUR', 'ASN'],
            'AgeGroup': ['0-18', '19-40']
        }
        comma_path = os.path.join(self.data_dir, "comma_format.csv")
        pd.DataFrame(comma_data).to_csv(comma_path, index=False)

        # Test with slash-separated format
        slash_data = {
            'IndividualID': ['S001', 'S002'],
            'HLA-A': ['A*01/A*02', 'A*03/A*03'],
            'HLA-B': ['B*07/B*08', 'B*15/B*15'],
            'Population': ['EUR', 'ASN'],
            'AgeGroup': ['0-18', '19-40']
        }
        slash_path = os.path.join(self.data_dir, "slash_format.csv")
        pd.DataFrame(slash_data).to_csv(slash_path, index=False)

        # Test both formats can be parsed
        from transphaser.data_preprocessing import GenotypeDataParser

        parser = GenotypeDataParser(
            locus_columns=['HLA-A', 'HLA-B'],
            covariate_columns=['Population', 'AgeGroup']
        )

        # Parse comma format
        comma_df = pd.read_csv(comma_path)
        parsed_comma, _ = parser.parse(comma_df)
        self.assertEqual(len(parsed_comma), 2)
        # First sample should have A*01:01 and A*02:01 (sorted)
        self.assertEqual(set(parsed_comma[0][0]), {'A*01:01', 'A*02:01'})

        # Parse slash format
        slash_df = pd.read_csv(slash_path)
        parsed_slash, _ = parser.parse(slash_df)
        self.assertEqual(len(parsed_slash), 2)
        # First sample should have A*01 and A*02 (sorted)
        self.assertEqual(set(parsed_slash[0][0]), {'A*01', 'A*02'})


class TestDataPreprocessingIntegration(unittest.TestCase):
    """Integration tests for data preprocessing components."""

    def test_tokenization_roundtrip(self):
        """Test that tokenization and detokenization are consistent."""
        from transphaser.data_preprocessing import AlleleTokenizer

        tokenizer = AlleleTokenizer()
        locus = "HLA-A"
        alleles = ["A*01:01", "A*02:01", "A*03:01"]

        # Build vocabulary
        tokenizer.build_vocabulary(locus, alleles)

        # Tokenize and detokenize each allele
        for allele in alleles:
            token = tokenizer.tokenize(locus, allele)
            detokenized = tokenizer.detokenize(locus, token)
            self.assertEqual(allele, detokenized,
                           f"Tokenization roundtrip failed for {allele}")

    def test_dataset_batch_generation(self):
        """Test that HLADataset generates valid batches."""
        from transphaser.data_preprocessing import (
            GenotypeDataParser, AlleleTokenizer, CovariateEncoder, HLADataset
        )
        from torch.utils.data import DataLoader

        # Create minimal data
        data = {
            'HLA-A': ['A*01/A*02', 'A*03/A*03'],
            'HLA-B': ['B*07/B*08', 'B*15/B*15'],
            'Population': ['EUR', 'ASN'],
            'AgeGroup': ['0-18', '19-40']
        }
        df = pd.DataFrame(data)

        # Parse
        parser = GenotypeDataParser(
            locus_columns=['HLA-A', 'HLA-B'],
            covariate_columns=['Population', 'AgeGroup']
        )
        genotypes, covariates_df = parser.parse(df)

        # Build tokenizer
        tokenizer = AlleleTokenizer()
        tokenizer.build_vocabulary_from_dataframe(df, ['HLA-A', 'HLA-B'])

        # Encode covariates
        encoder = CovariateEncoder(
            categorical_covariates=['Population', 'AgeGroup']
        )
        covariates_encoded = encoder.fit_transform(covariates_df).to_numpy()

        # Create dataset
        dataset = HLADataset(
            genotypes=genotypes,
            covariates=covariates_encoded,
            tokenizer=tokenizer,
            loci_order=['HLA-A', 'HLA-B']
        )

        # Create dataloader
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))

        # Verify batch structure
        self.assertIn('genotype_tokens', batch)
        self.assertIn('covariates', batch)
        self.assertEqual(batch['genotype_tokens'].shape[0], 2)  # batch size
        self.assertEqual(batch['genotype_tokens'].shape[1], 4)  # 2 loci * 2 alleles
        self.assertTrue(torch.is_tensor(batch['genotype_tokens']))
        self.assertTrue(torch.is_tensor(batch['covariates']))


if __name__ == '__main__':
    unittest.main()
