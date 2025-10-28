import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
import pandas as pd
import torch

# Assuming the runner class will be in src/runner.py
from transphaser.runner import HLAPhasingRunner # Reverted to src.
# Import config for testing initialization
from transphaser.config import HLAPhasingConfig # Reverted to src.

# Mock necessary components that the runner interacts with
# We don't need their actual logic, just need to check if they are called correctly
MockGenotypeDataParser = MagicMock()
MockAlleleTokenizer = MagicMock()
MockCovariateEncoder = MagicMock()
MockHLADataset = MagicMock()
MockDataLoader = MagicMock()
MockHLAPhasingModel = MagicMock()
MockELBOLoss = MagicMock()
MockKLAnnealingScheduler = MagicMock()
MockOptimizer = MagicMock()
MockHLAPhasingTrainer = MagicMock()
MockHLAPhasingMetrics = MagicMock()
MockPerformanceReporter = MagicMock() # Mock the reporter we just created

# Define dummy return values for mocks if needed by the runner logic
mock_parsed_genotypes = [[['A*01:01', 'A*02:01']]]
mock_parsed_covariates = pd.DataFrame({'Cov1': [1]})
mock_encoded_covariates = pd.DataFrame({'Cov1_enc': [0.5]})
mock_encoded_covariates_np = mock_encoded_covariates.to_numpy()
mock_phased_haplotypes = ['A*01:01']
mock_dataset_instance = MagicMock()
mock_dataloader_instance = MagicMock()
mock_model_instance = MagicMock()
mock_loss_instance = MagicMock()
mock_kl_scheduler_instance = MagicMock()
mock_optimizer_instance = MagicMock()
mock_trainer_instance = MagicMock()
mock_metrics_instance = MagicMock()
mock_reporter_instance = MagicMock()
# Configure mock dataset length (adjust if needed for train/val)
mock_dataset_instance.__len__.return_value = 1 # Set length for val_dataset mock
# Configure mock dataloader to be iterable and yield a sample batch (size 1 for val)
# Ensure it yields 'sample_id' consistent with HLADataset changes
mock_dataloader_instance.__iter__.return_value = iter([{
    'genotype_tokens': torch.tensor([[8, 9]], dtype=torch.long), # Batch size 1
    'covariates': torch.tensor([[0.2, 0.6]], dtype=torch.float32),
    'sample_id': ['S3'] # Use sample ID 'S3' corresponding to dummy_val_df
}])


# Configure mock return values
MockGenotypeDataParser.return_value.parse.return_value = (mock_parsed_genotypes, mock_parsed_covariates)
MockCovariateEncoder.return_value.fit_transform.return_value = mock_encoded_covariates
MockHLADataset.return_value = mock_dataset_instance
MockDataLoader.return_value = mock_dataloader_instance
MockHLAPhasingModel.return_value = mock_model_instance
MockELBOLoss.return_value = mock_loss_instance
MockKLAnnealingScheduler.return_value = mock_kl_scheduler_instance
# Mock the specific optimizer class if needed (e.g., Adam)
# For now, a generic mock optimizer instance is fine
MockOptimizer.return_value = mock_optimizer_instance
MockHLAPhasingTrainer.return_value = mock_trainer_instance
MockHLAPhasingMetrics.return_value = mock_metrics_instance
MockPerformanceReporter.return_value = mock_reporter_instance
# Configure the mock model's .to() method to return itself
mock_model_instance.to.return_value = mock_model_instance

# Mock model's predict method if called by runner
# Ensure it returns a tensor with batch size matching the dataloader yield (size 1)
mock_model_instance.predict_haplotypes.return_value = torch.tensor([[8]]) # Batch size 1, 1 locus

# Mock trainer's train method return value
mock_trainer_instance.train.return_value = ([0.5, 0.4], [0.6, 0.5]) # train_losses, val_losses

# Mock metrics calculator return value
mock_metrics_instance.calculate_metrics.return_value = {"accuracy": 0.95}

# Define a sample return value for the mocked dataset's __getitem__
# This should mimic the structure returned by the actual HLADataset
mock_sample_data = {
    'genotype_tokens': torch.tensor([4, 5, 8, 9], dtype=torch.long), # Example tensor
    'covariates': torch.tensor([0.1, -0.5], dtype=torch.float32), # Example tensor
    'sample_index': 0, # Example index
    # Optionally include target if needed by prediction/evaluation mocks
    'target_haplotype_tokens': torch.tensor([2, 4, 8, 3], dtype=torch.long),
    'target_locus_indices': torch.tensor([-1, 0, 1, -2], dtype=torch.long)
}
mock_dataset_instance.__getitem__.return_value = mock_sample_data


class TestHLAPhasingRunner(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test outputs (config saving etc.)
        self.test_dir = tempfile.mkdtemp()
        # Create dummy data files if the runner tries to load them
        self.dummy_unphased_path = os.path.join(self.test_dir, "unphased.csv")
        self.dummy_phased_path = os.path.join(self.test_dir, "phased.csv")
        # Add more samples to allow train/test split
        pd.DataFrame({
            'IndividualID': ['S1', 'S2', 'S3'],
            'HLA-A': ['A*01:01/A*02:01', 'A*01:01/A*01:01', 'A*02:01/A*03:01'],
            'Cov1': [1, 2, 1]
        }).to_csv(self.dummy_unphased_path, index=False)
        pd.DataFrame({
            'IndividualID': ['S1', 'S2', 'S3'],
            'Haplotype1': ['A*01:01', 'A*01:01', 'A*02:01'],
            'Haplotype2': ['A*02:01', 'A*01:01', 'A*03:01']
        }).to_csv(self.dummy_phased_path, index=False)

        # Create a default config for the runner
        self.config = HLAPhasingConfig()
        # Adjust config paths to use dummy files
        self.config.data.unphased_data_path = self.dummy_unphased_path
        self.config.data.phased_data_path = self.dummy_phased_path
        self.config.output_dir = self.test_dir
        self.config.data.locus_columns = ['HLA-A'] # Simplify for test
        self.config.data.covariate_columns = ['Cov1']
        self.config.training.epochs = 1 # Run only one epoch for faster test
        # Add LR scheduler config for testing
        self.config.training.lr_scheduler = {
             "type": "StepLR",
             "args": {"step_size": 10, "gamma": 0.1}
         }

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_runner_initialization(self):
        """Test if the runner initializes correctly with a config object."""
        # Reset the mock class before the test to ensure isolation
        MockPerformanceReporter.reset_mock()
        # Patch os.makedirs specifically where it's called in runner.py
        # Patch PerformanceReporter specifically where it's imported in runner.py
        with patch('transphaser.runner.os.makedirs') as mock_runner_makedirs, \
             patch('transphaser.runner.PerformanceReporter', MockPerformanceReporter) as mock_runner_reporter_cls:
            # Need to import the runner *inside* the test for patches to apply correctly
            from transphaser.runner import HLAPhasingRunner
            runner = HLAPhasingRunner(config=self.config)
            self.assertEqual(runner.config, self.config)
            # Check initialization calls using the correctly scoped mocks
            mock_runner_makedirs.assert_called_once_with(self.test_dir, exist_ok=True)
            mock_runner_reporter_cls.assert_called_once_with(output_dir=self.test_dir)


    def test_run_workflow_calls(self):
        """Test that the run method calls workflow steps in order."""
        # Define dummy train/val dfs returned by train_test_split mock
        dummy_train_df = pd.DataFrame({'IndividualID': ['S1', 'S2'], 'HLA-A': ['A*01:01/A*02:01', 'A*01:01/A*01:01'], 'Cov1': [1, 2]})
        dummy_val_df = pd.DataFrame({'IndividualID': ['S3'], 'HLA-A': ['A*02:01/A*03:01'], 'Cov1': [1]})
        # Use context managers for patching
        with patch('pandas.read_csv') as mock_read_csv, \
             patch('transphaser.runner.train_test_split', return_value=(dummy_train_df, dummy_val_df)) as mock_tts, \
             patch('transphaser.runner.GenotypeDataParser', MockGenotypeDataParser) as mock_parser_cls, \
             patch('transphaser.runner.AlleleTokenizer', MockAlleleTokenizer) as mock_tokenizer_cls, \
             patch('transphaser.runner.CovariateEncoder', MockCovariateEncoder) as mock_cov_encoder_cls, \
             patch('transphaser.runner.HLADataset', MockHLADataset) as mock_dataset_cls, \
             patch('transphaser.runner.DataLoader', MockDataLoader) as mock_loader_cls, \
             patch('transphaser.runner.HLAPhasingModel', MockHLAPhasingModel) as mock_model_cls, \
             patch('transphaser.runner.ELBOLoss', MockELBOLoss) as mock_loss_cls, \
             patch('transphaser.runner.KLAnnealingScheduler', MockKLAnnealingScheduler) as mock_kl_cls, \
              patch('transphaser.runner.Adam', MockOptimizer) as mock_optimizer_cls, \
              patch('transphaser.runner.StepLR') as mock_step_lr_cls, \
              patch('transphaser.runner.HLAPhasingTrainer', MockHLAPhasingTrainer) as mock_trainer_cls, \
              patch('transphaser.runner.HLAPhasingMetrics', MockHLAPhasingMetrics) as mock_metrics_cls, \
              patch('transphaser.runner.PerformanceReporter', MockPerformanceReporter) as mock_reporter_cls, \
              patch('torch.save') as mock_torch_save, \
              patch('transphaser.runner.os.makedirs') as mock_runner_makedirs: # Patch os.makedirs specifically in runner

            # Configure mock tokenizer detokenize method inside the 'with' block
            mock_tokenizer_cls.return_value.detokenize.return_value = "A*01:01" # Return a dummy allele string

            # Need to import the runner *inside* the test for patches to apply correctly
            from transphaser.runner import HLAPhasingRunner
            # Mock read_csv to return dummy dataframes (configure *inside* the with block)
            mock_read_csv.side_effect = [
                pd.DataFrame({'IndividualID': ['S1', 'S2', 'S3'], 'HLA-A': ['A*01:01/A*02:01', 'A*01:01/A*01:01', 'A*02:01/A*03:01'], 'Cov1': [1, 2, 1]}), # unphased
                pd.DataFrame({'IndividualID': ['S1', 'S2', 'S3'], 'Haplotype1': ['A*01:01', 'A*01:01', 'A*02:01'], 'Haplotype2': ['A*02:01', 'A*01:01', 'A*03:01']})  # phased
            ]

            # Instantiate and run the runner *inside* the with block
            runner = HLAPhasingRunner(config=self.config)
            runner.run()

            # Check if key methods/classes were initialized or called
            # Restore specific call check for read_csv
            mock_read_csv.assert_has_calls([call(self.dummy_unphased_path), call(self.dummy_phased_path)])
            mock_tts.assert_called_once()
            # Check class instantiation using the mock provided by 'as'
            mock_parser_cls.assert_called_once()
            mock_tokenizer_cls.assert_called_once()
            mock_tokenizer_cls.return_value.build_vocabulary_from_dataframe.assert_called() # Check vocab build was called
            mock_cov_encoder_cls.assert_called_once()
            mock_cov_encoder_cls.return_value.fit_transform.assert_called_once() # Check fit_transform called
            mock_dataset_cls.assert_called() # Called for train and val
            mock_loader_cls.assert_called() # Called for train and val
            mock_model_cls.assert_called_once()
            mock_loss_cls.assert_called_once()
            mock_kl_cls.assert_called_once()
            mock_optimizer_cls.assert_called_once()
            # Check LR scheduler instantiation
            mock_step_lr_cls.assert_called_once_with(
                 mock_optimizer_instance, # Check it got the optimizer instance
                 step_size=self.config.training.lr_scheduler['args']['step_size'],
                 gamma=self.config.training.lr_scheduler['args']['gamma']
             )
            mock_trainer_cls.assert_called_once()
            # Check trainer received the instantiated scheduler
            # call_args is a tuple (args, kwargs)
            _, trainer_kwargs = mock_trainer_cls.call_args
            self.assertIn('lr_scheduler', trainer_kwargs) # Check key exists
            self.assertIs(trainer_kwargs['lr_scheduler'], mock_step_lr_cls.return_value) # Check instance passed
            mock_trainer_cls.return_value.train.assert_called_once() # Check train was called
            mock_model_cls.return_value.eval.assert_called() # Check model set to eval mode
            mock_model_cls.return_value.predict_haplotypes.assert_called() # Check prediction was called
            mock_metrics_cls.assert_called_once()
            mock_metrics_cls.return_value.calculate_metrics.assert_called_once() # Check metrics calculation
        # Use the mock instance returned by the patched class
        mock_reporter_instance_local = mock_reporter_cls.return_value
        mock_reporter_cls.assert_called_once_with(output_dir=self.test_dir)
        mock_reporter_instance_local.log_metric.assert_called() # Check metrics were logged
        mock_reporter_instance_local.generate_report.assert_called() # Check report generated
        mock_reporter_instance_local.plot_training_curves.assert_called_once() # Check plot generated


if __name__ == '__main__':
    unittest.main()
