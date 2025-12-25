import unittest
import os
import tempfile
import json
from typing import List, Optional

try:
    from pydantic import BaseModel, Field, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    class BaseModel: pass
    class Field: pass
    class ValidationError(Exception): pass

# Import TransPhaser classes
from transphaser.config import TransPhaserConfig, DataConfig, ModelConfig, TrainingConfig, LocusMetadataManager, AlleleFrequencyPrior, CovariateManager

@unittest.skipIf(not PYDANTIC_AVAILABLE, "Pydantic library not installed")
class TestTransPhaserConfig(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for config files
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")

    def tearDown(self):
        # Remove the temporary directory after tests
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization_defaults(self):
        """Test TransPhaserConfig initialization with default values."""
        config = TransPhaserConfig()

        self.assertEqual(config.model_name, "TransPhaser-v1")
        self.assertEqual(config.seed, 42)
        # Check nested defaults
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertEqual(config.training.learning_rate, 1e-4)
        self.assertEqual(config.training.batch_size, 32)

    def test_initialization_from_dict(self):
        """Test TransPhaserConfig initialization from a dictionary."""
        config_dict = {
            "model_name": "TestModel",
            "training": {
                "learning_rate": 5e-5
            },
            "data": {
                 "covariate_columns": ["Age"]
            }
        }
        config = TransPhaserConfig(**config_dict)

        self.assertEqual(config.model_name, "TestModel")
        self.assertEqual(config.training.learning_rate, 5e-5)
        self.assertEqual(config.training.batch_size, 32) # Default
        self.assertEqual(config.data.covariate_columns, ["Age"])

    def test_initialization_from_file(self):
        """Test TransPhaserConfig initialization from a JSON file."""
        config_dict = {
            "model_name": "FileModel",
            "training": {
                "batch_size": 64
            },
             "data": {
                 "locus_columns": ["HLA-A", "HLA-B"]
             }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f)

        config = TransPhaserConfig(config_path=self.config_path)

        self.assertEqual(config.model_name, "FileModel")
        self.assertEqual(config.training.learning_rate, 1e-4) # Default
        self.assertEqual(config.training.batch_size, 64)
        self.assertEqual(config.data.locus_columns, ["HLA-A", "HLA-B"])

    def test_validation_error(self):
        """Test that Pydantic validation raises errors for incorrect types."""
        config_dict_invalid = {
            "training": {
                 "learning_rate": "not_a_float"
            }
        }
        with self.assertRaises(ValidationError):
             TransPhaserConfig(**config_dict_invalid)

class TestLocusMetadataManager(unittest.TestCase):

     def test_initialization(self):
         """Test LocusMetadataManager initialization."""
         manager_default = LocusMetadataManager()
         self.assertEqual(manager_default.metadata, {})

         locus_config_data = {
             "HLA-A": {"frequency_source": "db1", "resolution": 4},
             "HLA-B": {"frequency_source": "db2"}
         }
         manager_custom = LocusMetadataManager(locus_config=locus_config_data)
         self.assertEqual(manager_custom.metadata, locus_config_data)

class TestAlleleFrequencyPrior(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.dummy_prior_path = os.path.join(self.temp_dir, "dummy_freqs.csv")
        with open(self.dummy_prior_path, "w") as f:
            f.write("Locus,Allele,Frequency\n")
            f.write("HLA-A,A*01:01,0.25\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test AlleleFrequencyPrior initialization."""
        prior = AlleleFrequencyPrior(prior_source=self.dummy_prior_path, prior_weight=0.5)
        self.assertEqual(prior.prior_source, self.dummy_prior_path)
        self.assertEqual(prior.prior_weight, 0.5)
        self.assertIn("HLA-A", prior.frequency_data)

class TestCovariateManager(unittest.TestCase):

    def test_initialization(self):
        """Test CovariateManager initialization."""
        cat_cols = ['Race', 'Sex']
        num_cols = ['Age', 'BMI']
        manager = CovariateManager(
            categorical_covariates=cat_cols,
            numerical_covariates=num_cols
        )
        self.assertEqual(manager.categorical_covariates, cat_cols)
        self.assertEqual(manager.numerical_covariates, num_cols)

if __name__ == '__main__':
    unittest.main()
