import unittest
import os
import tempfile
import json
from typing import List, Optional # Added for CovariateManager test

# Placeholder for Pydantic BaseModel - test will fail if not installed
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Define dummy BaseModel if pydantic not installed, so tests can be defined
    class BaseModel: pass
    class Field: pass
    class ValidationError(Exception): pass


# Placeholder for the classes we are about to create
from transphaser.config import HLAPhasingConfig, DataConfig, ModelConfig, TrainingConfig, LocusMetadataManager, AlleleFrequencyPrior, CovariateManager # Reverted to src.

# Define a simple structure matching what HLAPhasingConfig might expect
# We need these for the test setup, even if HLAPhasingConfig defines its own
class MockSubConfig(BaseModel):
    param1: int = 1
    param2: str = "default"

class MockHLAPhasingConfigStructure(BaseModel):
    # Define some example fields with defaults
    model_name: str = "HLA-Phaser-v1"
    learning_rate: float = 1e-4
    batch_size: int = 32
    sub_config: MockSubConfig = Field(default_factory=MockSubConfig)


@unittest.skipIf(not PYDANTIC_AVAILABLE, "Pydantic library not installed")
class TestHLAPhasingConfig(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for config files
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")

    def tearDown(self):
        # Remove the temporary directory after tests
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_initialization_defaults(self):
        """Test HLAPhasingConfig initialization with default values."""
        config = HLAPhasingConfig()

        # Check default values from the actual HLAPhasingConfig structure
        self.assertEqual(config.model_name, "HLA-Phaser-v1")
        self.assertEqual(config.seed, 42)
        # Check nested defaults
        self.assertIsInstance(config.data, DataConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.training, TrainingConfig)
        self.assertEqual(config.training.learning_rate, 1e-4)
        self.assertEqual(config.training.batch_size, 32)
        self.assertEqual(config.data.locus_columns, ["HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"])


    def test_initialization_from_dict(self):
        """Test HLAPhasingConfig initialization from a dictionary."""
        # Use nested structure for overrides
        config_dict = {
            "model_name": "TestModel",
            "training": {
                "learning_rate": 5e-5
            },
            "data": {
                 "covariate_columns": ["Age"]
            }
        }
        config = HLAPhasingConfig(**config_dict)

        self.assertEqual(config.model_name, "TestModel")
        self.assertEqual(config.training.learning_rate, 5e-5)
        self.assertEqual(config.training.batch_size, 32) # Default
        self.assertEqual(config.data.covariate_columns, ["Age"])
        self.assertEqual(config.data.locus_columns, ["HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"]) # Default

    def test_initialization_from_file(self):
        """Test HLAPhasingConfig initialization from a JSON file."""
        # Use nested structure in the file
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

        config = HLAPhasingConfig(config_path=self.config_path)

        self.assertEqual(config.model_name, "FileModel")
        self.assertEqual(config.training.learning_rate, 1e-4) # Default
        self.assertEqual(config.training.batch_size, 64)
        self.assertEqual(config.data.locus_columns, ["HLA-A", "HLA-B"])
        self.assertEqual(config.data.covariate_columns, []) # Default

    def test_validation_error(self):
        """Test that Pydantic validation raises errors for incorrect types."""
        # Provide invalid data in the correct nested structure
        config_dict_invalid = {
            "training": {
                 "learning_rate": "not_a_float"
            }
        }
        with self.assertRaises(ValidationError):
             HLAPhasingConfig(**config_dict_invalid)

        config_dict_invalid_2 = {
             "data": {
                 "locus_columns": "not_a_list"
             }
        }
        with self.assertRaises(ValidationError):
             HLAPhasingConfig(**config_dict_invalid_2)


class TestLocusMetadataManager(unittest.TestCase):

     def test_initialization(self):
         """Test LocusMetadataManager initialization."""
         # Default initialization
         manager_default = LocusMetadataManager()
         # Check default attributes if any (e.g., empty metadata dict)
         self.assertEqual(manager_default.metadata, {})

         # Initialization with some config (structure TBD)
         locus_config_data = {
             "HLA-A": {"frequency_source": "db1", "resolution": 4},
             "HLA-B": {"frequency_source": "db2"}
         }
         manager_custom = LocusMetadataManager(locus_config=locus_config_data)
         self.assertEqual(manager_custom.metadata, locus_config_data)


class TestAlleleFrequencyPrior(unittest.TestCase):

    def setUp(self):
        # Create dummy prior file for tests that need it
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
        prior_source = self.dummy_prior_path # Use dummy file
        prior_weight = 0.5

        prior = AlleleFrequencyPrior(prior_source=prior_source, prior_weight=prior_weight)

        self.assertEqual(prior.prior_source, prior_source)
        self.assertEqual(prior.prior_weight, prior_weight)
        # Check if data was loaded (basic check based on placeholder)
        self.assertIn("HLA-A", prior.frequency_data)

        # Test default weight
        prior_default_weight = AlleleFrequencyPrior(prior_source=prior_source)
        self.assertEqual(prior_default_weight.prior_weight, 1.0)

        # Test warning for non-existent file
        with self.assertLogs(level='WARNING') as log:
             AlleleFrequencyPrior(prior_source="non_existent_file.csv")
             self.assertTrue(any("Prior source file not found" in msg for msg in log.output))


class TestCovariateManager(unittest.TestCase):

    def test_initialization(self):
        """Test CovariateManager initialization."""
        cat_cols = ['Race', 'Sex']
        num_cols = ['Age', 'BMI']

        # This will fail until the class is defined
        manager = CovariateManager(
            categorical_covariates=cat_cols,
            numerical_covariates=num_cols
        )

        self.assertEqual(manager.categorical_covariates, cat_cols)
        self.assertEqual(manager.numerical_covariates, num_cols)
        # Add checks for default encoding strategies if defined in init

        # Test default initialization
        manager_default = CovariateManager()
        self.assertEqual(manager_default.categorical_covariates, [])
        self.assertEqual(manager_default.numerical_covariates, [])


if __name__ == '__main__':
    unittest.main()
