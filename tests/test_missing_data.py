import unittest
import pandas as pd
import torch # Added torch import
import torch.nn as nn # Added nn import

# Placeholder for AlleleTokenizer (needed for mock)
from src.data_preprocessing import AlleleTokenizer
# Placeholder for the classes we are about to create
from src.missing_data import MissingDataDetector, MissingDataMarginalizer, AlleleImputer

# --- Mocks ---
class MockAlleleTokenizer:
    def __init__(self):
        # Define some special tokens, including UNK which might represent missing
        self.special_tokens = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
        # Assume UNK token string is 'UNK'
        self.unk_token_string = 'UNK'
        print("MockAlleleTokenizer Initialized for missing data tests")

    def tokenize(self, locus, allele):
        # Simple mock: return UNK ID if allele is None or specific missing markers
        if allele is None or allele == self.unk_token_string or allele == '':
            return self.special_tokens['UNK']
        # Otherwise return a dummy ID (not UNK)
        return 10 # Dummy ID for non-missing

    # Add other methods if needed by MissingDataDetector init or methods

class MockModel(nn.Module):
    """Minimal mock model needed for Marginalizer/Imputer init."""
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0)) # Dummy parameter
        print("MockModel Initialized for missing data tests")
    # Add forward if needed by marginalizer/imputer methods

# --- Test Classes ---
class TestMissingDataDetector(unittest.TestCase):

    def test_initialization(self):
        """Test MissingDataDetector initialization."""
        mock_tokenizer = MockAlleleTokenizer()

        detector = MissingDataDetector(tokenizer=mock_tokenizer)

        # Check attributes
        self.assertIs(detector.tokenizer, mock_tokenizer)
        # Check if UNK token ID/string is stored correctly if needed
        self.assertEqual(detector.unk_token_id, mock_tokenizer.special_tokens['UNK'])


class TestMissingDataMarginalizer(unittest.TestCase):

    def test_initialization(self):
        """Test MissingDataMarginalizer initialization."""
        mock_model = MockModel()
        sampling_iterations = 20

        marginalizer = MissingDataMarginalizer(
            model=mock_model,
            sampling_iterations=sampling_iterations
        )

        # Check attributes
        self.assertIs(marginalizer.model, mock_model)
        self.assertEqual(marginalizer.sampling_iterations, sampling_iterations)

        # Test default iterations
        marginalizer_default = MissingDataMarginalizer(model=mock_model)
        self.assertEqual(marginalizer_default.sampling_iterations, 10) # Default from description


class TestAlleleImputer(unittest.TestCase):

    def test_initialization(self):
        """Test AlleleImputer initialization."""
        mock_model = MockModel()

        # Default strategy
        imputer_default = AlleleImputer(model=mock_model)
        self.assertIs(imputer_default.model, mock_model)
        self.assertEqual(imputer_default.imputation_strategy, 'sampling') # Default from description

        # Custom strategy
        imputer_custom = AlleleImputer(model=mock_model, imputation_strategy='mode')
        self.assertIs(imputer_custom.model, mock_model)
        self.assertEqual(imputer_custom.imputation_strategy, 'mode')

        # Invalid strategy
        with self.assertRaises(ValueError):
            AlleleImputer(model=mock_model, imputation_strategy='invalid_strategy')


if __name__ == '__main__':
    unittest.main()
