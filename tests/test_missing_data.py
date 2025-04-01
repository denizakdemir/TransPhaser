import unittest
import pandas as pd
import torch # Added torch import
import torch.nn as nn # Added nn import

# Placeholder for AlleleTokenizer (needed for mock)
from transphaser.data_preprocessing import AlleleTokenizer # Reverted to src.
# Placeholder for the classes we are about to create
from transphaser.missing_data import MissingDataDetector, MissingDataMarginalizer, AlleleImputer # Reverted to src.

# --- Mocks ---
# Use the actual AlleleTokenizer for initialization tests
# class MockAlleleTokenizer:
#     def __init__(self):
#         # Define some special tokens, including UNK which might represent missing
#         self.special_tokens = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
#         # Assume UNK token string is 'UNK'
#         self.unk_token_string = 'UNK'
#         print("MockAlleleTokenizer Initialized for missing data tests")
#
#     def tokenize(self, locus, allele): # noqa: E704 -- flake8 doesn't like this but it's commented
#         # Simple mock: return UNK ID if allele is None or specific missing markers # noqa: E704
#         if allele is None or allele == self.unk_token_string or allele == '': # noqa: E704
#             return self.special_tokens['UNK'] # noqa: E704
#         # Otherwise return a dummy ID (not UNK) # noqa: E704
#         return 10 # Dummy ID for non-missing # noqa: E704
# # noqa: E704
#     # Add other methods if needed by MissingDataDetector init or methods # noqa: E704

class MockModel(nn.Module):
    """Mock model for Marginalizer/Imputer tests."""
    def __init__(self, vocab_size=10, unk_token_id=1):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0)) # Dummy parameter
        self.vocab_size = vocab_size
        self.unk_token_id = unk_token_id
        # print("MockModel Initialized for missing data tests") # Reduce noise

    def calculate_log_likelihood(self, batch):
        """
        Mock likelihood calculation. Returns a dummy value based on input.
        Assumes lower likelihood if UNK token is present (for basic testing).
        """
        # Example: Return a fixed negative value per sample, penalize if UNK present
        genotypes = batch.get('genotypes_tokens') # Assume this key exists
        if genotypes is None:
            return torch.tensor([-10.0] * batch.get('batch_size', 1)) # Default if no tokens

        batch_size = genotypes.shape[0]
        likelihoods = []
        for i in range(batch_size):
            # Simple mock: base likelihood - penalty for UNK
            penalty = torch.sum(genotypes[i] == self.unk_token_id).float() * 5.0 # Penalize 5 per UNK
            likelihoods.append(-10.0 - penalty)
        return torch.tensor(likelihoods)

    def sample_imputation(self, batch_with_missing):
        """
        Mock imputation sampling. Replaces UNK tokens with random valid tokens.
        """
        imputed_batch = batch_with_missing.copy() # Shallow copy is okay if modifying tensors in-place
        tokens = imputed_batch['genotypes_tokens'].clone() # Clone tensor to modify
        mask = (tokens == self.unk_token_id)
        num_missing = mask.sum()
        if num_missing > 0:
            # Sample random valid tokens (not UNK, not PAD if exists)
            # Assuming valid tokens are 2 to vocab_size-1
            valid_tokens = torch.arange(2, self.vocab_size) # Example valid range
            if len(valid_tokens) > 0:
                 random_imputations = valid_tokens[torch.randint(len(valid_tokens), (num_missing,))]
                 tokens[mask] = random_imputations
            else:
                 # Handle case where no valid tokens exist (edge case)
                 logging.warning("MockModel: No valid tokens to sample for imputation.")
                 # Replace UNK with a default non-UNK token, e.g., 2, if possible
                 if self.vocab_size > 2:
                     tokens[mask] = 2
                 # Otherwise, leave as UNK or handle as error

        imputed_batch['genotypes_tokens'] = tokens
        return imputed_batch

    def predict_missing_probabilities(self, batch_with_missing):
        """
        Mock prediction of probabilities for missing tokens.
        Returns dummy probabilities skewed so token '2' is the mode.
        """
        tokens = batch_with_missing['genotypes_tokens']
        mask = (tokens == self.unk_token_id)
        batch_size, seq_len = tokens.shape
        # Shape: (batch_size, seq_len, vocab_size)
        probs = torch.zeros(batch_size, seq_len, self.vocab_size, device=tokens.device) # Ensure same device

        if self.vocab_size > 2:
            # Assign high probability to token 2 to make it the mode deterministically
            high_prob = 0.9
            probs[:, :, 2] = high_prob
            # Distribute remaining probability among other valid tokens (excluding PAD, UNK)
            remaining_prob = (1.0 - high_prob)
            other_valid_indices = [i for i in range(self.vocab_size) if i != 0 and i != self.unk_token_id and i != 2]
            num_other_valid_tokens = len(other_valid_indices)

            if num_other_valid_tokens > 0:
                prob_per_other = remaining_prob / num_other_valid_tokens
                for i in other_valid_indices:
                    probs[:, :, i] = prob_per_other
            else:
                # If only PAD/UNK/token 2 exist, assign remaining prob to PAD (token 0)
                 if self.vocab_size == 3 and 0 not in other_valid_indices: # Check if PAD is available
                     probs[:, :, 0] = remaining_prob

        elif self.vocab_size == 2: # Only PAD, UNK
             # Cannot make token 2 the mode. Make PAD (0) the mode instead.
             probs[:, :, 0] = 0.9
             probs[:, :, 1] = 0.1 # UNK gets low prob
        elif self.vocab_size == 1: # Only PAD
             probs[:, :, 0] = 1.0
        # else vocab_size < 1 is invalid

        # Ensure probabilities sum to 1 (handle potential float precision issues)
        probs = torch.clamp(probs, min=0) # Ensure no negative probabilities due to precision
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # Handle cases where sum is zero (e.g., vocab_size=0 or all probs became zero)
        probs[torch.isnan(probs)] = 0 # Replace NaN with 0
        # If a row sums to zero, distribute uniformly (avoid NaN in argmax)
        zero_sum_mask = probs.sum(dim=-1) == 0
        if torch.any(zero_sum_mask):
             # Avoid division by zero if vocab_size is 0
             if self.vocab_size > 0:
                 uniform_prob = 1.0 / self.vocab_size
                 probs[zero_sum_mask] = uniform_prob
             else:
                 # If vocab_size is 0, probs should remain all zeros
                 pass


        return probs

    def sample_from_probabilities(self, probabilities):
        """Mock sampling from predicted probabilities."""
        # probabilities shape: (batch_size, seq_len, vocab_size)
        # Sample one token per position based on the probabilities
        batch_size, seq_len, vocab_size = probabilities.shape
        # Reshape for multinomial: (batch_size * seq_len, vocab_size)
        probs_flat = probabilities.view(-1, vocab_size)
        samples_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        # Reshape back: (batch_size, seq_len)
        return samples_flat.view(batch_size, seq_len)

    def get_mode_from_probabilities(self, probabilities):
        """Mock getting the mode from predicted probabilities."""
        # probabilities shape: (batch_size, seq_len, vocab_size)
        # Get the token with the highest probability for each position
        return torch.argmax(probabilities, dim=-1)

# --- Test Classes ---
class TestMissingDataDetector(unittest.TestCase): # noqa: E742 -- Class Name okay

    def test_initialization(self): # noqa: E742 -- Method Name okay
        """Test MissingDataDetector initialization.""" # noqa: E742
        # Use the actual tokenizer # noqa: E742
        real_tokenizer = AlleleTokenizer() # noqa: E742

        detector = MissingDataDetector(tokenizer=real_tokenizer) # noqa: E742

        # Check attributes # noqa: E742
        self.assertIs(detector.tokenizer, real_tokenizer) # noqa: E742
        # Check if UNK token ID/string is stored correctly if needed # noqa: E742
        self.assertEqual(detector.unk_token_id, real_tokenizer.special_tokens['UNK']) # noqa: E742


class TestMissingDataMarginalizer(unittest.TestCase): # noqa: E742

    def test_initialization(self): # noqa: E742
        """Test MissingDataMarginalizer initialization.""" # noqa: E742
        mock_model = MockModel() # noqa: E742
        sampling_iterations = 20 # noqa: E742

        marginalizer = MissingDataMarginalizer( # noqa: E742
            model=mock_model, # noqa: E742
            sampling_iterations=sampling_iterations # noqa: E742
        ) # noqa: E742

        # Check attributes # noqa: E742
        self.assertIs(marginalizer.model, mock_model) # noqa: E742
        self.assertEqual(marginalizer.sampling_iterations, sampling_iterations) # noqa: E742

        # Test default iterations # noqa: E742
        marginalizer_default = MissingDataMarginalizer(model=mock_model) # noqa: E742
        self.assertEqual(marginalizer_default.sampling_iterations, 10) # Default from description # noqa: E742

    def test_calculate_marginal_likelihood(self):
        """Test marginal likelihood calculation."""
        unk_token_id = 1
        vocab_size = 10
        mock_model = MockModel(vocab_size=vocab_size, unk_token_id=unk_token_id)
        sampling_iterations = 5 # Use fewer iterations for test speed
        marginalizer = MissingDataMarginalizer(
            model=mock_model,
            sampling_iterations=sampling_iterations
        )

        # Create a batch with missing data (UNK tokens)
        # Shape: (batch_size, num_loci * 2) assuming flattened genotypes
        batch_size = 2
        num_features = 4 # e.g., 2 loci * 2 alleles
        genotypes_tokens = torch.tensor([
            [2, 3, 4, unk_token_id], # Sample 1 with one missing
            [5, unk_token_id, 6, unk_token_id]  # Sample 2 with two missing
        ], dtype=torch.long)

        batch_with_missing = {
            'genotypes_tokens': genotypes_tokens,
            'batch_size': batch_size
            # Add other necessary batch elements if the implementation requires them
        }

        # --- Act ---
        marginal_log_likelihood = marginalizer.marginalize_likelihood(batch_with_missing)

        # --- Assert ---
        # Check return type and shape
        self.assertIsInstance(marginal_log_likelihood, torch.Tensor)
        # Expecting a scalar average likelihood or likelihood per sample?
        # Let's assume per sample for now, matching MockModel output shape
        self.assertEqual(marginal_log_likelihood.shape, (batch_size,))

        # Check values are plausible (negative log-likelihoods)
        self.assertTrue(torch.all(marginal_log_likelihood <= 0))

        # Optional: Check if likelihood is generally lower than a complete batch
        # (This depends heavily on the mock model's behavior)
        genotypes_complete = torch.tensor([
            [2, 3, 4, 5], # Sample 1 complete
            [5, 2, 6, 3]  # Sample 2 complete
        ], dtype=torch.long)
        batch_complete = {'genotypes_tokens': genotypes_complete, 'batch_size': batch_size}
        # Need a way to get direct likelihood from the model for comparison
        direct_complete_ll = mock_model.calculate_log_likelihood(batch_complete)
        # The marginal LL might be higher or lower depending on the imputed values
        # and the mock model's penalty. A simple check might be difficult.
        # print(f"Marginal LL: {marginal_log_likelihood}") # For debugging # noqa: E704
        # print(f"Complete LL: {direct_complete_ll}")     # For debugging # noqa: E704

    def test_impute(self):
        """Test imputation using the marginalizer."""
        unk_token_id = 1
        vocab_size = 10
        mock_model = MockModel(vocab_size=vocab_size, unk_token_id=unk_token_id)
        marginalizer = MissingDataMarginalizer(model=mock_model) # Use default iterations

        # Create a batch with missing data
        batch_size = 2
        num_features = 4
        genotypes_tokens = torch.tensor([
            [2, 3, 4, unk_token_id], # Sample 1 with one missing
            [5, unk_token_id, 6, unk_token_id]  # Sample 2 with two missing
        ], dtype=torch.long)
        batch_with_missing = {'genotypes_tokens': genotypes_tokens, 'batch_size': batch_size}

        # --- Act ---
        imputed_batch = marginalizer.impute(batch_with_missing)

        # --- Assert ---
        # Check return type is a dict
        self.assertIsInstance(imputed_batch, dict)
        # Check if 'genotypes_tokens' key exists
        self.assertIn('genotypes_tokens', imputed_batch)
        imputed_tokens = imputed_batch['genotypes_tokens']
        # Check shape is preserved
        self.assertEqual(imputed_tokens.shape, genotypes_tokens.shape)
        # Check dtype is preserved
        self.assertEqual(imputed_tokens.dtype, genotypes_tokens.dtype)
        # Check that UNK tokens have been replaced
        self.assertFalse(torch.any(imputed_tokens == unk_token_id))
        # Check that original non-UNK tokens are preserved
        original_mask = (genotypes_tokens != unk_token_id)
        self.assertTrue(torch.equal(imputed_tokens[original_mask], genotypes_tokens[original_mask]))
        # Check that imputed values are within the valid range (based on MockModel)
        imputed_mask = (genotypes_tokens == unk_token_id)
        self.assertTrue(torch.all(imputed_tokens[imputed_mask] >= 2))
        self.assertTrue(torch.all(imputed_tokens[imputed_mask] < vocab_size))

class TestAlleleImputer(unittest.TestCase): # noqa: E742

    def test_initialization(self): # noqa: E742
        """Test AlleleImputer initialization.""" # noqa: E742
        mock_model = MockModel() # noqa: E742

        # Default strategy # noqa: E742
        imputer_default = AlleleImputer(model=mock_model) # noqa: E742
        self.assertIs(imputer_default.model, mock_model) # noqa: E742
        self.assertEqual(imputer_default.imputation_strategy, 'sampling') # Default from description # noqa: E742

        # Custom strategy # noqa: E742
        imputer_custom = AlleleImputer(model=mock_model, imputation_strategy='mode') # noqa: E742
        self.assertIs(imputer_custom.model, mock_model) # noqa: E742
        self.assertEqual(imputer_custom.imputation_strategy, 'mode') # noqa: E742

        # Invalid strategy # noqa: E742
        with self.assertRaises(ValueError): # noqa: E742
            AlleleImputer(model=mock_model, imputation_strategy='invalid_strategy') # noqa: E742

    def test_impute_alleles_sampling(self):
        """Test allele imputation using 'sampling' strategy."""
        unk_token_id = 1
        vocab_size = 10
        mock_model = MockModel(vocab_size=vocab_size, unk_token_id=unk_token_id)
        imputer = AlleleImputer(model=mock_model, imputation_strategy='sampling')

        # Create a batch with missing data
        batch_size = 2
        num_features = 4
        genotypes_tokens = torch.tensor([
            [2, 3, 4, unk_token_id], # Sample 1 with one missing
            [5, unk_token_id, 6, unk_token_id]  # Sample 2 with two missing
        ], dtype=torch.long)
        batch_with_missing = {'genotypes_tokens': genotypes_tokens, 'batch_size': batch_size}

        # --- Act ---
        imputed_batch = imputer.impute_alleles(batch_with_missing)

        # --- Assert ---
        self.assertIsInstance(imputed_batch, dict)
        self.assertIn('genotypes_tokens', imputed_batch)
        imputed_tokens = imputed_batch['genotypes_tokens']
        self.assertEqual(imputed_tokens.shape, genotypes_tokens.shape)
        self.assertEqual(imputed_tokens.dtype, genotypes_tokens.dtype)
        # Check UNK tokens are replaced
        self.assertFalse(torch.any(imputed_tokens == unk_token_id))
        # Check original tokens are preserved
        original_mask = (genotypes_tokens != unk_token_id)
        self.assertTrue(torch.equal(imputed_tokens[original_mask], genotypes_tokens[original_mask]))
        # Check imputed values are valid tokens
        imputed_mask = (genotypes_tokens == unk_token_id)
        self.assertTrue(torch.all(imputed_tokens[imputed_mask] >= 2)) # Assuming 0=PAD, 1=UNK
        self.assertTrue(torch.all(imputed_tokens[imputed_mask] < vocab_size))

    def test_impute_alleles_mode(self):
        """Test allele imputation using 'mode' strategy."""
        unk_token_id = 1
        vocab_size = 10
        mock_model = MockModel(vocab_size=vocab_size, unk_token_id=unk_token_id)
        imputer = AlleleImputer(model=mock_model, imputation_strategy='mode')

        # Create a batch with missing data
        batch_size = 2
        num_features = 4
        genotypes_tokens = torch.tensor([
            [2, 3, 4, unk_token_id], # Sample 1 with one missing
            [5, unk_token_id, 6, unk_token_id]  # Sample 2 with two missing
        ], dtype=torch.long)
        batch_with_missing = {'genotypes_tokens': genotypes_tokens, 'batch_size': batch_size}

        # --- Act ---
        imputed_batch = imputer.impute_alleles(batch_with_missing)

        # --- Assert ---
        self.assertIsInstance(imputed_batch, dict)
        self.assertIn('genotypes_tokens', imputed_batch)
        imputed_tokens = imputed_batch['genotypes_tokens']
        self.assertEqual(imputed_tokens.shape, genotypes_tokens.shape)
        self.assertEqual(imputed_tokens.dtype, genotypes_tokens.dtype)
        # Check UNK tokens are replaced
        self.assertFalse(torch.any(imputed_tokens == unk_token_id))
        # Check original tokens are preserved
        original_mask = (genotypes_tokens != unk_token_id)
        self.assertTrue(torch.equal(imputed_tokens[original_mask], genotypes_tokens[original_mask]))
        # Check imputed values are valid tokens (specifically the mode '2' from mock)
        imputed_mask = (genotypes_tokens == unk_token_id)
        # Based on the enhanced MockModel, the mode should be token '2'
        if vocab_size > 2:
             self.assertTrue(torch.all(imputed_tokens[imputed_mask] == 2))
        else:
             # Handle edge case where '2' isn't a valid token
             self.assertTrue(torch.all(imputed_tokens[imputed_mask] != unk_token_id))

if __name__ == '__main__': # noqa: E742
    unittest.main() # noqa: E742
