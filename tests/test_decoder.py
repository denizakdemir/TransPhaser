import unittest
import torch
import torch.nn as nn
import json # Added json import

# Placeholder for the class we are about to create
from transphaser.decoder import HaplotypeDecoderTransformer # Reverted to src.
# Import embedding classes needed for tests
from transphaser.embeddings import AlleleEmbedding, LocusPositionalEmbedding # Reverted to src.
# Import AlleleTokenizer
from transphaser.data_preprocessing import AlleleTokenizer # Reverted to src.

# Minimal placeholder for config
def get_mock_decoder_config():
    # Create and build a mock tokenizer
    tokenizer = AlleleTokenizer()
    mock_alleles_a = ['A*01:01', 'A*02:01', 'A*03:01']
    mock_alleles_b = ['B*07:02', 'B*08:01', 'B*15:01']
    tokenizer.build_vocabulary('HLA-A', mock_alleles_a)
    tokenizer.build_vocabulary('HLA-B', mock_alleles_b)

    return {
        "tokenizer": tokenizer, # Add the tokenizer instance
        "vocab_sizes": {'HLA-A': tokenizer.get_vocab_size('HLA-A'), 'HLA-B': tokenizer.get_vocab_size('HLA-B')}, # Use tokenizer vocab size
        "num_loci": 2,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 128,
        "dropout": 0.1,
        "max_seq_len": 10, # Max length for generation (used by Autoregressive), not internal seq len
        "covariate_dim": 5, # Example input covariate dim
        # Add other necessary config params based on implementation
        "latent_dim": 32 # Added latent_dim for numerical stability test
    }

class TestHaplotypeDecoderTransformer(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = get_mock_decoder_config()
        self.num_loci = self.config["num_loci"]
        self.covariate_dim = self.config["covariate_dim"]
        self.latent_dim = self.config["latent_dim"]
        self.decoder = HaplotypeDecoderTransformer(self.config)

    def test_initialization(self):
        """Test HaplotypeDecoderTransformer initialization."""
        # Check if config attributes are stored (assuming they are)
        self.assertEqual(self.decoder.config, self.config)
        # Add more specific checks based on how __init__ uses the config
        self.assertEqual(self.decoder.embedding_dim, 64)
        self.assertEqual(self.decoder.num_layers, 2)
        # Check for correct embedding types
        self.assertIsInstance(self.decoder.allele_embedding, AlleleEmbedding)
        self.assertIsInstance(self.decoder.positional_embedding, LocusPositionalEmbedding) # Corrected type check
        self.assertIsInstance(self.decoder.transformer_layers, nn.TransformerEncoder)
        # Check output heads ModuleDict
        self.assertIsInstance(self.decoder.output_heads, nn.ModuleDict)
        self.assertEqual(set(self.decoder.output_heads.keys()), set(self.config["vocab_sizes"].keys()))

        # Check if it's an nn.Module
        self.assertIsInstance(self.decoder, nn.Module)

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        batch_size = 4
        # Input seq len should match num_loci for this decoder design
        seq_len = self.num_loci
        max_vocab_size = max(self.config["vocab_sizes"].values())

        # Create input tokens (batch, seq_len) - values need to be valid per locus
        input_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
        locus_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1) # Shape (batch, seq_len)
        for i, locus in enumerate(self.decoder.loci_order):
             vocab_size = self.config["vocab_sizes"][locus]
             # Ensure indices are within the valid range [0, vocab_size - 1]
             input_tokens[:, i] = torch.randint(0, vocab_size, (batch_size,))

        # Create optional masks (all valid for this test)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool) # False means not masked

        # Create optional covariates
        covariates = torch.randn(batch_size, self.covariate_dim) if self.covariate_dim > 0 else None

        # Pass through forward, providing locus_indices
        # Note: The forward pass currently returns combined logits temporarily
        logits = self.decoder(
            input_tokens=input_tokens,
            locus_indices=locus_indices,
            covariates=covariates,
            attention_mask=attention_mask
        )

        # Check output type and keys
        self.assertIsInstance(logits, dict)
        self.assertEqual(set(logits.keys()), set(self.config["vocab_sizes"].keys()))

        # Check shape of logits for each locus
        # The output logits correspond to predictions for the *next* token.
        # The dictionary keys are the loci being predicted.
        for locus, locus_logits in logits.items():
            # The shape should be (batch_size, vocab_size_for_locus)
            expected_locus_shape = (batch_size, self.config["vocab_sizes"][locus])
            self.assertEqual(locus_logits.shape, expected_locus_shape)
            self.assertEqual(locus_logits.dtype, torch.float32) # Assuming float output

    # Add more tests later for masking, covariate handling, actual logic

    def test_numerical_stability(self):
        """Test numerical stability of decoder under various input conditions."""
        # Only enable anomaly detection if we're specifically testing for numerical issues
        test_anomalies = False  # Set to True only when investigating specific numerical issues
        
        # Prepare test inputs
        batch_size = 2
        seq_len = 4
        
        # Get valid token indices for each locus
        input_tokens = []
        locus_indices = []
        for i in range(batch_size):
            sample_tokens = []
            sample_indices = []
            for j in range(seq_len):
                locus_name = self.decoder.loci_order[j % self.num_loci]
                vocab_size = self.config["vocab_sizes"][locus_name]
                sample_tokens.append(torch.randint(0, vocab_size, (1,)).item())
                sample_indices.append(j % self.num_loci)
            input_tokens.append(sample_tokens)
            locus_indices.append(sample_indices)
        
        input_tokens = torch.tensor(input_tokens)
        locus_indices = torch.tensor(locus_indices)
        covariates = torch.randn(batch_size, self.covariate_dim)
        latent_variable = torch.randn(batch_size, self.latent_dim)
        
        # Test with different input scales
        input_scales = [0.1, 1.0, 10.0]
        
        for scale in input_scales:
            scaled_covariates = covariates * scale
            scaled_latent = latent_variable * scale
            
            if test_anomalies:
                with torch.autograd.detect_anomaly():
                    outputs = self.decoder(
                        input_tokens=input_tokens,
                        locus_indices=locus_indices,
                        covariates=scaled_covariates,
                        latent_variable=scaled_latent
                    )
            else:
                outputs = self.decoder(
                    input_tokens=input_tokens,
                    locus_indices=locus_indices,
                    covariates=scaled_covariates,
                    latent_variable=scaled_latent
                )
            
            # Check outputs for each locus
            for locus_name, logits in outputs.items():
                # Verify no NaN or Inf values
                self.assertFalse(torch.isnan(logits).any(), f"NaN values found in logits for locus {locus_name}")
                self.assertFalse(torch.isinf(logits).any(), f"Inf values found in logits for locus {locus_name}")
                
                # Check logits are in a reasonable range
                self.assertTrue((logits.abs() < 1e6).all(), f"Extremely large logits found for locus {locus_name}")


if __name__ == '__main__':
    unittest.main()
