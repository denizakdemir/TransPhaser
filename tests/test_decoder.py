import unittest
import torch
import torch.nn as nn

# Placeholder for the class we are about to create
from src.decoder import HaplotypeDecoderTransformer
# Import embedding classes needed for tests
from src.embeddings import AlleleEmbedding, LocusPositionalEmbedding
# Import AlleleTokenizer
from src.data_preprocessing import AlleleTokenizer

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
        "covariate_dim": 5 # Example input covariate dim
        # Add other necessary config params based on implementation
    }

class TestHaplotypeDecoderTransformer(unittest.TestCase):

    def test_initialization(self):
        """Test HaplotypeDecoderTransformer initialization."""
        config = get_mock_decoder_config()

        decoder = HaplotypeDecoderTransformer(config)

        # Check if config attributes are stored (assuming they are)
        self.assertEqual(decoder.config, config)
        # Add more specific checks based on how __init__ uses the config
        self.assertEqual(decoder.embedding_dim, 64)
        self.assertEqual(decoder.num_layers, 2)
        # Check for correct embedding types
        self.assertIsInstance(decoder.allele_embedding, AlleleEmbedding)
        self.assertIsInstance(decoder.positional_embedding, LocusPositionalEmbedding) # Corrected type check
        self.assertIsInstance(decoder.transformer_layers, nn.TransformerEncoder)
        # Check output heads ModuleDict
        self.assertIsInstance(decoder.output_heads, nn.ModuleDict)
        self.assertEqual(set(decoder.output_heads.keys()), set(config["vocab_sizes"].keys()))

        # Check if it's an nn.Module
        self.assertIsInstance(decoder, nn.Module)

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        config = get_mock_decoder_config()
        decoder = HaplotypeDecoderTransformer(config)

        batch_size = 4
        # Input seq len should match num_loci for this decoder design
        seq_len = config["num_loci"]
        max_vocab_size = max(config["vocab_sizes"].values())

        # Create input tokens (batch, seq_len) - values need to be valid per locus
        input_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
        locus_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1) # Shape (batch, seq_len)
        for i, locus in enumerate(decoder.loci_order):
             vocab_size = config["vocab_sizes"][locus]
             # Ensure indices are within the valid range [0, vocab_size - 1]
             input_tokens[:, i] = torch.randint(0, vocab_size, (batch_size,))

        # Create optional masks (all valid for this test)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool) # False means not masked

        # Create optional covariates
        covariate_dim = config.get("covariate_dim", 0)
        covariates = torch.randn(batch_size, covariate_dim) if covariate_dim > 0 else None

        # Pass through forward, providing locus_indices
        # Note: The forward pass currently returns combined logits temporarily
        logits = decoder(
            input_tokens=input_tokens,
            locus_indices=locus_indices,
            covariates=covariates,
            attention_mask=attention_mask
        )

        # Check output type and keys
        self.assertIsInstance(logits, dict)
        self.assertEqual(set(logits.keys()), set(config["vocab_sizes"].keys()))

        # Check shape of logits for each locus
        # The output logits correspond to predictions for the *next* token.
        # The dictionary keys are the loci being predicted.
        for locus, locus_logits in logits.items():
            # The shape should be (batch_size, vocab_size_for_locus)
            expected_locus_shape = (batch_size, config["vocab_sizes"][locus])
            self.assertEqual(locus_logits.shape, expected_locus_shape)
            self.assertEqual(locus_logits.dtype, torch.float32) # Assuming float output

    # Add more tests later for masking, covariate handling, actual logic

    def test_numerical_stability(self):
        """Test that the forward pass does not produce NaN or Inf values."""
        config = get_mock_decoder_config()
        # Add latent_dim to config for the test
        config["latent_dim"] = 32
        decoder = HaplotypeDecoderTransformer(config)

        batch_size = 4
        # Input seq len for decoder training is num_loci + 1 (BOS + k alleles)
        seq_len = config["num_loci"] + 1

        # Create input tokens (batch, seq_len) - including BOS
        input_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_tokens[:, 0] = config["tokenizer"].special_tokens.get("BOS", 2) # Set BOS token
        for i, locus in enumerate(decoder.loci_order):
             vocab_size = config["vocab_sizes"][locus]
             # Ensure indices are within the valid range [0, vocab_size - 1] for alleles
             input_tokens[:, i+1] = torch.randint(0, vocab_size, (batch_size,))

        # Create optional covariates
        covariate_dim = config.get("covariate_dim", 0)
        covariates = torch.randn(batch_size, covariate_dim) if covariate_dim > 0 else None

        # Create latent variable
        latent_dim = config.get("latent_dim", 0)
        latent_variable = torch.randn(batch_size, latent_dim) if latent_dim > 0 else None

        # --- Test in train() mode ---
        decoder.train() # Ensure dropout is active if it matters
        logits_dict_train = decoder(
            input_tokens=input_tokens,
            locus_indices=None, # Not used by current decoder
            covariates=covariates,
            latent_variable=latent_variable,
            attention_mask=None # No padding mask for this test
        )
        # Check for NaN/Inf in output logits (train mode)
        for locus, locus_logits in logits_dict_train.items():
            self.assertFalse(torch.isnan(locus_logits).any(), f"NaN detected in train() logits for locus {locus}")
            self.assertFalse(torch.isinf(locus_logits).any(), f"Inf detected in train() logits for locus {locus}")

        # --- Test in eval() mode ---
        decoder.eval() # Set to evaluation mode
        with torch.autograd.detect_anomaly(): # Enable anomaly detection
            # Removed torch.no_grad() to allow anomaly detection traceback
            # Run eval mode *without* conditioning first
            logits_dict_eval_no_cond = decoder(
                input_tokens=input_tokens,
                locus_indices=None,
                covariates=None, # Test without covariates
                latent_variable=None, # Test without latent variable
                attention_mask=None
             )
        # Check for NaN/Inf without conditioning
        for locus, locus_logits in logits_dict_eval_no_cond.items():
             self.assertFalse(torch.isnan(locus_logits).any(), f"NaN detected in eval() logits (no cond) for locus {locus}")
             self.assertFalse(torch.isinf(locus_logits).any(), f"Inf detected in eval() logits (no cond) for locus {locus}")

        # Now test again *with* conditioning (original test)
        # Removed torch.no_grad() here too
        logits_dict_eval_with_cond = decoder(
            input_tokens=input_tokens,
            locus_indices=None,
                 covariates=covariates, # Test with covariates
                 latent_variable=latent_variable, # Test with latent variable
                 attention_mask=None
             )
        # Check for NaN/Inf with conditioning
        for locus, locus_logits in logits_dict_eval_with_cond.items():
            self.assertFalse(torch.isnan(locus_logits).any(), f"NaN detected in eval() logits (with cond) for locus {locus}")
            self.assertFalse(torch.isinf(locus_logits).any(), f"Inf detected in eval() logits (with cond) for locus {locus}")


if __name__ == '__main__':
    unittest.main()
