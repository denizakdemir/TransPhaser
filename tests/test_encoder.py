import unittest
import torch
import torch.nn as nn

# Placeholder for the class we are about to create
from transphaser.encoder import GenotypeEncoderTransformer # Reverted to src.
# Import embedding classes needed for tests
from transphaser.embeddings import AlleleEmbedding # Reverted to src.

# Minimal placeholder for config (can reuse decoder's for now if similar)
def get_mock_encoder_config():
    return {
        "vocab_sizes": {'HLA-A': 10, 'HLA-B': 12}, # Example
        "num_loci": 2,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "ff_dim": 128,
        "dropout": 0.1,
        "max_seq_len": 10, # Example - Note: Encoder uses num_loci * 2 internally
        "covariate_dim": 5, # Example input covariate dim
        "latent_dim": 64 # Example latent dim
        # Add other necessary config params based on implementation
    }

class TestGenotypeEncoderTransformer(unittest.TestCase):

    def test_initialization(self):
        """Test GenotypeEncoderTransformer initialization."""
        config = get_mock_encoder_config()

        encoder = GenotypeEncoderTransformer(config)

        # Check if config attributes are stored (assuming they are)
        self.assertEqual(encoder.config, config)
        # Add more specific checks based on how __init__ uses the config
        self.assertEqual(encoder.embedding_dim, 64)
        self.assertEqual(encoder.num_layers, 2)
        self.assertIsInstance(encoder.allele_embedding, AlleleEmbedding) # Corrected type check
        self.assertIsInstance(encoder.positional_embedding, nn.Embedding)
        self.assertIsInstance(encoder.type_embedding, nn.Embedding) # Check type embedding
        self.assertIsInstance(encoder.transformer_encoder, nn.TransformerEncoder)
        self.assertIsInstance(encoder.output_head, nn.Linear)

        # Check if it's an nn.Module
        self.assertIsInstance(encoder, nn.Module)

    def test_forward_pass_shape(self):
        """Test the output shape of the forward pass."""
        config = get_mock_encoder_config()
        encoder = GenotypeEncoderTransformer(config)

        batch_size = 4
        # Input seq len = num_loci * 2
        seq_len = config["num_loci"] * 2
        # Generate input tokens with valid indices for each locus
        input_tokens_list = []
        locus_names = list(config["vocab_sizes"].keys()) # Get locus names in order
        for i, locus in enumerate(locus_names):
            vocab_size = config["vocab_sizes"][locus]
            # Generate tokens for allele 1 and allele 2 of this locus
            tokens_locus_1 = torch.randint(0, vocab_size, (batch_size, 1))
            tokens_locus_2 = torch.randint(0, vocab_size, (batch_size, 1))
            input_tokens_list.extend([tokens_locus_1, tokens_locus_2])

        # Concatenate tokens for all loci
        input_tokens = torch.cat(input_tokens_list, dim=1)
        # Ensure the final shape matches seq_len
        assert input_tokens.shape == (batch_size, seq_len), f"Shape mismatch: {input_tokens.shape} vs {(batch_size, seq_len)}"

        # Create optional masks (all valid for this test)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool) # False means not masked

        # Create optional covariates
        covariate_dim = config["covariate_dim"]
        covariates = torch.randn(batch_size, covariate_dim) if covariate_dim > 0 else None

        # Pass through forward
        posterior_params = encoder(input_tokens, covariates=covariates, attention_mask=attention_mask)

        # Check output shape (batch_size, latent_dim * 2 for mean+logvar)
        expected_latent_dim = config.get("latent_dim", 64) # Get latent dim used in encoder
        expected_shape = (batch_size, expected_latent_dim * 2)
        self.assertEqual(posterior_params.shape, expected_shape)

    # Add more tests later for masking, actual logic

    def test_numerical_stability(self):
        """Test that the forward pass does not produce NaN or Inf values."""
        config = get_mock_encoder_config()
        encoder = GenotypeEncoderTransformer(config)
        encoder.train() # Ensure dropout is active if it matters

        batch_size = 4
        seq_len = config["num_loci"] * 2

        # Generate input tokens
        input_tokens_list = []
        locus_names = list(config["vocab_sizes"].keys())
        for i, locus in enumerate(locus_names):
            vocab_size = config["vocab_sizes"][locus]
            tokens_locus_1 = torch.randint(0, vocab_size, (batch_size, 1))
            tokens_locus_2 = torch.randint(0, vocab_size, (batch_size, 1))
            input_tokens_list.extend([tokens_locus_1, tokens_locus_2])
        input_tokens = torch.cat(input_tokens_list, dim=1)

        # Create optional covariates
        covariate_dim = config["covariate_dim"]
        covariates = torch.randn(batch_size, covariate_dim) if covariate_dim > 0 else None

        # Pass through forward
        posterior_params = encoder(input_tokens, covariates=covariates, attention_mask=None)

        # Check for NaN/Inf in output
        self.assertFalse(torch.isnan(posterior_params).any(), "NaN detected in encoder output")
        self.assertFalse(torch.isinf(posterior_params).any(), "Inf detected in encoder output")


if __name__ == '__main__':
    unittest.main()
