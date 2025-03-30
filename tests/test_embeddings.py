import unittest
import torch
import torch.nn as nn

# Placeholder for the classes we are about to create
from src.embeddings import LocusPositionalEmbedding, AlleleEmbedding

class TestLocusPositionalEmbedding(unittest.TestCase):

    def test_initialization(self):
        """Test LocusPositionalEmbedding initialization."""
        num_loci = 6
        embedding_dim = 128

        # This will fail until the class is defined
        pos_embedding = LocusPositionalEmbedding(
            num_loci=num_loci,
            embedding_dim=embedding_dim
        )

        # Check if it's an nn.Module
        self.assertIsInstance(pos_embedding, nn.Module)

        # Check if the embedding layer is created with the correct dimensions
        # Assuming it uses nn.Embedding internally
        self.assertIsInstance(pos_embedding.locus_embeddings, nn.Embedding)
        self.assertEqual(pos_embedding.locus_embeddings.num_embeddings, num_loci)
        self.assertEqual(pos_embedding.locus_embeddings.embedding_dim, embedding_dim)

    def test_forward_pass(self):
        """Test the forward pass of LocusPositionalEmbedding."""
        num_loci = 6
        embedding_dim = 128
        pos_embedding = LocusPositionalEmbedding(num_loci=num_loci, embedding_dim=embedding_dim)

        # Example input: batch_size=4, seq_len=10 (e.g., representing 10 loci positions)
        # Indices should be within [0, num_loci-1]
        batch_size = 4
        seq_len = num_loci # Assume sequence length matches num_loci for simplicity here
        # Create indices representing loci 0, 1, ..., num_loci-1 for each batch item
        locus_indices = torch.arange(num_loci).unsqueeze(0).repeat(batch_size, 1) # Shape: (batch_size, seq_len)

        # Pass through the forward method
        output_embeddings = pos_embedding(locus_indices)

        # Check output shape
        expected_shape = (batch_size, seq_len, embedding_dim)
        self.assertEqual(output_embeddings.shape, expected_shape)

        # Test with different input shape
        locus_indices_single = torch.tensor([0, 2, 1]) # Shape: (3,)
        output_single = pos_embedding(locus_indices_single)
        expected_shape_single = (3, embedding_dim)
        self.assertEqual(output_single.shape, expected_shape_single)

        # Test invalid index raises error
        invalid_indices = torch.tensor([0, num_loci]) # Contains index >= num_loci
        with self.assertRaises(ValueError):
            pos_embedding(invalid_indices)


class TestAlleleEmbedding(unittest.TestCase):

    def test_initialization(self):
        """Test AlleleEmbedding initialization with per-locus embeddings."""
        vocab_sizes = {'HLA-A': 10, 'HLA-B': 12, 'HLA-C': 8}
        embedding_dim = 64

        # This will fail until the class is defined
        allele_embedding = AlleleEmbedding(
            vocab_sizes=vocab_sizes,
            embedding_dim=embedding_dim
        )

        # Check if it's an nn.Module
        self.assertIsInstance(allele_embedding, nn.Module)

        # Check if ModuleDict is used and contains keys for each locus
        self.assertIsInstance(allele_embedding.locus_embedders, nn.ModuleDict)
        self.assertEqual(set(allele_embedding.locus_embedders.keys()), set(vocab_sizes.keys()))

        # Check each embedding layer within the ModuleDict
        for locus, size in vocab_sizes.items():
            self.assertIn(locus, allele_embedding.locus_embedders)
            locus_embedder = allele_embedding.locus_embedders[locus]
            self.assertIsInstance(locus_embedder, nn.Embedding)
            self.assertEqual(locus_embedder.num_embeddings, size)
            self.assertEqual(locus_embedder.embedding_dim, embedding_dim)

    def test_forward_pass(self):
        """Test the forward pass of AlleleEmbedding."""
        vocab_sizes = {'HLA-A': 10, 'HLA-B': 12}
        embedding_dim = 64
        allele_embedding = AlleleEmbedding(vocab_sizes=vocab_sizes, embedding_dim=embedding_dim)

        # Example input: batch_size=4
        # HLA-A tokens: shape (4, 2) -> 2 alleles per sample
        # HLA-B tokens: shape (4, 2) -> 2 alleles per sample
        batch_size = 4
        tokens_a = torch.randint(0, vocab_sizes['HLA-A'], (batch_size, 2))
        tokens_b = torch.randint(0, vocab_sizes['HLA-B'], (batch_size, 2))

        input_tokens_dict = {'HLA-A': tokens_a, 'HLA-B': tokens_b}

        # Pass through the forward method
        output_embeddings_dict = allele_embedding(input_tokens_dict)

        # Check output type and keys
        self.assertIsInstance(output_embeddings_dict, dict)
        self.assertEqual(set(output_embeddings_dict.keys()), set(vocab_sizes.keys()))

        # Check output shapes for each locus
        expected_shape_a = (batch_size, 2, embedding_dim)
        self.assertEqual(output_embeddings_dict['HLA-A'].shape, expected_shape_a)

        expected_shape_b = (batch_size, 2, embedding_dim)
        self.assertEqual(output_embeddings_dict['HLA-B'].shape, expected_shape_b)

        # Test with unknown locus in input raises error
        invalid_input = {'HLA-A': tokens_a, 'HLA-D': tokens_a} # HLA-D not in vocab_sizes
        with self.assertRaises(KeyError):
            allele_embedding(invalid_input)


if __name__ == '__main__':
    unittest.main()
