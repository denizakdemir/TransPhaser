
import unittest
import torch
import torch.nn as nn
from transphaser.model import TransPhaser, ProposalNetwork, ConditionalHaplotypePrior, HWEHaplotypePairPrior, ConstrainedEmissionModel

class TestTransPhaserModel(unittest.TestCase):
    
    def setUp(self):
        # Config for testing
        self.vocab_sizes = {
            "HLA-A": 10,
            "HLA-B": 15,
            "HLA-C": 12
        }
        self.config_dict = {
            "vocab_sizes": self.vocab_sizes,
            "embedding_dim": 8,
            "latent_dim": 4,
            "num_loci": 3,
            "top_k": 4,
            "padding_idx": 0,
            "loci_order": ["HLA-A", "HLA-B", "HLA-C"],
            # Encoder params
            "num_heads": 2,
            "num_layers": 1,
        }
        
    def test_components_init(self):
        """Test that all components initialize correctly."""
        model = TransPhaser(self.config_dict)
        self.assertIsInstance(model.proposal, ProposalNetwork)
        self.assertIsInstance(model.prior, ConditionalHaplotypePrior)
        self.assertIsInstance(model.hwe_prior, HWEHaplotypePairPrior)
        self.assertIsInstance(model.emission, ConstrainedEmissionModel)
        
    def test_proposal_network(self):
        """Test ProposalNetwork output shape."""
        net = ProposalNetwork(self.config_dict)
        batch_size = 5
        # Genotype tokens: (batch, num_loci * 2)
        genotypes = torch.randint(0, 10, (batch_size, 6))
        
        candidates, log_q, phasing_logits = net(genotypes)
        
        # candidates: (batch, k, num_loci, 2)
        self.assertEqual(candidates.shape, (batch_size, 4, 3, 2))
        # log_q: (batch, k)
        self.assertEqual(log_q.shape, (batch_size, 4))
        # phasing_logits: (batch, num_loci)
        self.assertEqual(phasing_logits.shape, (batch_size, 3))
        
    def test_priors(self):
        """Test Prior networks."""
        prior = ConditionalHaplotypePrior(self.config_dict)
        batch_size = 5
        k = 4
        # Candidates: (batch, k, num_loci)
        candidates = torch.randint(0, 10, (batch_size, k, 3))
        
        log_pi = prior(candidates)
        self.assertEqual(log_pi.shape, (batch_size, k))
        
        # Test HWE Prior
        hwe = HWEHaplotypePairPrior(self.config_dict)
        is_homozygous = torch.zeros(batch_size, k, dtype=torch.bool)
        log_pair = hwe.compute_pair_log_prob(log_pi, log_pi, is_homozygous)
        self.assertEqual(log_pair.shape, (batch_size, k))
        
    def test_emission(self):
        """Test Emission model."""
        emit = ConstrainedEmissionModel(self.config_dict)
        batch_size = 5
        # Genotypes: (batch, num_loci, 2)
        genotypes = torch.randint(0, 10, (batch_size, 3, 2))
        
        # Haplotypes must match genotypes to have high prob
        # Let's create compatible haplotypes
        h1 = genotypes[:, :, 0]
        h2 = genotypes[:, :, 1]
        
        log_prob = emit(genotypes, h1, h2)
        # Should be near 0 (log(1))
        self.assertTrue(torch.all(log_prob > -1e-5))
        
        # Incompatible
        h1_bad = torch.zeros_like(h1) + 100 # assume out of range or just diff
        log_prob_bad = emit(genotypes, h1_bad, h2)
        # Should be very low
        self.assertTrue(torch.all(log_prob_bad < -5))

    def test_forward_pass(self):
        """Test data flow through the entire model."""
        model = TransPhaser(self.config_dict)
        batch_size = 2
        
        batch = {
            "genotype_tokens": torch.randint(0, 10, (batch_size, 6)),
        }
        
        output = model(batch)
        
        self.assertIn("responsibilities", output)
        self.assertIn("h1_tokens", output)
        self.assertIn("h2_tokens", output)
        self.assertIn("log_likelihood", output)
        
        self.assertEqual(output["h1_tokens"].shape, (batch_size, 3))
        self.assertEqual(output["h2_tokens"].shape, (batch_size, 3))
        self.assertEqual(output["responsibilities"].shape, (batch_size, 4))
        
    def test_predict(self):
        """Test prediction method."""
        model = TransPhaser(self.config_dict)
        model.eval()
        batch = {
            "genotype_tokens": torch.randint(0, 10, (2, 6)),
        }
        h1, h2 = model.predict_haplotypes(batch)
        self.assertEqual(h1.shape, (2, 3))
        self.assertEqual(h2.shape, (2, 3))
        
if __name__ == '__main__':
    unittest.main()
