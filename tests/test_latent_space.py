import unittest
import random # For sampling test

# Placeholder for HaplotypeCompatibilityChecker (already exists)
from transphaser.compatibility import HaplotypeCompatibilityChecker, HLACompatibilityRules # Reverted to src.
# Placeholder for the class we are about to create
from transphaser.latent_space import HaplotypeSpaceExplorer # Reverted to src.

class TestHaplotypeSpaceExplorer(unittest.TestCase): # noqa: E742 -- Class Name okay

    def test_initialization(self):
        """Test HaplotypeSpaceExplorer initialization."""
        mock_checker = HaplotypeCompatibilityChecker() # Use the real one as a mock for init

        # Default initialization
        explorer_default = HaplotypeSpaceExplorer(compatibility_checker=mock_checker)
        self.assertEqual(explorer_default.sampling_temperature, 1.0)
        self.assertIs(explorer_default.compatibility_checker, mock_checker)

        # Initialization with specific temperature
        explorer_temp = HaplotypeSpaceExplorer(compatibility_checker=mock_checker, sampling_temperature=0.7)
        self.assertEqual(explorer_temp.sampling_temperature, 0.7) # noqa: E742
        self.assertIs(explorer_temp.compatibility_checker, mock_checker) # noqa: E742

    def test_sample_exhaustive(self):
        """Test haplotype sampling with the 'exhaustive' strategy."""
        checker = HaplotypeCompatibilityChecker()
        explorer = HaplotypeSpaceExplorer(compatibility_checker=checker)
        rules = HLACompatibilityRules() # To get expected pairs

        # --- Heterozygous Case ---
        genotype_het = ['A*01:01', 'A*02:01']
        expected_pairs_het = set(rules.get_valid_haplotype_pairs(genotype_het))
        self.assertEqual(len(expected_pairs_het), 2) # Should be ('A*01:01', 'A*02:01') and ('A*02:01', 'A*01:01')

        # Test requesting all samples
        sampled_het_all = explorer.sample(genotype_het, num_samples=2, strategy='exhaustive')
        self.assertEqual(len(sampled_het_all), 2)
        self.assertEqual(set(sampled_het_all), expected_pairs_het)

        # Test requesting fewer samples
        sampled_het_less = explorer.sample(genotype_het, num_samples=1, strategy='exhaustive')
        self.assertEqual(len(sampled_het_less), 1)
        self.assertIn(sampled_het_less[0], expected_pairs_het) # Check the sampled one is valid

        # Test requesting more samples than available
        sampled_het_more = explorer.sample(genotype_het, num_samples=5, strategy='exhaustive')
        self.assertEqual(len(sampled_het_more), 2) # Should return only the valid ones
        self.assertEqual(set(sampled_het_more), expected_pairs_het)

        # --- Homozygous Case ---
        genotype_hom = ['B*15:01', 'B*15:01']
        expected_pairs_hom = set(rules.get_valid_haplotype_pairs(genotype_hom))
        self.assertEqual(len(expected_pairs_hom), 1) # Should be ('B*15:01', 'B*15:01')

        sampled_hom_all = explorer.sample(genotype_hom, num_samples=1, strategy='exhaustive')
        self.assertEqual(len(sampled_hom_all), 1)
        self.assertEqual(set(sampled_hom_all), expected_pairs_hom)

        sampled_hom_more = explorer.sample(genotype_hom, num_samples=3, strategy='exhaustive')
        self.assertEqual(len(sampled_hom_more), 1)
        self.assertEqual(set(sampled_hom_more), expected_pairs_hom)

    def test_sample_unknown_strategy(self):
        """Test that an unknown strategy raises NotImplementedError."""
        checker = HaplotypeCompatibilityChecker()
        explorer = HaplotypeSpaceExplorer(compatibility_checker=checker)
        genotype = ['A*01:01', 'A*02:01']

        with self.assertRaisesRegex(NotImplementedError, "Haplotype sampling strategy 'unknown_strategy' is not yet implemented."):
            explorer.sample(genotype, num_samples=1, strategy='unknown_strategy')


if __name__ == '__main__': # noqa: E742
    unittest.main() # noqa: E742
