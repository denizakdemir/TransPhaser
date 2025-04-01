import unittest
import torch # Added for mask generator test later if needed

# Import necessary classes
from transphaser.compatibility import HaplotypeCompatibilityChecker, HLACompatibilityRules, CompatibilityMaskGenerator, HaplotypeConstraintPropagator # Reverted to src.
from transphaser.data_preprocessing import AlleleTokenizer # Needed for mock, Reverted to src.
import logging # Added for mask generator warning check

# --- Mocks ---
# Mock Tokenizer needed for MaskGenerator tests
class MockAlleleTokenizer:
    def __init__(self):
        self.special_tokens = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
        self.locus_vocabularies = {
            'HLA-A': {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'A*01:01': 4, 'A*02:01': 5, 'A*03:01': 6},
            'HLA-B': {'PAD': 0, 'UNK': 1, 'BOS': 2, 'EOS': 3, 'B*07:02': 4, 'B*08:01': 5, 'B*15:01': 6}
        }
        self.locus_reverse_vocabularies = {
             locus: {v: k for k, v in vocab.items()}
             for locus, vocab in self.locus_vocabularies.items()
        }
        # print("MockAlleleTokenizer Initialized for compatibility tests") # Reduce noise

    def tokenize(self, locus, allele):
        return self.locus_vocabularies.get(locus, {}).get(allele, self.special_tokens['UNK'])

    def get_vocab_size(self, locus):
         return len(self.locus_vocabularies.get(locus, {}))

# --- Test Classes ---
class TestHaplotypeCompatibilityChecker(unittest.TestCase):

    def test_initialization(self):
        """Test HaplotypeCompatibilityChecker initialization."""
        checker_default = HaplotypeCompatibilityChecker()
        self.assertFalse(checker_default.allow_imputation)
        checker_impute = HaplotypeCompatibilityChecker(allow_imputation=True)
        self.assertTrue(checker_impute.allow_imputation)
        checker_no_impute = HaplotypeCompatibilityChecker(allow_imputation=False)
        self.assertFalse(checker_no_impute.allow_imputation)

    def test_check_compatibility(self):
        """Test the check method for haplotype compatibility."""
        checker = HaplotypeCompatibilityChecker()
        self.assertTrue(checker.check(['A*01:01', 'A*02:01'], 'A*01:01', 'A*02:01'))
        self.assertTrue(checker.check(['A*01:01', 'A*02:01'], 'A*02:01', 'A*01:01'))
        self.assertTrue(checker.check(['A*03:01', 'A*03:01'], 'A*03:01', 'A*03:01'))
        self.assertFalse(checker.check(['A*01:01', 'A*02:01'], 'A*03:01', 'A*11:01'))
        self.assertFalse(checker.check(['A*01:01', 'A*02:01'], 'A*01:01', 'A*11:01'))
        self.assertFalse(checker.check(['A*01:01', 'A*02:01'], 'A*01:01', 'A*01:01'))
        self.assertFalse(checker.check(['A*03:01', 'A*03:01'], 'A*03:01', 'A*11:01'))
        self.assertFalse(checker.check(['A*03:01', 'A*03:01'], 'A*11:01', 'A*03:01'))
        self.assertFalse(checker.check(['A*03:01', 'A*03:01'], 'A*11:01', 'A*24:02'))


class TestHLACompatibilityRules(unittest.TestCase):

    def test_initialization(self):
        """Test HLACompatibilityRules initialization."""
        rules_default = HLACompatibilityRules()
        self.assertTrue(rules_default.strict_mode)
        rules_relaxed = HLACompatibilityRules(strict_mode=False)
        self.assertFalse(rules_relaxed.strict_mode)

    def test_get_valid_haplotype_pairs(self):
        """Test getting valid haplotype pairs for a genotype."""
        rules = HLACompatibilityRules()

        # Heterozygous case
        geno_het = ['A*01:01', 'A*02:01']
        expected_het = [('A*01:01', 'A*02:01'), ('A*02:01', 'A*01:01')]
        # Use set comparison to ignore order of pairs in the list
        self.assertEqual(set(rules.get_valid_haplotype_pairs(geno_het)), set(expected_het))

        # Homozygous case
        geno_hom = ['B*07:02', 'B*07:02']
        expected_hom = [('B*07:02', 'B*07:02')]
        self.assertEqual(set(rules.get_valid_haplotype_pairs(geno_hom)), set(expected_hom))

        # Test invalid input
        with self.assertRaises(TypeError):
            rules.get_valid_haplotype_pairs(['A*01:01']) # Not enough alleles
        with self.assertRaises(TypeError):
             rules.get_valid_haplotype_pairs("not_a_list")


class TestCompatibilityMaskGenerator(unittest.TestCase):

    def test_initialization(self):
         """Test CompatibilityMaskGenerator initialization."""
         mock_tokenizer = MockAlleleTokenizer()
         mock_rules = HLACompatibilityRules()

         mask_generator = CompatibilityMaskGenerator(
             tokenizer=mock_tokenizer,
             compatibility_rules=mock_rules
         )

         self.assertIs(mask_generator.tokenizer, mock_tokenizer)
         self.assertIs(mask_generator.compatibility_rules, mock_rules)

    def test_generate_mask(self):
         """Test generating compatibility masks."""
         mock_tokenizer = MockAlleleTokenizer()
         mock_rules = HLACompatibilityRules()
         mask_generator = CompatibilityMaskGenerator(
             tokenizer=mock_tokenizer,
             compatibility_rules=mock_rules
         )

         locus_a = 'HLA-A'
         vocab_size_a = mock_tokenizer.get_vocab_size(locus_a)
         # Allele tokens: A*01:01 -> 4, A*02:01 -> 5, A*03:01 -> 6
         token_a0101 = 4
         token_a0201 = 5
         token_a0301 = 6

         # --- Case 1: Predicting first haplotype allele for Het genotype ---
         geno_het = ['A*01:01', 'A*02:01']
         mask1 = mask_generator.generate_mask(locus=locus_a, genotype=geno_het, partial_haplotype1=None)
         expected_mask1 = torch.zeros(vocab_size_a, dtype=torch.bool)
         expected_mask1[token_a0101] = True
         expected_mask1[token_a0201] = True
         self.assertTrue(torch.equal(mask1, expected_mask1))

         # --- Case 2: Predicting second haplotype allele for Het genotype ---
         # Given hap1 = A*01:01, only A*02:01 should be allowed for hap2
         mask2 = mask_generator.generate_mask(locus=locus_a, genotype=geno_het, partial_haplotype1='A*01:01')
         expected_mask2 = torch.zeros(vocab_size_a, dtype=torch.bool)
         expected_mask2[token_a0201] = True
         self.assertTrue(torch.equal(mask2, expected_mask2))

         # Given hap1 = A*02:01, only A*01:01 should be allowed for hap2
         mask3 = mask_generator.generate_mask(locus=locus_a, genotype=geno_het, partial_haplotype1='A*02:01')
         expected_mask3 = torch.zeros(vocab_size_a, dtype=torch.bool)
         expected_mask3[token_a0101] = True
         self.assertTrue(torch.equal(mask3, expected_mask3))

         # --- Case 3: Predicting first haplotype allele for Hom genotype ---
         geno_hom = ['A*03:01', 'A*03:01']
         mask4 = mask_generator.generate_mask(locus=locus_a, genotype=geno_hom, partial_haplotype1=None)
         expected_mask4 = torch.zeros(vocab_size_a, dtype=torch.bool)
         expected_mask4[token_a0301] = True
         self.assertTrue(torch.equal(mask4, expected_mask4))

         # --- Case 4: Predicting second haplotype allele for Hom genotype ---
         # Given hap1 = A*03:01, only A*03:01 should be allowed for hap2
         mask5 = mask_generator.generate_mask(locus=locus_a, genotype=geno_hom, partial_haplotype1='A*03:01')
         expected_mask5 = torch.zeros(vocab_size_a, dtype=torch.bool)
         expected_mask5[token_a0301] = True
         self.assertTrue(torch.equal(mask5, expected_mask5))

         # --- Case 5: Incompatible partial haplotype (should allow nothing) ---
         mask6 = mask_generator.generate_mask(locus=locus_a, genotype=geno_hom, partial_haplotype1='A*01:01')
         expected_mask6 = torch.zeros(vocab_size_a, dtype=torch.bool) # All False
         self.assertTrue(torch.equal(mask6, expected_mask6))


class TestHaplotypeConstraintPropagator(unittest.TestCase):

    def test_initialization(self):
        """Test HaplotypeConstraintPropagator initialization."""
        mock_rules = HLACompatibilityRules() # Depends on rules

        propagator = HaplotypeConstraintPropagator(compatibility_rules=mock_rules)

        self.assertIs(propagator.compatibility_rules, mock_rules)


if __name__ == '__main__':
    unittest.main()
