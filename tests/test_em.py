import pytest
import numpy as np
from transphaser.em import EMHaplotypePhaser

class TestEMHaplotypePhaser:
    
    def test_initialization(self):
        """Test that the phaser initializes with correct defaults."""
        phaser = EMHaplotypePhaser(tolerance=1e-5, max_iterations=50)
        assert phaser.tolerance == 1e-5
        assert phaser.max_iterations == 50
        assert phaser.haplotype_frequencies == {}
        assert not phaser.converged

    def test_get_possible_phasings_homozygous(self):
        """Test phasing expansion for a single homozygous locus."""
        # Genotype: Locus1=(A, A)
        genotype = [("A", "A")]
        phaser = EMHaplotypePhaser()
        phasings = phaser._get_possible_phasings(genotype)
        
        # Should be just one possibility: h1=(A,), h2=(A,)
        assert len(phasings) == 1
        h1, h2 = phasings[0]
        assert h1 == ("A",)
        assert h2 == ("A",)

    def test_get_possible_phasings_heterozygous(self):
        """Test phasing expansion for a single heterozygous locus."""
        # Genotype: Locus1=(A, B)
        genotype = [("A", "B")]
        phaser = EMHaplotypePhaser()
        phasings = phaser._get_possible_phasings(genotype)
        
        # Should be two possibilities: h1=(A,), h2=(B,)  OR  h1=(B,), h2=(A,)
        assert len(phasings) == 2
        
        expected_pairs = {
            (("A",), ("B",)),
            (("B",), ("A",))
        }
        
        actual_pairs = set(phasings)
        assert actual_pairs == expected_pairs

    def test_get_possible_phasings_multilocus(self):
        """Test phasing expansion for multiple loci."""
        # Genotype: Locus1=(A, A) [Homo], Locus2=(C, D) [Hetero]
        genotype = [("A", "A"), ("C", "D")]
        phaser = EMHaplotypePhaser()
        phasings = phaser._get_possible_phasings(genotype)
        
        # Combinations:
        # Locus 1 choices: (A, A)
        # Locus 2 choices: (C, D) or (D, C)
        # Total = 1 * 2 = 2 phasings
        
        assert len(phasings) == 2
        
        # Expected:
        # 1. h1=(A, C), h2=(A, D)
        # 2. h1=(A, D), h2=(A, C)
        
        p1 = (("A", "C"), ("A", "D"))
        p2 = (("A", "D"), ("A", "C"))
        
        assert p1 in phasings
        assert p2 in phasings

    def test_fit_simple_case(self):
        """
        Test fitting on a trivial population where we know the answer.
        Case: 
        Sample 1: A/A, C/C -> Haps: (A, C), (A, C)
        Sample 2: B/B, D/D -> Haps: (B, D), (B, D)
        Result: f(A,C)=0.5, f(B,D)=0.5
        """
        genotypes = [
            [("A", "A"), ("C", "C")],
            [("B", "B"), ("D", "D")]
        ]
        
        phaser = EMHaplotypePhaser()
        phaser.fit(genotypes)
        
        freqs = phaser.haplotype_frequencies
        
        # Check expected haplotypes exist
        h_ac = ("A", "C")
        h_bd = ("B", "D")
        
        assert h_ac in freqs
        assert h_bd in freqs
        
        # Frequencies might have slight float errors, use approx
        assert freqs[h_ac] == pytest.approx(0.5)
        assert freqs[h_bd] == pytest.approx(0.5)

    def test_predict_ambiguous(self):
        """
        Test prediction on an ambiguous case using pre-set frequencies.
        Genotype: A/B, C/D
        Possible assignments:
          1. (A, C) + (B, D)
          2. (A, D) + (B, C)
        
        If we force f(A,C) and f(B,D) to be high, it should pick #1.
        """
        phaser = EMHaplotypePhaser()
        # Manually set frequencies
        phaser.haplotype_frequencies = {
            ("A", "C"): 0.4,
            ("B", "D"): 0.4,
            ("A", "D"): 0.1,
            ("B", "C"): 0.1
        }
        
        genotype = [[("A", "B"), ("C", "D")]]
        predictions = phaser.predict(genotype)
        
        h1, h2 = predictions[0]
        
        # We expect (A, C) and (B, D)
        # Note: predict returns lists of alleles
        h1_tuple = tuple(h1)
        h2_tuple = tuple(h2)
        
        # Since order in pair doesn't matter for correctness but matters for the specific tuple check
        pair = {h1_tuple, h2_tuple}
        expected = {("A", "C"), ("B", "D")}
        
        assert pair == expected

    def test_evaluate_accuracy(self):
        """Test the evaluate method."""
        # Setup: Perfect retrieval
        phaser = EMHaplotypePhaser()
        # Mock predict to return exactly the ground truth
        phaser.predict = lambda x: [(["A"], ["B"])]
        
        genotypes = [[("A", "B")]] # Dummy input
        ground_truth = [(["A"], ["B"])]
        
        results = phaser.evaluate(genotypes, ground_truth)
        assert results['phasing_accuracy'] == 1.0
        
        # Setup: Fail
        phaser.predict = lambda x: [(["X"], ["Y"])]
        results = phaser.evaluate(genotypes, ground_truth)
        assert results['phasing_accuracy'] == 0.0

    def test_fit_convergence(self):
        """Test that the EM algorithm converges on a slightly more complex case."""
        # 10 Samples:
        # 4x (A, A) / (C, C) -> Haps (A, C)
        # 4x (B, B) / (D, D) -> Haps (B, D)
        # 2x (A, B) / (C, D) -> Ambiguous: (A, C)/(B, D) OR (A, D)/(B, C)
        # Since (A,C) and (B,D) are common implies they should be preferred for the ambiguous ones.
        
        genotypes = []
        for _ in range(4):
            genotypes.append([("A", "A"), ("C", "C")])
        for _ in range(4):
            genotypes.append([("B", "B"), ("D", "D")])
        for _ in range(2):
            genotypes.append([("A", "B"), ("C", "D")])
            
        phaser = EMHaplotypePhaser(tolerance=1e-6, max_iterations=50)
        phaser.fit(genotypes)
        
        assert phaser.converged
        
        # Check that common haplotypes dominated
        freqs = phaser.haplotype_frequencies
        
        assert freqs.get(("A", "C"), 0) > 0.45

        assert freqs.get(("B", "D"), 0) > 0.45
        assert freqs.get(("A", "D"), 0) < 0.05

    def test_get_estimated_frequencies(self):
        """Test getting estimated frequencies after fitting."""
        genotypes = [
            [("A", "A"), ("C", "C")],
            [("A", "A"), ("C", "C")],
            [("B", "B"), ("D", "D")]
        ]
        phaser = EMHaplotypePhaser()
        returned_freqs = phaser.fit(genotypes)
        
        # Total haplotypes: 2 samples * 2 haps each = 4 (A, C)
        # 1 sample * 2 haps each = 2 (B, D)
        # Total = 6 haplotypes.
        # f(A, C) = 4/6 = 2/3
        # f(B, D) = 2/6 = 1/3
        
        freqs = phaser.get_estimated_frequencies(sort=True)
        # Top frequency should be (A, C)
        haps = list(freqs.keys())
        values = list(freqs.values())
        
        assert haps[0] == ("A", "C")
        assert values[0] == pytest.approx(2/3)
        assert haps[1] == ("B", "D")
        assert values[1] == pytest.approx(1/3)
        
        # Check threshold
        sparse_freqs = phaser.get_estimated_frequencies(threshold=0.5)
        assert len(sparse_freqs) == 1
        assert ("A", "C") in sparse_freqs
        
        # Check that fit returns the same thing (sorted by default)
        assert returned_freqs == freqs

