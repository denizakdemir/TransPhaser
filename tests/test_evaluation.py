"""
TDD tests for updated evaluation.py with dual-haplotype format.

Tests the HLAPhasingMetrics class with new (H1, H2) prediction format.
"""

import unittest
import pytest
from transphaser.evaluation import HLAPhasingMetrics
from transphaser.data_preprocessing import AlleleTokenizer


class TestHLAPhasingMetricsDualHaplotype(unittest.TestCase):
    """Tests for HLAPhasingMetrics with dual-haplotype predictions."""
    
    def setUp(self):
        """Set up tokenizer and metrics."""
        self.tokenizer = AlleleTokenizer()
        self.loci_order = ['HLA-A', 'HLA-B', 'HLA-DRB1']
        
        # Build vocab
        for locus in self.loci_order:
            alleles = ['A*01:01', 'A*02:01', 'B*07:02', 'B*08:01', 
                      'DRB1*03:01', 'DRB1*15:01']
            self.tokenizer.build_vocabulary(locus, alleles)
        
        self.metrics = HLAPhasingMetrics(self.tokenizer)
    
    def test_perfect_predictions(self):
        """Test that perfect predictions give 100% accuracy."""
        # Perfect predictions (H1, H2) match truth
        predicted = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'], 
             ['A*02:01', 'B*08:01', 'DRB1*15:01']),
            (['A*02:01', 'B*08:01', 'DRB1*15:01'],
             ['A*01:01', 'B*07:02', 'DRB1*03:01'])
        ]
        
        true = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*02:01', 'B*08:01', 'DRB1*15:01']),
            (['A*02:01', 'B*08:01', 'DRB1*15:01'],
             ['A*01:01', 'B*07:02', 'DRB1*03:01'])
        ]
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        self.assertEqual(metrics['phasing_accuracy'], 1.0)
        self.assertEqual(metrics['avg_hamming_distance'], 0.0)
        self.assertEqual(metrics['avg_switch_errors'], 0.0)
    
    def test_phase_swapped_predictions(self):
        """Test that swapped phases still give 100% accuracy."""
        # Predicted has H1/H2 swapped but still correct
        predicted = [
            (['A*02:01', 'B*08:01', 'DRB1*15:01'],  # Swapped H1
             ['A*01:01', 'B*07:02', 'DRB1*03:01'])  # Swapped H2
        ]
        
        true = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*02:01', 'B*08:01', 'DRB1*15:01'])
        ]
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        # Should recognize swapped phases as correct
        self.assertEqual(metrics['phasing_accuracy'], 1.0)
        self.assertEqual(metrics['avg_hamming_distance'], 0.0)
    
    def test_partial_errors(self):
        """Test predictions with some errors."""
        predicted = [
            # Sample 1: Perfect
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*02:01', 'B*08:01', 'DRB1*15:01']),
            # Sample 2: One allele wrong
            (['A*01:01', 'B*07:02', 'DRB1*15:01'],  # DRB1 wrong!
             ['A*02:01', 'B*08:01', 'DRB1*15:01'])
        ]
        
        true = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*02:01', 'B*08:01', 'DRB1*15:01']),
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*02:01', 'B*08:01', 'DRB1*15:01'])
        ]
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        # 50% accuracy (1 out of 2 correct)
        self.assertEqual(metrics['phasing_accuracy'], 0.5)
        # Hamming distance should be > 0
        self.assertGreater(metrics['avg_hamming_distance'], 0.0)
    
    def test_string_format_haplotypes(self):
        """Test with underscore-separated string format."""
        predicted = [
            ('A*01:01_B*07:02_DRB1*03:01', 
             'A*02:01_B*08:01_DRB1*15:01')
        ]
        
        true = [
            ('A*01:01_B*07:02_DRB1*03:01',
             'A*02:01_B*08:01_DRB1*15:01')
        ]
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        self.assertEqual(metrics['phasing_accuracy'], 1.0)
    
    def test_empty_input(self):
        """Test with empty predictions."""
        metrics = self.metrics.calculate_metrics([], [])
        
        # Should handle gracefully
        self.assertEqual(metrics['phasing_accuracy'], 0.0)
        self.assertEqual(metrics['avg_hamming_distance'], 0.0)
        self.assertEqual(metrics['avg_switch_errors'], 0.0)
    
    def test_homozygous_samples(self):
        """Test with identical H1 and H2 (homozygous)."""
        predicted = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*01:01', 'B*07:02', 'DRB1*03:01'])  # Same!
        ]
        
        true = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*01:01', 'B*07:02', 'DRB1*03:01'])
        ]
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        self.assertEqual(metrics['phasing_accuracy'], 1.0)
        self.assertEqual(metrics['avg_hamming_distance'], 0.0)
    
    def test_switch_error_calculation(self):
        """Test switch error for predictions with phase switches."""
        # Prediction has a phase switch in middle
        predicted = [
            (['A*01:01', 'B*08:01', 'DRB1*03:01'],  # B is from H2!
             ['A*02:01', 'B*07:02', 'DRB1*15:01'])  # B is from H1!
        ]
        
        true = [
            (['A*01:01', 'B*07:02', 'DRB1*03:01'],
             ['A*02:01', 'B*08:01', 'DRB1*15:01'])
        ]
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        # Should detect switch error
        self.assertGreater(metrics['avg_switch_errors'], 0.0)
        self.assertLess(metrics['phasing_accuracy'], 1.0)
    
    def test_hamming_distance_calculation(self):
        """Test Hamming distance calculation."""
        hap1 = 'A*01:01_B*07:02_DRB1*03:01'
        hap2_same = 'A*01:01_B*07:02_DRB1*03:01'
        hap2_diff1 = 'A*02:01_B*07:02_DRB1*03:01'  # 1 diff
        hap2_diff3 = 'A*02:01_B*08:01_DRB1*15:01'  # 3 diff
        
        self.assertEqual(self.metrics._calculate_hamming(hap1, hap2_same), 0)
        self.assertEqual(self.metrics._calculate_hamming(hap1, hap2_diff1), 1)
        self.assertEqual(self.metrics._calculate_hamming(hap1, hap2_diff3), 3)


class TestHLAPhasingMetricsIntegration(unittest.TestCase):
    """Integration tests for metrics with realistic scenarios."""
    
    def setUp(self):
        """Set up with realistic HLA data."""
        self.tokenizer = AlleleTokenizer()
        self.loci_order = ['HLA-A', 'HLA-B', 'HLA-DRB1']
        
        # Build realistic vocab
        alleles = [
            'A*01:01', 'A*02:01', 'A*03:01', 'A*24:02',
            'B*07:02', 'B*08:01', 'B*35:01', 'B*44:02',
            'DRB1*03:01', 'DRB1*04:01', 'DRB1*15:01', 'DRB1*01:01'
        ]
        for locus in self.loci_order:
            self.tokenizer.build_vocabulary(locus, alleles)
        
        self.metrics = HLAPhasingMetrics(self.tokenizer)
    
    def test_batch_evaluation(self):
        """Test evaluation on a batch of predictions."""
        # Create 10 predictions, 7 perfect, 3 with errors
        predicted = []
        true = []
        
        # 7 perfect predictions
        for i in range(7):
            pred_pair = (
                ['A*01:01', 'B*07:02', 'DRB1*03:01'],
                ['A*02:01', 'B*08:01', 'DRB1*15:01']
            )
            true_pair = (
                ['A*01:01', 'B*07:02', 'DRB1*03:01'],
                ['A*02:01', 'B*08:01', 'DRB1*15:01']
            )
            predicted.append(pred_pair)
            true.append(true_pair)
        
        # 3 predictions with errors
        for i in range(3):
            pred_pair = (
                ['A*03:01', 'B*35:01', 'DRB1*04:01'],  # Wrong!
                ['A*24:02', 'B*44:02', 'DRB1*01:01']   # Wrong!
            )
            true_pair = (
                ['A*01:01', 'B*07:02', 'DRB1*03:01'],
                ['A*02:01', 'B*08:01', 'DRB1*15:01']
            )
            predicted.append(pred_pair)
            true.append(true_pair)
        
        metrics = self.metrics.calculate_metrics(predicted, true)
        
        # Should be 70% accuracy
        self.assertAlmostEqual(metrics['phasing_accuracy'], 0.7, places=2)
        
        # Average hamming should be < 2 (some perfect, some wrong)
        self.assertLess(metrics['avg_hamming_distance'], 2.0)
        self.assertGreater(metrics['avg_hamming_distance'], 0.0)


if __name__ == '__main__':
    unittest.main()
