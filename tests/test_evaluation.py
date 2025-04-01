import unittest
import torch # Likely needed for metric calculations
import torch.nn as nn # Added for mock model
from unittest.mock import patch, MagicMock, ANY # Import mock utilities
import numpy as np # Import numpy

# Placeholder for AlleleTokenizer (needed for mock)
from transphaser.data_preprocessing import AlleleTokenizer # Reverted to src.
# Placeholder for the classes we are about to create
from transphaser.evaluation import HLAPhasingMetrics, PhasingUncertaintyEstimator, HaplotypeCandidateRanker, PhasingResultVisualizer # Reverted to src.
# Import MATPLOTLIB_AVAILABLE flag to check availability in tests
from transphaser.evaluation import MATPLOTLIB_AVAILABLE

# --- Mocks ---
# Use the actual AlleleTokenizer for initialization tests
# class MockAlleleTokenizer:
#     def __init__(self):
#         # Define some special tokens
#         self.special_tokens = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
#         print("MockAlleleTokenizer Initialized for evaluation tests")
#     # Add methods like tokenize/detokenize if needed by HLAPhasingMetrics

class MockModel(nn.Module):
    """Minimal mock model needed for UncertaintyEstimator/Ranker init."""
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(1.0)) # Dummy parameter
        # print("MockModel Initialized for evaluation tests") # Reduce noise

    def score_haplotype_pair(self, batch_sample_info, haplotype_pair):
        """
        Mock scoring function. Returns a deterministic score based on the pair.
        Lower score for pairs containing 'BAD'.
        """
        # batch_sample_info is ignored in this simple mock
        hap1, hap2 = haplotype_pair
        score = -10.0 # Base score
        # Simple logic: penalize if 'BAD' allele is present
        if 'BAD' in hap1 or 'BAD' in hap2:
            score -= 5.0
        # Add small unique value based on hash to differentiate pairs
        score += (hash(haplotype_pair) % 100) / 100.0
        return torch.tensor(score) # Return as tensor

# --- Test Classes ---
class TestHLAPhasingMetrics(unittest.TestCase):

    def test_initialization(self):
        """Test HLAPhasingMetrics initialization."""
        # Use the actual tokenizer
        real_tokenizer = AlleleTokenizer()

        metrics_calculator = HLAPhasingMetrics(tokenizer=real_tokenizer)

        # Check attributes
        self.assertIs(metrics_calculator.tokenizer, real_tokenizer)
        # Add checks for other attributes if any (e.g., metric storage)


class TestPhasingUncertaintyEstimator(unittest.TestCase):

    def test_initialization(self):
        """Test PhasingUncertaintyEstimator initialization."""
        mock_model = MockModel()
        sampling_iterations = 50

        # Default initialization
        estimator_default = PhasingUncertaintyEstimator(model=mock_model)
        self.assertIs(estimator_default.model, mock_model)
        self.assertEqual(estimator_default.sampling_iterations, 100) # Default from description

        # Custom initialization
        estimator_custom = PhasingUncertaintyEstimator(
            model=mock_model,
            sampling_iterations=sampling_iterations
        )
        self.assertIs(estimator_custom.model, mock_model)
        self.assertEqual(estimator_custom.sampling_iterations, sampling_iterations)


class TestHaplotypeCandidateRanker(unittest.TestCase):

     def test_initialization(self):
         """Test HaplotypeCandidateRanker initialization."""
         mock_model = MockModel()
         num_candidates = 5
         diversity_weight = 0.2

         # Default initialization
         ranker_default = HaplotypeCandidateRanker(model=mock_model)
         self.assertIs(ranker_default.model, mock_model)
         self.assertEqual(ranker_default.num_candidates, 10) # Default from description
         self.assertEqual(ranker_default.diversity_weight, 0.1) # Default from description

         # Custom initialization
         ranker_custom = HaplotypeCandidateRanker(
             model=mock_model,
             num_candidates=num_candidates,
             diversity_weight=diversity_weight
         )
         self.assertIs(ranker_custom.model, mock_model)
         self.assertEqual(ranker_custom.num_candidates, num_candidates) # noqa: E742
         self.assertEqual(ranker_custom.diversity_weight, diversity_weight) # noqa: E742

     def test_rank_candidates(self):
         """Test ranking of candidate haplotype pairs."""
         mock_model = MockModel()
         num_candidates_to_return = 3
         ranker = HaplotypeCandidateRanker(model=mock_model, num_candidates=num_candidates_to_return)

         # Mock batch data (content doesn't matter much for this mock model)
         batch = {'genotypes': ['Sample1_Geno', 'Sample2_Geno']}
         batch_size = len(batch['genotypes'])

         # Mock candidate haplotypes per sample
         candidates_sample1 = [
             ('A*01_B*01', 'A*02_B*02'), # Good pair 1
             ('A*01_B*01', 'A*03_B*BAD'), # Bad pair 1
             ('A*01_B*01', 'A*02_B*03'), # Good pair 2
             ('A*04_B*BAD', 'A*02_B*02'), # Bad pair 2
             ('A*01_B*04', 'A*02_B*02'), # Good pair 3
         ]
         candidates_sample2 = [
             ('C*01_D*01', 'C*01_D*BAD'), # Bad pair 3
             ('C*01_D*01', 'C*02_D*02'), # Good pair 4
         ]
         all_candidates = [candidates_sample1, candidates_sample2]

         # --- Act ---
         ranked_results = ranker.rank_candidates(batch, all_candidates)

         # --- Assert ---
         # Check overall structure
         self.assertIsInstance(ranked_results, list)
         self.assertEqual(len(ranked_results), batch_size)

         # Check sample 1 results
         ranked_sample1 = ranked_results[0]
         self.assertIsInstance(ranked_sample1, list)
         # Should return top 'num_candidates_to_return'
         self.assertEqual(len(ranked_sample1), num_candidates_to_return)
         # Check format (pair, score)
         self.assertIsInstance(ranked_sample1[0], tuple)
         self.assertEqual(len(ranked_sample1[0]), 2)
         self.assertIsInstance(ranked_sample1[0][0], tuple) # The pair itself
         self.assertIsInstance(ranked_sample1[0][1], torch.Tensor) # The score

         # Check scores are descending
         scores_sample1 = [score.item() for _, score in ranked_sample1]
         self.assertEqual(scores_sample1, sorted(scores_sample1, reverse=True))

         # Check that the top pairs are the 'Good' ones based on mock scoring
         top_pairs_sample1 = [pair for pair, _ in ranked_sample1]
         self.assertNotIn(('A*01_B*01', 'A*03_B*BAD'), top_pairs_sample1)
         self.assertNotIn(('A*04_B*BAD', 'A*02_B*02'), top_pairs_sample1)
         self.assertIn(('A*01_B*01', 'A*02_B*02'), top_pairs_sample1)

         # Check sample 2 results (fewer candidates than requested)
         ranked_sample2 = ranked_results[1]
         self.assertIsInstance(ranked_sample2, list)
         self.assertEqual(len(ranked_sample2), len(candidates_sample2)) # Returns all available
         scores_sample2 = [score.item() for _, score in ranked_sample2]
         self.assertEqual(scores_sample2, sorted(scores_sample2, reverse=True))
         # Check the good pair is ranked higher
         self.assertEqual(ranked_sample2[0][0], ('C*01_D*01', 'C*02_D*02'))

     # Test for diversity weighting (MMR)
     def test_rank_candidates_with_diversity(self):
         """Test ranking promotes diversity using MMR logic."""
         # Import random for shuffling test data
         import random
         mock_model = MockModel()
         num_candidates_to_return = 3
         diversity_lambda = 0.5 # MMR lambda (diversity weight)
         ranker = HaplotypeCandidateRanker(model=mock_model, num_candidates=num_candidates_to_return, diversity_weight=diversity_lambda)

         # Mock batch data
         batch = {'genotypes': ['Sample1_Geno']} # Content doesn't matter for this test
         batch_size = 1

         # Mock candidates with scores designed to show diversity effect
         # P1: Best score
         # P2: 2nd best score, very similar to P1
         # P3: 3rd best score, dissimilar to P1/P2
         pair1 = ('A*01_B*01', 'A*02_B*02') # Score -10.0
         pair2 = ('A*01_B*01', 'A*02_B*01') # Score -10.1 (Similar to P1)
         pair3 = ('C*01_D*01', 'C*02_D*02') # Score -10.2 (Dissimilar)
         pair4 = ('E*01_F*01', 'E*02_F*02') # Score -11.0 (Dissimilar)

         candidates_with_scores = [
             (pair1, torch.tensor(-10.0)),
             (pair2, torch.tensor(-10.1)),
             (pair3, torch.tensor(-10.2)),
             (pair4, torch.tensor(-11.0)),
         ]
         random.shuffle(candidates_with_scores) # Shuffle initial list
         all_candidates = [[p for p, s in candidates_with_scores]] # Structure for ranker

         # Mock the scoring function to return predefined scores
         def mock_score_func(batch_info, pair):
             for p, s in candidates_with_scores:
                 if p == pair: return s
             return torch.tensor(float('-inf'))
         ranker.model.score_haplotype_pair = MagicMock(side_effect=mock_score_func)

         # Mock a similarity function (higher value means more similar)
         # Assume similarity = 1 / (1 + hamming_distance)
         def mock_similarity(p_a, p_b):
             if p_a == p_b: return 1.0
             # Simplified Hamming for test pairs
             if (p_a == pair1 and p_b == pair2) or (p_a == pair2 and p_b == pair1): return 1 / (1 + 1) # 0.5 (dist=1)
             if (p_a == pair1 and p_b == pair3) or (p_a == pair3 and p_b == pair1): return 1 / (1 + 4) # 0.2 (dist=4)
             if (p_a == pair2 and p_b == pair3) or (p_a == pair3 and p_b == pair2): return 1 / (1 + 4) # 0.2 (dist=4)
             return 0.1 # Default low similarity for others

         # Patch the internal similarity calculation method
         with patch.object(HaplotypeCandidateRanker, '_calculate_pair_similarity', side_effect=mock_similarity):
             # --- Act ---
             ranked_results = ranker.rank_candidates(batch, all_candidates)

         # --- Assert ---
         self.assertEqual(len(ranked_results), 1)
         ranked_sample1 = ranked_results[0]
         self.assertEqual(len(ranked_sample1), num_candidates_to_return)

         # Expected order with MMR (lambda = 0.5):
         # Formula: MMR = lambda * score - (1 - lambda) * max_similarity
         # 1. Select P1 (score -10.0). Selected = [P1]
         # 2. MMR(P2) = 0.5*(-10.1) - 0.5*sim(P2,P1) = -5.05 - 0.5*0.5 = -5.30
         #    MMR(P3) = 0.5*(-10.2) - 0.5*sim(P3,P1) = -5.10 - 0.5*0.2 = -5.20
         #    MMR(P4) = 0.5*(-11.0) - 0.5*sim(P4,P1) = -5.50 - 0.5*0.1 = -5.55
         #    Select P3 (MMR -5.20). Selected = [P1, P3]
         # 3. MMR(P2) = 0.5*(-10.1) - 0.5*max(sim(P2,P1), sim(P2,P3)) = -5.05 - 0.5*max(0.5, 0.2) = -5.05 - 0.25 = -5.30
         #    MMR(P4) = 0.5*(-11.0) - 0.5*max(sim(P4,P1), sim(P4,P3)) = -5.50 - 0.5*max(0.1, 0.1) = -5.50 - 0.05 = -5.55
         #    Select P2 (MMR -5.30). Selected = [P1, P3, P2]

         expected_ranking = [pair1, pair3, pair2]
         actual_ranking = [p for p, s in ranked_sample1]
         self.assertEqual(actual_ranking, expected_ranking, "Ranking with diversity did not match expected MMR order.")


class TestPhasingResultVisualizer(unittest.TestCase):

    def test_initialization(self):
        """Test PhasingResultVisualizer initialization."""
        # Use the actual tokenizer
        real_tokenizer = AlleleTokenizer()

        # This will fail until the class is defined
        visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)

        # Check attributes
        self.assertIs(visualizer.tokenizer, real_tokenizer)
        # Check if plt attribute is set based on availability
        try:
            import matplotlib.pyplot as plt
            self.assertIsNotNone(visualizer.plt)
        except ImportError:
            self.assertIsNone(visualizer.plt)

    # Replace the old test_plot_likelihoods with this one using mock
    @patch('transphaser.evaluation.plt', MagicMock()) # Mock the plt object within the module
    def test_plot_likelihoods_with_matplotlib(self):
        """Test plot_likelihoods calls matplotlib functions correctly when available."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("matplotlib not available, skipping plotting test.")

        # --- Setup ---
        real_tokenizer = AlleleTokenizer()
        visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)
        # Mock ranked candidates data for the first sample
        mock_scores = [-1.0, -2.5, -3.0]
        ranked_candidates = [
            [(('A*01', 'B*01'), torch.tensor(mock_scores[0])),
             (('A*02', 'B*01'), torch.tensor(mock_scores[1])),
             (('A*03', 'B*03'), torch.tensor(mock_scores[2]))],
            # Add data for a second sample (should only plot first by default)
            [(('C*01', 'D*01'), torch.tensor(-0.5))]
        ]
        output_filename = "test_likelihoods.png"
        expected_output_path = os.path.join(visualizer.output_dir, output_filename) # Assuming output_dir exists

        # Access the mocked plt object via the visualizer instance
        mock_plt = visualizer.plt
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # --- Act ---
        visualizer.plot_likelihoods(ranked_candidates, output_path=expected_output_path)

        # --- Assert ---
        mock_plt.subplots.assert_called_once_with(figsize=(10, 6)) # Check figsize
        # Check bar call with ranks starting from 1 and color argument
        expected_ranks = range(1, len(mock_scores) + 1)
        mock_ax.bar.assert_called_once_with(expected_ranks, mock_scores, color='skyblue')
        mock_ax.set_title.assert_called_once_with(ANY) # Check title is set
        mock_ax.set_xlabel.assert_called_once_with(ANY) # Check xlabel is set
        mock_ax.set_ylabel.assert_called_once_with(ANY) # Check ylabel is set
        mock_ax.set_xticks.assert_called_once_with(expected_ranks) # Check xticks
        mock_ax.grid.assert_called_once_with(axis='y', linestyle='--', alpha=0.7) # Check grid call
        mock_fig.savefig.assert_called_once_with(expected_output_path)
        mock_plt.close.assert_called_once_with(mock_fig)

    def test_plot_likelihoods_no_matplotlib(self):
        """Test plot_likelihoods does nothing gracefully when matplotlib is unavailable."""
        if MATPLOTLIB_AVAILABLE:
            self.skipTest("matplotlib is available, skipping no-matplotlib test.")

        # --- Setup ---
        real_tokenizer = AlleleTokenizer()
        # Temporarily patch MATPLOTLIB_AVAILABLE to False for this test's scope
        with patch('transphaser.evaluation.MATPLOTLIB_AVAILABLE', False):
            visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)
            self.assertIsNone(visualizer.plt) # Verify plt is None

            # Mock ranked candidates data
            ranked_candidates = [
                [(('A*01', 'B*01'), torch.tensor(-1.0))],
            ]
            output_filename = "test_likelihoods_no_mpl.png"

            # --- Act & Assert ---
            # Should run without error and without calling any plotting functions
            try:
                # Use assertLogs to check for the warning message
                with self.assertLogs(level='WARNING') as log:
                    visualizer.plot_likelihoods(ranked_candidates, output_path=output_filename)
                    # Check if the specific warning message is in the logs
                    self.assertTrue(any("matplotlib not available. Skipping plot_likelihoods." in msg for msg in log.output))
            except Exception as e:
                self.fail(f"plot_likelihoods raised an unexpected exception when matplotlib is unavailable: {e}")


    # Update test_plot_uncertainty
    @patch('transphaser.evaluation.plt', MagicMock())
    def test_plot_uncertainty_with_matplotlib(self):
        """Test plot_uncertainty calls matplotlib functions correctly when available."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("matplotlib not available, skipping plotting test.")

        # --- Setup ---
        real_tokenizer = AlleleTokenizer()
        visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)
        mock_entropies = np.array([0.5, 0.8, 0.2, 0.55, 0.1])
        uncertainty_estimates = {'mean_prediction_entropy': torch.tensor(mock_entropies)}
        output_filename = "test_uncertainty.png"
        expected_output_path = os.path.join(visualizer.output_dir, output_filename)

        mock_plt = visualizer.plt
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # --- Act ---
        visualizer.plot_uncertainty(uncertainty_estimates, output_path=expected_output_path)

        # --- Assert ---
        mock_plt.subplots.assert_called_once()
        # Check that hist is called with the numpy array of entropies and bins
        mock_ax.hist.assert_called_once()
        call_args, call_kwargs = mock_ax.hist.call_args
        np.testing.assert_array_equal(call_args[0], mock_entropies) # Check data passed to hist
        self.assertIn('bins', call_kwargs) # Check bins argument was passed
        mock_ax.set_title.assert_called_once_with(ANY)
        mock_ax.set_xlabel.assert_called_once_with(ANY)
        mock_ax.set_ylabel.assert_called_once_with(ANY)
        mock_fig.savefig.assert_called_once_with(expected_output_path)
        mock_plt.close.assert_called_once_with(mock_fig)

    def test_plot_uncertainty_no_matplotlib(self):
        """Test plot_uncertainty does nothing gracefully when matplotlib is unavailable."""
        if MATPLOTLIB_AVAILABLE:
            self.skipTest("matplotlib is available, skipping no-matplotlib test.")

        # --- Setup ---
        real_tokenizer = AlleleTokenizer()
        with patch('transphaser.evaluation.MATPLOTLIB_AVAILABLE', False):
            visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)
            self.assertIsNone(visualizer.plt)
            uncertainty_estimates = {'mean_prediction_entropy': torch.tensor([0.5])}
            output_filename = "test_uncertainty_no_mpl.png"

            # --- Act & Assert ---
            try:
                with self.assertLogs(level='WARNING') as log:
                    visualizer.plot_uncertainty(uncertainty_estimates, output_path=output_filename)
                    self.assertTrue(any("matplotlib not available. Skipping plot_uncertainty." in msg for msg in log.output))
            except Exception as e:
                self.fail(f"plot_uncertainty raised an unexpected exception when matplotlib is unavailable: {e}")


    # Update test_visualize_alignment
    @patch('transphaser.evaluation.plt', MagicMock())
    def test_visualize_alignment_with_matplotlib(self):
        """Test visualize_alignment calls matplotlib functions correctly when available."""
        if not MATPLOTLIB_AVAILABLE:
            self.skipTest("matplotlib not available, skipping plotting test.")

        # --- Setup ---
        real_tokenizer = AlleleTokenizer()
        # Build a minimal vocab for detokenization if needed by implementation
        real_tokenizer.build_vocabulary('HLA-A', ['A*01', 'A*02'])
        real_tokenizer.build_vocabulary('HLA-B', ['B*01', 'B*02'])

        visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)
        # Mock data (using strings for simplicity, implementation might need tokens)
        genotype_str = {'HLA-A': 'A*01/A*02', 'HLA-B': 'B*01/B*02'} # Example format
        predicted_pair = ('A*01_B*01', 'A*02_B*02')
        true_pair = ('A*01_B*02', 'A*02_B*01')
        output_filename = "test_alignment.png"
        expected_output_path = os.path.join(visualizer.output_dir, output_filename)

        mock_plt = visualizer.plt
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # --- Act ---
        visualizer.visualize_alignment(
            genotype_str, # Pass string representation
            predicted_pair, # Pass tuple
            true_haplotypes=true_pair, # Pass tuple
            output_path=expected_output_path
        )

        # --- Assert ---
        mock_plt.subplots.assert_called_once()
        # Check if text function was called with the expected content structure
        mock_ax.text.assert_called_once()
        call_args, call_kwargs = mock_ax.text.call_args
        alignment_text_arg = call_args[2] # The text content is the 3rd positional arg
        # Check if key elements are in the generated text
        self.assertIn("Locus", alignment_text_arg)
        self.assertIn("Genotype", alignment_text_arg)
        self.assertIn("Pred H1", alignment_text_arg)
        self.assertIn("True H2", alignment_text_arg)
        self.assertIn("A*01/A*02", alignment_text_arg) # Genotype
        self.assertIn("A*01 ", alignment_text_arg) # Check for Pred H1 allele A*01 (with space padding)
        self.assertIn("B*01 ", alignment_text_arg) # Check for Pred H1 allele B*01 (with space padding)
        self.assertIn("A*02 ", alignment_text_arg) # Check for True H2 allele A*02
        self.assertIn("B*01 ", alignment_text_arg) # Check for True H2 allele B*01
        self.assertIn("monospace", call_kwargs.get('family')) # Check font

        mock_ax.axis.assert_called_once_with('off') # Check if axes are turned off
        mock_ax.set_title.assert_called_once_with(ANY) # Check title
        mock_fig.savefig.assert_called_once_with(expected_output_path, bbox_inches='tight', pad_inches=0.1) # Check savefig args
        mock_plt.close.assert_called_once_with(mock_fig)


    def test_visualize_alignment_no_matplotlib(self):
        """Test visualize_alignment does nothing gracefully when matplotlib is unavailable."""
        if MATPLOTLIB_AVAILABLE:
            self.skipTest("matplotlib is available, skipping no-matplotlib test.")

        # --- Setup ---
        real_tokenizer = AlleleTokenizer()
        with patch('transphaser.evaluation.MATPLOTLIB_AVAILABLE', False):
            visualizer = PhasingResultVisualizer(tokenizer=real_tokenizer)
            self.assertIsNone(visualizer.plt)
            genotype = {}
            predicted_haplotypes = ()
            output_filename = "test_alignment_no_mpl.png"

            # --- Act & Assert ---
            try:
                with self.assertLogs(level='WARNING') as log:
                    visualizer.visualize_alignment(genotype, predicted_haplotypes, output_path=output_filename)
                    self.assertTrue(any("matplotlib not available. Skipping visualize_alignment." in msg for msg in log.output))
            except Exception as e:
                self.fail(f"visualize_alignment raised an unexpected exception when matplotlib is unavailable: {e}")


# Need to import os for path joining in the test
import os

if __name__ == '__main__':
    unittest.main()
