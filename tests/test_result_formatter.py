import unittest
import json
from transphaser.result_formatter import PhasingResultFormatter # Reverted to src.
# Assuming AlleleTokenizer exists and can be mocked or instantiated simply
from transphaser.data_preprocessing import AlleleTokenizer # Reverted to src.

class TestPhasingResultFormatter(unittest.TestCase):

    def setUp(self):
        # Basic tokenizer setup for testing
        self.tokenizer = AlleleTokenizer()
        # Add dummy vocab for loci used in tests if needed by formatter logic
        # self.tokenizer.build_vocabulary('HLA-A', ['A*01:01', 'A*02:01'])
        # self.tokenizer.build_vocabulary('HLA-B', ['B*07:02', 'B*08:01'])

    def test_initialization(self):
        """Test if the formatter initializes correctly."""
        formatter = PhasingResultFormatter(tokenizer=self.tokenizer, num_candidates=3)
        self.assertIsNotNone(formatter)
        self.assertEqual(formatter.num_candidates, 3)
        self.assertEqual(formatter.tokenizer, self.tokenizer)

    def test_format_single_result_no_uncertainty(self):
        """Test formatting a single result with ranked candidates."""
        formatter = PhasingResultFormatter(tokenizer=self.tokenizer, num_candidates=2)
        # Example ranked candidates for one sample: list of (hap_pair, score) tuples
        # hap_pair is tuple(hap1_str, hap2_str)
        ranked_candidates = [
            (('A*01:01_B*07:02', 'A*02:01_B*08:01'), -0.5), # Score 1 (higher is better)
            (('A*01:01_B*08:01', 'A*02:01_B*07:02'), -1.2)  # Score 2
        ]
        sample_id = "Sample_001"

        formatted_output = formatter.format_result(sample_id, ranked_candidates)

        # Expected JSON structure (as string for comparison, or load as dict)
        expected_dict = {
            "sample_id": "Sample_001",
            "phasing_results": [
                {
                    "rank": 1,
                    "haplotype1": "A*01:01_B*07:02",
                    "haplotype2": "A*02:01_B*08:01",
                    "score": -0.5,
                    "uncertainty": None # No uncertainty provided in this test
                },
                {
                    "rank": 2,
                    "haplotype1": "A*01:01_B*08:01",
                    "haplotype2": "A*02:01_B*07:02",
                    "score": -1.2,
                    "uncertainty": None
                }
            ],
            "metadata": {} # No metadata provided in this test
        }

        # Compare loaded JSON from output with expected dict
        self.assertEqual(json.loads(formatted_output), expected_dict)

    def test_format_single_result_with_uncertainty_and_metadata(self):
        """Test formatting with uncertainty and metadata."""
        formatter = PhasingResultFormatter(tokenizer=self.tokenizer, num_candidates=1)
        ranked_candidates = [
            (('C*01:01', 'C*02:02'), 0.95)
        ]
        sample_id = "Sample_002"
        uncertainty_metrics = {"entropy": 0.15, "confidence": 0.9}
        metadata = {"run_id": "run_abc", "model_version": "1.1"}

        formatted_output = formatter.format_result(
            sample_id,
            ranked_candidates,
            uncertainty=uncertainty_metrics,
            metadata=metadata
        )

        expected_dict = {
            "sample_id": "Sample_002",
            "phasing_results": [
                {
                    "rank": 1,
                    "haplotype1": "C*01:01",
                    "haplotype2": "C*02:02",
                    "score": 0.95,
                    "uncertainty": {"entropy": 0.15, "confidence": 0.9}
                }
            ],
            "metadata": {"run_id": "run_abc", "model_version": "1.1"}
        }
        self.assertEqual(json.loads(formatted_output), expected_dict)

    def test_format_limits_candidates(self):
        """Test that formatting respects the num_candidates limit."""
        formatter = PhasingResultFormatter(tokenizer=self.tokenizer, num_candidates=1)
        ranked_candidates = [
            (('A*01:01', 'A*02:01'), -0.1),
            (('A*01:01', 'A*01:01'), -0.5), # Lower score
        ]
        sample_id = "Sample_003"
        formatted_output = formatter.format_result(sample_id, ranked_candidates)
        result_dict = json.loads(formatted_output)
        self.assertEqual(len(result_dict["phasing_results"]), 1) # Only top 1 candidate
        self.assertEqual(result_dict["phasing_results"][0]["haplotype1"], "A*01:01")
        self.assertEqual(result_dict["phasing_results"][0]["haplotype2"], "A*02:01")


if __name__ == '__main__':
    unittest.main()
