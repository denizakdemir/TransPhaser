import json
from typing import List, Tuple, Any, Dict, Optional

# Assuming AlleleTokenizer is available for type hinting
# from .data_preprocessing import AlleleTokenizer

class PhasingResultFormatter:
    """
    Formats phasing results, including ranked haplotype candidates, scores,
    uncertainty metrics, and metadata, into a structured JSON output.
    """
    def __init__(self, tokenizer: Any, num_candidates: int = 5):
        """
        Initializes the PhasingResultFormatter.

        Args:
            tokenizer: An initialized AlleleTokenizer instance (or mock).
                       Used for potential future allele validation or info lookup.
            num_candidates (int): The maximum number of candidate pairs to include
                                  in the formatted output. Defaults to 5.
        """
        if not hasattr(tokenizer, 'tokenize') or not hasattr(tokenizer, 'detokenize'):
             # Basic check for tokenizer-like object, replace Any with actual type hint if possible
             raise TypeError("tokenizer must be an AlleleTokenizer-like object.")
        if not isinstance(num_candidates, int) or num_candidates <= 0:
            raise ValueError("num_candidates must be a positive integer.")

        self.tokenizer = tokenizer
        self.num_candidates = num_candidates

    def format_result(self,
                      sample_id: str,
                      ranked_candidates: List[Tuple[Tuple[str, str], float]],
                      uncertainty: Optional[Dict[str, Any]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Formats the phasing result for a single sample into a JSON string.

        Args:
            sample_id (str): The identifier for the sample.
            ranked_candidates (List[Tuple[Tuple[str, str], float]]): A list of
                ranked candidate haplotype pairs and their scores. Each element is
                a tuple: ((haplotype1_str, haplotype2_str), score). The list
                should be pre-sorted with the best candidate first.
            uncertainty (Optional[Dict[str, Any]]): Optional dictionary containing
                uncertainty metrics (e.g., {'entropy': 0.1, 'confidence': 0.9}).
                Defaults to None.
            metadata (Optional[Dict[str, Any]]): Optional dictionary containing
                metadata (e.g., {'run_id': 'xyz', 'model_version': '1.0'}).
                Defaults to None.

        Returns:
            str: A JSON string representing the formatted result.
        """
        if not isinstance(sample_id, str):
            raise TypeError("sample_id must be a string.")
        if not isinstance(ranked_candidates, list):
            raise TypeError("ranked_candidates must be a list.")

        output_dict = {
            "sample_id": sample_id,
            "phasing_results": [],
            "metadata": metadata if metadata is not None else {}
        }

        # Limit the number of candidates based on self.num_candidates
        candidates_to_format = ranked_candidates[:self.num_candidates]

        for rank, (hap_pair, score) in enumerate(candidates_to_format, start=1):
            hap1_str, hap2_str = hap_pair
            result_entry = {
                "rank": rank,
                "haplotype1": hap1_str,
                "haplotype2": hap2_str,
                "score": score,
                # Include uncertainty if provided, otherwise None or omit
                "uncertainty": uncertainty # Include the whole dict if present
            }
            output_dict["phasing_results"].append(result_entry)

        # Convert the dictionary to a JSON string
        try:
            # Use separators=(',', ':') for compact JSON matching test expectations
            # Use sort_keys=True to ensure consistent key order for comparison
            json_output = json.dumps(output_dict, indent=None, separators=(',', ':'), sort_keys=True)

            # Reload and dump again with indent=4 for better readability if needed,
            # but return the compact version for programmatic use / testing.
            # For consistency with test which compares dicts, maybe return dict?
            # Let's return the JSON string as per initial plan.
            # Re-parse and dump with indentation to ensure consistent key order from sort_keys
            # This is a bit redundant but ensures the output matches the test's expected dict structure order
            parsed_dict = json.loads(json_output)
            return json.dumps(parsed_dict, indent=None, separators=(',', ':'), sort_keys=True)


        except TypeError as e:
            raise TypeError(f"Error serializing result to JSON: {e}. Check data types.")

    # Potential future method to format multiple results
    # def format_batch(self, batch_results: List[Dict]) -> str:
    #     pass
