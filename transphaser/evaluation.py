
import torch
import numpy as np
import logging
import torch.nn as nn # Added import for nn.Module
import os # Import os for path joining
from typing import Optional, List, Tuple, Any, Dict # Import Dict, List, Tuple, Any

# Import necessary components
from transphaser.data_preprocessing import AlleleTokenizer # Import AlleleTokenizer

# Attempt to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Log warning here instead of inside the class?
    # logging.warning("matplotlib not found. Plotting functionality will be disabled.")

def normalize_haplotype(hap):
    """Convert haplotype to string format.
    
    Accepts both list and string formats:
    - List: ['A*01:01', 'B*07:02', 'DRB1*03:01']
    - String: 'A*01:01_B*07:02_DRB1*03:01'
    
    Returns:
        str: Haplotype in string format with underscores.
    """
    if isinstance(hap, list):
        return '_'.join(hap)
    return hap


class HLAPhasingMetrics:
    """
    Calculates various metrics for evaluating HLA phasing performance,
    such as Hamming distance and switch error rate.
    """
    def __init__(self, tokenizer):
        """
        Initializes the HLAPhasingMetrics calculator.

        Args:
            tokenizer: An initialized AlleleTokenizer instance, used for mapping
                       tokens back to alleles if needed for certain metrics (currently not used).
        """
        # Uncommented and corrected the type check
        if not isinstance(tokenizer, AlleleTokenizer):
             raise TypeError("tokenizer must be an instance of AlleleTokenizer")

        self.tokenizer = tokenizer
        logging.debug("HLAPhasingMetrics initialized.")

    def _calculate_hamming(self, hap1_str: str, hap2_str: str) -> int:
        """Calculates Hamming distance between two haplotype strings."""
        # Assumes haplotypes are strings like "A*01:01_B*07:02_C*01:02"
        alleles1 = hap1_str.split('_')
        alleles2 = hap2_str.split('_')
        if len(alleles1) != len(alleles2):
            logging.warning(f"Haplotype lengths differ for Hamming calculation: {len(alleles1)} vs {len(alleles2)}. Returning high distance.")
            # Return a high distance or handle based on requirements
            return len(alleles1) # Max possible distance if lengths differ significantly?

        distance = 0
        for a1, a2 in zip(alleles1, alleles2):
            if a1 != a2:
                distance += 1
        return distance

    def _calculate_switch_error(self, pred_h1: str, true_h1: str, pred_h2: str, true_h2: str) -> int:
        """
        Calculates the minimum number of switches needed to align predicted haplotypes
        to the true haplotypes for a single alignment possibility.
        Assumes pred_h1 aligns with true_h1, and pred_h2 with true_h2.
        """
        alleles_pred_h1 = pred_h1.split('_')
        alleles_true_h1 = true_h1.split('_')
        alleles_pred_h2 = pred_h2.split('_')
        alleles_true_h2 = true_h2.split('_')

        num_loci = len(alleles_true_h1)
        if not (len(alleles_pred_h1) == num_loci and len(alleles_pred_h2) == num_loci and len(alleles_true_h2) == num_loci):
             logging.warning("Haplotype lengths differ for Switch Error calculation. Returning high error count.")
             return num_loci - 1  # Max possible switches

        # Calculate switch errors based on phase state transitions
        # State 0: pred_h1[i] == true_h1[i] (and pred_h2[i] == true_h2[i])
        # State 1: pred_h1[i] == true_h2[i] (and pred_h2[i] == true_h1[i])
        # State -1: Mismatch (neither matches)
        switches = 0
        # Determine initial phase state (match = 0, flip = 1, mismatch = -1)
        if alleles_pred_h1[0] == alleles_true_h1[0] and alleles_pred_h2[0] == alleles_true_h2[0]:
            last_phase_state = 0
        elif alleles_pred_h1[0] == alleles_true_h2[0] and alleles_pred_h2[0] == alleles_true_h1[0]:
            last_phase_state = 1
        else:
            last_phase_state = -1 # Mismatch at first locus

        for i in range(1, num_loci):
            # Determine current phase state
            if alleles_pred_h1[i] == alleles_true_h1[i] and alleles_pred_h2[i] == alleles_true_h2[i]:
                current_phase_state = 0
            elif alleles_pred_h1[i] == alleles_true_h2[i] and alleles_pred_h2[i] == alleles_true_h1[i]:
                current_phase_state = 1
            else:
                current_phase_state = -1 # Mismatch

            # Count switch if phase state changes between non-mismatched states
            if current_phase_state != -1 and last_phase_state != -1 and current_phase_state != last_phase_state:
                switches += 1

            # Update last state only if current wasn't a mismatch
            if current_phase_state != -1:
                last_phase_state = current_phase_state
            # If current is mismatch, keep last_phase_state to compare against the next non-mismatch

        return switches


    def calculate_metrics(self, predicted_haplotypes, true_haplotypes):
        """
        Calculates phasing metrics by comparing predicted and true haplotypes.

        Args:
            predicted_haplotypes (list):
                Predicted haplotype pairs.
                Example: [('A*01:01_B*07:02', 'A*02:01_B*08:01'), ...] where each element is a tuple
                         representing the sorted pair of haplotype strings for one sample.
            true_haplotypes (list):
                Ground truth haplotype pairs in the same format as predicted_haplotypes.

        Returns:
            dict: A dictionary containing calculated metrics, e.g.,
                  {'phasing_accuracy': float,
                   'avg_hamming_distance': float,
                   'avg_switch_errors': float}
        """
        if len(predicted_haplotypes) != len(true_haplotypes):
            raise ValueError("Number of predicted and true haplotype pairs must match.")

        num_samples = len(predicted_haplotypes)
        if num_samples == 0:
            logging.warning("Received empty lists for metric calculation.")
            return {'phasing_accuracy': 0.0, 'avg_hamming_distance': 0.0, 'avg_switch_errors': 0.0}

        correct_phases = 0
        total_hamming_dist = 0
        total_switch_errors = 0

        for i in range(num_samples):
            # Assumes input is already sorted tuples of strings like ('HapA_HapB', 'HapC_HapD')
            pred_pair = predicted_haplotypes[i]
            true_pair = true_haplotypes[i]

            # Ensure pairs are tuples and sorted (defensive check)
            if not isinstance(pred_pair, tuple) or len(pred_pair) != 2:
                 logging.warning(f"Sample {i}: Predicted data is not a tuple of 2 haplotypes: {pred_pair}. Skipping sample.")
                 continue
            if not isinstance(true_pair, tuple) or len(true_pair) != 2:
                 logging.warning(f"Sample {i}: True data is not a tuple of 2 haplotypes: {true_pair}. Skipping sample.")
                 continue

            # Normalize haplotypes to string format (handles both list and string inputs)
            pred_pair = tuple(normalize_haplotype(h) for h in pred_pair)
            true_pair = tuple(normalize_haplotype(h) for h in true_pair)
            
            pred_pair = tuple(sorted(pred_pair))
            true_pair = tuple(sorted(true_pair))

            pred_h1, pred_h2 = pred_pair
            true_h1, true_h2 = true_pair

            # --- Phasing Accuracy ---
            if pred_pair == true_pair:
                 correct_phases += 1
                 min_hamming = 0
                 min_switches = 0
            else:
                 # If not perfectly matched, calculate Hamming and Switch Error for both alignments
                 try:
                      # Alignment 1: pred_h1 vs true_h1, pred_h2 vs true_h2
                      hamming1 = self._calculate_hamming(pred_h1, true_h1) + self._calculate_hamming(pred_h2, true_h2)
                      switches1 = self._calculate_switch_error(pred_h1, true_h1, pred_h2, true_h2)

                      # Alignment 2: pred_h1 vs true_h2, pred_h2 vs true_h1
                      hamming2 = self._calculate_hamming(pred_h1, true_h2) + self._calculate_hamming(pred_h2, true_h1)
                      switches2 = self._calculate_switch_error(pred_h1, true_h2, pred_h2, true_h1)

                      # Choose the alignment with the minimum switch errors
                      # If switch errors are equal, choose the one with minimum Hamming distance
                      if switches1 < switches2:
                           min_switches = switches1
                           min_hamming = hamming1
                      elif switches2 < switches1:
                           min_switches = switches2
                           min_hamming = hamming2
                      else: # Equal switches, choose min Hamming
                           min_switches = switches1 # or switches2
                           min_hamming = min(hamming1, hamming2)

                 except Exception as e:
                      logging.error(f"Error calculating Hamming/Switch for sample {i}: {e}. Skipping sample metrics.")
                      min_hamming = float('nan') # Indicate error
                      min_switches = float('nan')


            # Accumulate only if calculations were successful
            if not np.isnan(min_hamming):
                 total_hamming_dist += min_hamming
            if not np.isnan(min_switches):
                 total_switch_errors += min_switches


        # Calculate averages, avoiding division by zero
        accuracy = correct_phases / num_samples if num_samples > 0 else 0.0
        # Calculate average over samples where calculation didn't fail (count non-NaNs if needed, simpler for now)
        avg_hamming_dist = total_hamming_dist / num_samples if num_samples > 0 else 0.0
        avg_switch_errors = total_switch_errors / num_samples if num_samples > 0 else 0.0


        return {
            'phasing_accuracy': accuracy,
            'avg_hamming_distance': avg_hamming_dist,
            'avg_switch_errors': avg_switch_errors
        }
