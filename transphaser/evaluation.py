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
             return num_loci -1 # Max possible switches

        switches = 0
        current_phase_correct = True # Assume starting phase is correct

        for i in range(num_loci):
            # Check if the current phase matches the true phase at this locus
            locus_phase_correct = (alleles_pred_h1[i] == alleles_true_h1[i] and alleles_pred_h2[i] == alleles_true_h2[i])

            # If the locus phase is incorrect, we might need a switch
            if not locus_phase_correct:
                # Check if the alternative phasing (swapped prediction) matches the truth
                alternative_phase_correct = (alleles_pred_h1[i] == alleles_true_h2[i] and alleles_pred_h2[i] == alleles_true_h1[i])

                if alternative_phase_correct:
                    # The phase is flipped compared to the current alignment assumption
                    if current_phase_correct:
                        # We were assuming correct phase, but it's flipped now -> switch needed
                        if i > 0: # Don't count switch at the very beginning
                             switches += 1
                        current_phase_correct = False # Now assuming flipped phase
                else:
                    # Neither direct nor alternative phase matches - this indicates an allele mismatch (Hamming error)
                    # Switch error doesn't count allele mismatches directly, but phase consistency.
                    # If the current phase assumption was wrong, we need a switch.
                    if not current_phase_correct and i > 0:
                         switches += 1
                         current_phase_correct = True # Assume it switched back to correct due to mismatch? Or keep assumption?
                         # Let's assume a mismatch forces a re-evaluation, potentially switching back.
                         # This logic can be complex; simpler versions might just count transitions
                         # between matching and non-matching states relative to one alignment.

                    # Alternative simpler logic: Count switches between phase states (match vs mismatch)
                    # This might be closer to standard definitions. Let's try that.

                    pass # Keep current_phase_correct as is, mismatch doesn't flip assumption here.


            # More standard switch error logic:
            # Compare the phase state at locus i vs i-1
            # State 0: pred_h1[i] == true_h1[i] (and pred_h2[i] == true_h2[i])
            # State 1: pred_h1[i] == true_h2[i] (and pred_h2[i] == true_h1[i])
            # State -1: Mismatch (neither matches)

        # --- Reimplementing Switch Error based on phase state transitions ---
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


class PhasingUncertaintyEstimator:
    """
    Estimates uncertainty in phasing predictions.

    This can be done using various methods, such as:
    - Calculating the entropy of the posterior probability distribution over
      possible haplotype pairs, if the model provides it.
    - Sampling multiple haplotype pairs from the model's posterior distribution
      and calculating statistics like variance or frequency of the most likely pair.
    """
    def __init__(self, model: nn.Module, sampling_iterations=100):
        """
        Initializes the PhasingUncertaintyEstimator.

        Args:
            model: The trained phasing model (e.g., HLAPhasingModel). It's expected
                   that this model can provide access to the posterior distribution
                   or allow sampling from it.
            sampling_iterations (int): Number of samples to draw if using Monte Carlo
                                       methods for uncertainty estimation. Defaults to 100.
        """
        if not isinstance(model, nn.Module): # Basic check
             raise TypeError("model must be a PyTorch nn.Module.")
        if not isinstance(sampling_iterations, int) or sampling_iterations <= 0:
             raise ValueError("sampling_iterations must be a positive integer.")

        self.model = model
        self.sampling_iterations = sampling_iterations
        # Note: sampling_iterations is not used in the current entropy-based method
        logging.info(f"PhasingUncertaintyEstimator initialized. Using prediction entropy method.")

    @torch.no_grad()
    def estimate_uncertainty(self, batch):
        """
        Estimates phasing uncertainty based on the entropy of the predicted
        probability distribution at each step of the autoregressive generation.

        Args:
            batch (dict): A batch dictionary containing input data required by the model
                          (e.g., 'genotype_tokens', 'covariates').

        Returns:
            dict: A dictionary containing uncertainty estimates. Currently includes:
                  {'mean_prediction_entropy': tensor([batch_size])} - Average entropy
                   across prediction steps for each sample. Higher values indicate
                   more uncertainty during generation.
        """
        logging.debug("Estimating phasing uncertainty using prediction entropy...")
        self.model.eval() # Ensure model is in evaluation mode
        uncertainty_estimates = {}
        batch_size = batch.get('genotype_tokens', torch.empty(0)).shape[0]
        device = next(self.model.parameters()).device # Get model device

        try:
            # Call predict_haplotypes to get the sequence of logits
            # Ensure the batch tensors are on the correct device
            predict_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _, step_logits_list = self.model.predict_haplotypes(predict_batch, return_logits=True)

            if not step_logits_list:
                logging.warning("predict_haplotypes did not return logits. Cannot calculate entropy.")
                mean_entropy = torch.full((batch_size,), float('nan'), device=device)
            else:
                entropies_per_step = []
                for step_logits in step_logits_list:
                    # step_logits shape: (batch_size, vocab_size)
                    # Masked logits are already -inf for invalid tokens

                    # Check if all valid logits are -inf for any sample
                    valid_logits_mask = step_logits > -float('inf') # Mask of valid logits
                    all_valid_inf = ~torch.any(valid_logits_mask, dim=-1) # True if all valid logits are -inf

                    # Calculate probabilities using softmax
                    # Add a small epsilon to prevent log(0)
                    probs = torch.softmax(step_logits, dim=-1)
                    log_probs = torch.log(probs + 1e-9)

                    # Calculate entropy: -sum(p * log(p))
                    entropy = -torch.sum(probs * log_probs, dim=-1) # Shape: (batch_size,)

                    # Handle cases where entropy is NaN (due to all valid logits being -inf)
                    # Set entropy to 0.0 in these cases, representing certainty in an impossible prediction
                    entropy[all_valid_inf] = 0.0
                    # Also handle any other potential NaNs by setting them to 0.0
                    entropy[torch.isnan(entropy)] = 0.0

                    entropies_per_step.append(entropy)

                # Stack entropies across steps and calculate mean per sample
                if entropies_per_step:
                    all_entropies = torch.stack(entropies_per_step, dim=0) # Shape: (num_steps, batch_size)
                    # Use regular mean now, as NaNs should have been replaced by 0.0
                    mean_entropy = torch.mean(all_entropies, dim=0) # Shape: (batch_size,)
                else:
                     # Should not happen if step_logits_list was not empty, but handle defensively
                     logging.warning("No step entropies were calculated.")
                     mean_entropy = torch.full((batch_size,), float('nan'), device=device)

            uncertainty_estimates['mean_prediction_entropy'] = mean_entropy

        except AttributeError as e:
             logging.error(f"Model does not support 'return_logits' in predict_haplotypes or other attribute error: {e}")
             uncertainty_estimates['mean_prediction_entropy'] = torch.full((batch_size,), float('nan'), device=device)
        except Exception as e:
            logging.error(f"An error occurred during uncertainty estimation: {e}", exc_info=True)
            uncertainty_estimates['mean_prediction_entropy'] = torch.full((batch_size,), float('nan'), device=device)


        return uncertainty_estimates


class HaplotypeCandidateRanker:
    """
    Ranks candidate haplotype pairs based on likelihood or probability scores
    obtained from the model. Can incorporate diversity weighting.
    """
    def __init__(self, model: nn.Module, num_candidates=10, diversity_weight=0.1):
        """
        Initializes the HaplotypeCandidateRanker.

        Args:
            model: The trained model (e.g., HLAPhasingModel) capable of scoring haplotypes.
            num_candidates (int): The maximum number of candidate pairs to return. Defaults to 10.
            diversity_weight (float): Weight for promoting diversity among top candidates (lambda in MMR).
                                      Defaults to 0.1.
        """
        if not isinstance(model, nn.Module): # Basic check
             raise TypeError("model must be a PyTorch nn.Module.")
        if not isinstance(num_candidates, int) or num_candidates <= 0:
             raise ValueError("num_candidates must be a positive integer.")
        if not isinstance(diversity_weight, (int, float)) or diversity_weight < 0:
             raise ValueError("diversity_weight must be a non-negative number.")

        self.model = model
        self.num_candidates = num_candidates
        self.diversity_weight = diversity_weight # This is lambda in MMR
        logging.debug("HaplotypeCandidateRanker initialized.")

    def _calculate_pair_similarity(self, pair1: Tuple[str, str], pair2: Tuple[str, str]) -> float:
        """
        Calculates similarity between two haplotype pairs.
        Currently uses inverse Hamming distance on concatenated strings.
        Higher value means more similar.
        Assumes pairs are tuples of strings: (hap1_str, hap2_str).
        """
        # Simple similarity: 1 / (1 + Hamming distance between concatenated strings)
        # This is a basic metric; more sophisticated ones could be used.
        hap1_a, hap2_a = pair1
        hap1_b, hap2_b = pair2

        # Concatenate for a single distance measure (simplistic)
        # A better approach might average distances or consider loci separately.
        str_a = hap1_a + "_" + hap2_a
        str_b = hap1_b + "_" + hap2_b

        alleles_a = str_a.split('_')
        alleles_b = str_b.split('_')

        if len(alleles_a) != len(alleles_b):
            return 0.0 # Cannot compare if lengths differ

        distance = sum(1 for a, b in zip(alleles_a, alleles_b) if a != b)
        similarity = 1.0 / (1.0 + float(distance)) # Ensure float division
        return similarity


    @torch.no_grad()
    def rank_candidates(self, batch: Dict[str, Any], candidate_haplotypes: List[List[Tuple[str, str]]]) -> List[List[Tuple[Tuple[str, str], torch.Tensor]]]:
        """
        Ranks provided candidate haplotype pairs for a batch of genotypes,
        optionally applying Maximal Marginal Relevance (MMR) for diversity.

        Args:
            batch (dict): Batch dictionary containing input data (genotypes, covariates).
                          Content might be used by the model's scoring function.
            candidate_haplotypes (list): A list where each element corresponds to a sample
                                         in the batch. Each element is itself a list of
                                         candidate haplotype pairs (tuples of allele strings).

        Returns:
            list: A list (one per sample) of ranked candidate haplotype pairs,
                  each element being a tuple: ((hap1_str, hap2_str), score_tensor).
        """
        self.model.eval() # Ensure model is in eval mode
        batch_size = len(candidate_haplotypes) # Assume outer list length is batch size
        ranked_results_batch = []

        if not hasattr(self.model, 'score_haplotype_pair'):
             raise AttributeError("Model must have a 'score_haplotype_pair' method for ranking.")

        for i in range(batch_size):
            sample_candidates = candidate_haplotypes[i]
            if not sample_candidates:
                ranked_results_batch.append([]) # No candidates for this sample
                continue

            # Extract necessary info for this sample from the batch if needed by the model's scorer
            # For now, pass the whole batch, assuming the scorer handles indexing if needed.
            batch_sample_info = batch

            scored_candidates = []
            for candidate_pair in sample_candidates:
                try:
                    # Score the pair using the model
                    score = self.model.score_haplotype_pair(batch_sample_info, candidate_pair)
                    # Ensure score is on CPU for sorting if it's a tensor
                    scored_candidates.append((candidate_pair, score.cpu()))
                except Exception as e:
                     logging.error(f"Error scoring candidate pair {candidate_pair} for sample {i}: {e}", exc_info=True)
                     # Assign a very low score or skip? Assign low score for now.
                     scored_candidates.append((candidate_pair, torch.tensor(float('-inf'))))


            # Sort candidates by original score (descending)
            # Use .item() if scores are scalar tensors
            scored_candidates.sort(key=lambda x: x[1].item() if isinstance(x[1], torch.Tensor) else x[1], reverse=True)

            # Apply MMR if diversity_weight > 0 and there's more than one candidate
            if self.diversity_weight > 0 and len(scored_candidates) > 1:
                lambda_val = self.diversity_weight # MMR lambda
                ranked_mmr = []
                candidates_pool = scored_candidates.copy()

                # Select the first candidate (highest original score)
                best_initial = candidates_pool.pop(0)
                ranked_mmr.append(best_initial)

                while len(ranked_mmr) < self.num_candidates and candidates_pool:
                    mmr_scores = []
                    for candidate_idx, (candidate_pair, score) in enumerate(candidates_pool):
                        max_similarity = 0.0
                        # Calculate max similarity to already selected candidates
                        for selected_pair, _ in ranked_mmr:
                            similarity = self._calculate_pair_similarity(candidate_pair, selected_pair)
                            max_similarity = max(max_similarity, similarity)

                        # MMR formula: lambda * score - (1 - lambda) * max_similarity
                        # Note: Assuming higher score is better (e.g., log probability)
                        # Ensure score is float for calculation
                        score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                        mmr_score = lambda_val * score_val - (1 - lambda_val) * max_similarity
                        mmr_scores.append((mmr_score, candidate_idx)) # Store score and original index

                    # Find the candidate with the highest MMR score
                    mmr_scores.sort(key=lambda x: x[0], reverse=True)
                    best_mmr_idx_in_pool = mmr_scores[0][1]

                    # Add the best MMR candidate to the ranked list and remove from pool
                    ranked_mmr.append(candidates_pool.pop(best_mmr_idx_in_pool))

                # Replace original sorted list with MMR ranked list
                top_n_candidates = ranked_mmr[:self.num_candidates] # Ensure we don't exceed num_candidates

            else:
                # No diversity applied, just take top N from original sorting
                top_n_candidates = scored_candidates[:self.num_candidates]

            ranked_results_batch.append(top_n_candidates)

        return ranked_results_batch


class PhasingResultVisualizer:
    """
    Generates visualizations for phasing results, such as likelihood distributions,
    uncertainty plots, or haplotype alignments.
    """
    def __init__(self, tokenizer):
        """
        Initializes the PhasingResultVisualizer.

        Args:
            tokenizer: An initialized AlleleTokenizer instance.
        """
        # Uncommented and corrected the type check
        if not isinstance(tokenizer, AlleleTokenizer):
             raise TypeError("tokenizer must be an instance of AlleleTokenizer")

        self.tokenizer = tokenizer
        # Store output_dir, needed for saving plots
        # Assuming visualizer might be created independently or needs output dir
        # Add output_dir to __init__? For now, assume it's handled externally or passed to methods.
        # Let's add it to __init__ for consistency with Reporter
        self.output_dir = "visualizer_output" # Default or get from config? Needs refinement.
        os.makedirs(self.output_dir, exist_ok=True)

        # Use the module-level flag and import
        self.plt = plt if MATPLOTLIB_AVAILABLE else None
        if not MATPLOTLIB_AVAILABLE:
             logging.warning("Matplotlib not found. Plotting functions in PhasingResultVisualizer will be disabled.")

        logging.debug("PhasingResultVisualizer initialized.")

    def plot_likelihoods(self, ranked_candidates, output_path: Optional[str] = None):
        """
        Plots the likelihood/score distribution of ranked haplotype candidates for the first sample.

        Args:
            ranked_candidates (list): Output from HaplotypeCandidateRanker, a list
                                      (per sample) of (hap_pair, score) tuples.
            output_path (str, optional): Full path to save the plot. If None, the plot is not saved.
        """
        if not self.plt:
            logging.warning("Matplotlib not available. Skipping plot_likelihoods.")
            return

        if not ranked_candidates or not ranked_candidates[0]:
            logging.warning("No ranked candidates provided for the first sample. Skipping plot_likelihoods.")
            return

        # Plot only for the first sample for simplicity
        scores = [score.item() for _, score in ranked_candidates[0]]
        num_candidates = len(scores)
        ranks = range(1, num_candidates + 1) # Ranks from 1 to N

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))
            ax.bar(ranks, scores, color='skyblue')
            ax.set_title("Candidate Haplotype Scores (First Sample)")
            ax.set_xlabel("Candidate Rank")
            ax.set_ylabel("Score (Log Likelihood or similar)")
            ax.set_xticks(ranks) # Ensure ticks match ranks
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            if output_path:
                # Ensure the directory exists before saving
                output_dir = os.path.dirname(output_path)
                if output_dir: # Handle cases where output_path might be just a filename
                     os.makedirs(output_dir, exist_ok=True)
                fig.savefig(output_path)
                logging.info(f"Likelihood plot saved to {output_path}")
            else:
                logging.info("Likelihood plot generated but not saved (no output_path provided).")

            self.plt.close(fig) # Close the figure to free memory

        except Exception as e:
             logging.error(f"Error during likelihood plotting: {e}", exc_info=True)
             # Ensure plot is closed if error occurs after figure creation
             if 'fig' in locals() and self.plt:
                 self.plt.close(fig)


    def plot_uncertainty(self, uncertainty_estimates, output_path: Optional[str] = None, bins: int = 20):
        """
        Plots a histogram of the estimated uncertainty (e.g., mean prediction entropy).

        Args:
            uncertainty_estimates (dict): Output from PhasingUncertaintyEstimator, expected
                                          to contain 'mean_prediction_entropy'.
            output_path (str, optional): Full path to save the plot. If None, plot is not saved.
            bins (int): Number of bins for the histogram. Defaults to 20.
        """
        if not self.plt:
            logging.warning("Matplotlib not available. Skipping plot_uncertainty.")
            return

        if not isinstance(uncertainty_estimates, dict) or 'mean_prediction_entropy' not in uncertainty_estimates:
            logging.warning("Invalid or missing 'mean_prediction_entropy' in uncertainty_estimates. Skipping plot.")
            return

        entropies_tensor = uncertainty_estimates['mean_prediction_entropy']
        if not isinstance(entropies_tensor, torch.Tensor) or entropies_tensor.numel() == 0:
             logging.warning("Mean prediction entropy data is not a valid tensor or is empty. Skipping plot.")
             return

        # Convert to numpy, handling potential NaNs
        entropies = entropies_tensor.cpu().numpy()
        valid_entropies = entropies[~np.isnan(entropies)] # Filter out NaNs for plotting

        if valid_entropies.size == 0:
             logging.warning("No valid (non-NaN) entropy values found. Skipping plot.")
             return

        try:
            fig, ax = self.plt.subplots(figsize=(10, 6))
            ax.hist(valid_entropies, bins=bins, color='lightcoral', edgecolor='black')
            ax.set_title("Distribution of Phasing Uncertainty (Mean Prediction Entropy)")
            ax.set_xlabel("Mean Prediction Entropy")
            ax.set_ylabel("Frequency")
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            if output_path:
                # Ensure the directory exists before saving
                output_dir = os.path.dirname(output_path)
                if output_dir:
                     os.makedirs(output_dir, exist_ok=True)
                fig.savefig(output_path)
                logging.info(f"Uncertainty histogram saved to {output_path}")
            else:
                logging.info("Uncertainty histogram generated but not saved (no output_path provided).")

            self.plt.close(fig)

        except Exception as e:
             logging.error(f"Error during uncertainty plotting: {e}", exc_info=True)
             if 'fig' in locals() and self.plt:
                 self.plt.close(fig)


    def visualize_alignment(self, genotype_str_dict, predicted_pair, true_haplotypes=None, output_path=None):
        """
        Creates a text-based visual alignment of genotype and predicted/true haplotypes.

        Args:
            genotype_str_dict (dict): Dictionary mapping locus name to genotype string (e.g., "A*01/A*02").
            predicted_pair (tuple): Tuple of predicted haplotype strings (e.g., ('HapA_HapB', 'HapC_HapD')).
            true_haplotypes (tuple, optional): Tuple of ground truth haplotype strings. Defaults to None.
            output_path (str, optional): Full path to save the visualization. If None, not saved.
        """
        if not self.plt:
            logging.warning("Matplotlib not available. Skipping visualize_alignment.")
            return

        # --- Prepare Alignment Data ---
        loci = list(genotype_str_dict.keys()) # Infer loci from genotype dict
        pred_h1_alleles = predicted_pair[0].split('_')
        pred_h2_alleles = predicted_pair[1].split('_')

        if len(pred_h1_alleles) != len(loci) or len(pred_h2_alleles) != len(loci):
             logging.warning("Mismatch between number of loci in genotype and predicted haplotypes. Skipping alignment.")
             return

        alignment_lines = []
        header = f"{'Locus':<10} {'Genotype':<15} {'Pred H1':<15} {'Pred H2':<15}"
        if true_haplotypes:
            true_h1_alleles = true_haplotypes[0].split('_')
            true_h2_alleles = true_haplotypes[1].split('_')
            if len(true_h1_alleles) != len(loci) or len(true_h2_alleles) != len(loci):
                 logging.warning("Mismatch between number of loci in genotype and true haplotypes. Skipping true data in alignment.")
                 true_haplotypes = None # Disable true data display
            else:
                 header += f" {'True H1':<15} {'True H2':<15}"
        alignment_lines.append(header)
        alignment_lines.append("-" * len(header))

        for i, locus in enumerate(loci):
            geno = genotype_str_dict.get(locus, "N/A")
            p1 = pred_h1_alleles[i]
            p2 = pred_h2_alleles[i]
            line = f"{locus:<10} {geno:<15} {p1:<15} {p2:<15}"
            if true_haplotypes:
                t1 = true_h1_alleles[i]
                t2 = true_h2_alleles[i]
                # Add simple match indicator
                match_p1_t1 = "*" if p1 == t1 else " "
                match_p2_t2 = "*" if p2 == t2 else " "
                match_p1_t2 = "+" if p1 == t2 else " "
                match_p2_t1 = "+" if p2 == t1 else " "
                line += f" {t1:<15} {t2:<15}  ({match_p1_t1}{match_p2_t2} / {match_p1_t2}{match_p2_t1})"
            alignment_lines.append(line)

        alignment_text = "\n".join(alignment_lines)

        # --- Plotting ---
        try:
            # Estimate figure size needed based on lines and line length
            num_lines = len(alignment_lines)
            max_len = max(len(line) for line in alignment_lines)
            # Basic estimation, might need refinement
            fig_height = max(4, num_lines * 0.3)
            fig_width = max(8, max_len * 0.1)

            fig, ax = self.plt.subplots(figsize=(fig_width, fig_height))
            # Use monospace font for better alignment
            ax.text(0.01, 0.99, alignment_text, family='monospace', va='top', ha='left', wrap=False)
            ax.axis('off')
            ax.set_title("Haplotype Alignment")

            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:
                     os.makedirs(output_dir, exist_ok=True)
                fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1) # Use tight bbox
                logging.info(f"Alignment visualization saved to {output_path}")
            else:
                logging.info("Alignment visualization generated but not saved (no output_path provided).")

            self.plt.close(fig)

        except Exception as e:
             logging.error(f"Error during alignment visualization: {e}", exc_info=True)
             if 'fig' in locals() and self.plt:
                 self.plt.close(fig)
