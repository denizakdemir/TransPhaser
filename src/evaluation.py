import torch
import numpy as np
import logging
import torch.nn as nn # Added for type hint

# Placeholder import
# from .data_preprocessing import AlleleTokenizer

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
                       tokens back to alleles if needed for certain metrics.
        """
        # Need to import AlleleTokenizer if not done globally
        # from .data_preprocessing import AlleleTokenizer
        # if not isinstance(tokenizer, AlleleTokenizer):
        #     raise TypeError("tokenizer must be an instance of AlleleTokenizer")

        self.tokenizer = tokenizer
        # Removed placeholder print

    def calculate_metrics(self, predicted_haplotypes, true_haplotypes):
        """
        Calculates phasing metrics by comparing predicted and true haplotypes.

        Args:
            predicted_haplotypes (list or np.ndarray or torch.Tensor):
                Predicted haplotype pairs. Format depends on model output,
                e.g., list of tuples of allele strings, or tensor of token IDs.
                Example structure: [[(pred_h1_s1, pred_h2_s1)], [(pred_h1_s2, pred_h2_s2)], ...]
                where each inner tuple represents the pair for one locus.
                Needs to handle multiple loci per sample.
            true_haplotypes (list or np.ndarray or torch.Tensor):
                Ground truth haplotype pairs in the same format as predicted_haplotypes.

        Returns:
            dict: A dictionary containing calculated metrics, e.g.,
                  {'phasing_accuracy': ...} # Add other metrics later
        """
        # Removed placeholder print
        # Actual implementation would involve:
        # 1. Aligning predicted and true haplotypes (handling potential phase flips).
        # 2. Calculating Hamming distance (mismatched alleles).
        # 3. Calculating Switch Error Rate (number of switches needed to match true phase).
        # Actual implementation would involve:
        # 1. Aligning predicted and true haplotypes (handling potential phase flips).
        # 2. Calculating Hamming distance (mismatched alleles).
        # 3. Calculating Switch Error Rate (number of switches needed to match true phase).

        if len(predicted_haplotypes) != len(true_haplotypes):
            raise ValueError("Number of predicted and true haplotype pairs must match.")

        num_samples = len(predicted_haplotypes)
        if num_samples == 0:
            return {'phasing_accuracy': 0.0} # Or handle as appropriate

        correct_phases = 0
        for i in range(num_samples):
            pred_h1, pred_h2 = predicted_haplotypes[i]
            true_h1, true_h2 = true_haplotypes[i]

            # Check for match in either order
            match_order1 = (pred_h1 == true_h1 and pred_h2 == true_h2)
            match_order2 = (pred_h1 == true_h2 and pred_h2 == true_h1)

            if match_order1 or match_order2:
                correct_phases += 1

        accuracy = correct_phases / num_samples

        # TODO: Implement Hamming distance and Switch Error Rate later
        return {'phasing_accuracy': accuracy}

    # Helper methods for specific metrics can be added here
    # def _calculate_hamming(self, pred_pair, true_pair): pass
    # def _calculate_switch_error(self, ...): pass


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
            diversity_weight (float): Weight for promoting diversity among top candidates.
                                      (Specific mechanism TBD). Defaults to 0.1.
        """
        if not isinstance(model, nn.Module): # Basic check
             raise TypeError("model must be a PyTorch nn.Module.")
        if not isinstance(num_candidates, int) or num_candidates <= 0:
             raise ValueError("num_candidates must be a positive integer.")
        if not isinstance(diversity_weight, (int, float)) or diversity_weight < 0:
             raise ValueError("diversity_weight must be a non-negative number.")

        self.model = model
        self.num_candidates = num_candidates
        self.diversity_weight = diversity_weight
        # Removed placeholder print (already removed, ensuring no regression)

    @torch.no_grad()
    def rank_candidates(self, batch, candidate_haplotypes):
        """
        Ranks provided candidate haplotype pairs for a batch of genotypes.

        Args:
            batch (dict): Batch dictionary containing input data (genotypes, covariates).
            candidate_haplotypes (list): A list where each element corresponds to a sample
                                         in the batch. Each element is itself a list of
                                         candidate haplotype pairs (e.g., tuples of allele strings
                                         or token ID tensors) for that sample.

        Returns:
            list: A list (one per sample) of ranked candidate haplotype pairs,
                  potentially with associated scores. Structure TBD.
                  Example: [[(hap_pair1, score1), (hap_pair2, score2), ...], ...]
        """
        # Removed placeholder print
        # Actual implementation would involve:
        # 1. Using the model (e.g., decoder or full model) to calculate a score
        #    (e.g., log probability log p(h|c) or log p(h|g,c)) for each candidate pair.
        # 2. Potentially applying diversity promotion techniques (e.g., penalizing
        #    candidates similar to already selected top candidates).
        # 3. Sorting candidates based on the final score.
        # 4. Returning the top 'self.num_candidates'.

        # Dummy return value: just return the first N candidates from input
        ranked_results = []
        for sample_candidates in candidate_haplotypes:
            # Assign dummy scores and take top N
            scored_candidates = [(cand, -i) for i, cand in enumerate(sample_candidates)] # Higher index = better score (dummy)
            ranked_sample = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
            ranked_results.append(ranked_sample[:self.num_candidates])

        return ranked_results


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
        # Need to import AlleleTokenizer if not done globally
        # from .data_preprocessing import AlleleTokenizer
        # if not isinstance(tokenizer, AlleleTokenizer):
        #     raise TypeError("tokenizer must be an instance of AlleleTokenizer")

        self.tokenizer = tokenizer
        # May need plotting libraries like matplotlib or seaborn
        # import matplotlib.pyplot as plt
        # Removed placeholder print (already removed, ensuring no regression)

    def plot_likelihoods(self, ranked_candidates, output_path=None):
        """
        Plots the likelihood/score distribution of ranked haplotype candidates.

        Args:
            ranked_candidates (list): Output from HaplotypeCandidateRanker, a list
                                      (per sample) of (hap_pair, score) tuples.
            output_path (str, optional): Path to save the plot. If None, might display it.
        """
        # Removed placeholder print
        # Actual implementation would use matplotlib/seaborn to create bar charts or distributions.

    def plot_uncertainty(self, uncertainty_estimates, output_path=None):
        """
        Plots the estimated uncertainty for samples or loci.

        Args:
            uncertainty_estimates (dict or torch.Tensor): Output from PhasingUncertaintyEstimator.
            output_path (str, optional): Path to save the plot.
        """
        # Removed placeholder print

    def visualize_alignment(self, genotype, predicted_haplotypes, true_haplotypes=None, output_path=None):
        """
        Creates a visual alignment of genotype and predicted/true haplotypes.

        Args:
            genotype: The input genotype data for a sample.
            predicted_haplotypes: The predicted haplotype pair(s) for the sample.
            true_haplotypes: Optional ground truth haplotype pair.
            output_path (str, optional): Path to save the visualization.
        """
        # Removed placeholder print
