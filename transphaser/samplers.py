import torch
import torch.nn.functional as F
import math
import logging # Added for logging

from transphaser.compatibility import HaplotypeCompatibilityChecker # Import HaplotypeCompatibilityChecker

class GumbelSoftmaxSampler:
    """
    Implements differentiable sampling from a categorical distribution using the
    Gumbel-Softmax trick (also known as the Concrete distribution).
    Includes temperature annealing.
    """
    def __init__(self, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.003):
        """
        Initializes the GumbelSoftmaxSampler.

        Args:
            initial_temperature (float): Starting temperature for Gumbel-Softmax.
            min_temperature (float): The minimum temperature to anneal down to.
            anneal_rate (float): The rate at which temperature decreases (exponential decay).
        """
        if not (isinstance(initial_temperature, (int, float)) and initial_temperature > 0):
            raise ValueError("initial_temperature must be a positive number.")
        if not (isinstance(min_temperature, (int, float)) and min_temperature > 0):
            raise ValueError("min_temperature must be a positive number.")
        if not (isinstance(anneal_rate, (int, float)) and anneal_rate > 0):
             raise ValueError("anneal_rate must be a positive number.")
        if min_temperature >= initial_temperature:
            raise ValueError("min_temperature must be less than initial_temperature.")

        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        self.current_temperature = initial_temperature
        self.step = 0
        logging.debug("GumbelSoftmaxSampler initialized.")

    def anneal_temperature(self):
        """Updates the current temperature based on the annealing schedule."""
        self.current_temperature = max(
            self.min_temperature,
            self.initial_temperature * math.exp(-self.anneal_rate * self.step)
        )
        self.step += 1
        # print(f"Step: {self.step}, Temp: {self.current_temperature:.4f}") # Optional logging

    def sample(self, logits, hard=False):
        """
        Samples from the Gumbel-Softmax distribution.

        Args:
            logits (torch.Tensor): Logits of the categorical distribution
                                   (unnormalized log probabilities). Shape (..., num_categories).
            hard (bool): If True, returns a one-hot vector by taking argmax in the
                         forward pass but uses the Gumbel-Softmax sample for gradients
                         (Straight-Through Gumbel-Softmax). If False, returns the
                         relaxed Gumbel-Softmax sample directly. Defaults to False.

        Returns:
            torch.Tensor: Sampled tensor. Shape (..., num_categories). If hard=True,
                          it's a one-hot tensor; otherwise, it's a relaxed sample.
        """
        logging.debug(f"Gumbel-Softmax sampling (hard={hard}). Temp={self.current_temperature:.4f}")
        # Use torch.nn.functional.gumbel_softmax
        y_soft = F.gumbel_softmax(logits, tau=self.current_temperature, hard=hard, eps=1e-10, dim=-1)

        # Straight-Through part if hard=True is handled by gumbel_softmax itself when hard=True
        return y_soft


class ConstrainedHaplotypeSampler:
    """
    Samples haplotype pairs ensuring they satisfy genotype constraints.
    Relies on a HaplotypeCompatibilityChecker.
    This is a conceptual placeholder - the actual implementation depends heavily
    on how haplotypes are represented (e.g., latent variables, sequences of tokens)
    and how constraints are applied during the sampling process (e.g., masking, rejection sampling).
    """
    def __init__(self, compatibility_checker):
        """
        Initializes the ConstrainedHaplotypeSampler.

        Args:
            compatibility_checker: An instance of HaplotypeCompatibilityChecker.
        """
        # Uncommented and corrected the type check
        if not isinstance(compatibility_checker, HaplotypeCompatibilityChecker):
             raise TypeError("compatibility_checker must be an instance of HaplotypeCompatibilityChecker")

        self.compatibility_checker = compatibility_checker
        logging.debug("ConstrainedHaplotypeSampler initialized.")

    def sample(self, genotype_info, num_samples=1, **kwargs):
        """
        Samples haplotype pairs that are compatible with the given genotype information.

        Args:
            genotype_info: Information about the genotype (e.g., allele strings, tokens).
                           The exact format depends on the overall model design.
            num_samples (int): Number of compatible pairs to sample.
            **kwargs: Additional arguments needed for the specific sampling strategy
                      (e.g., model logits, latent variables).

        Returns:
            list: A list of sampled compatible haplotype pairs. The format of pairs
                  depends on the representation (e.g., tuples of allele strings, tensors of tokens).
        """
        # Removed placeholder warning
        # Actual implementation would involve:
        # 1. Generating candidate haplotype pairs (e.g., from a model posterior).
        # 2. Using self.compatibility_checker.check() to filter or mask invalid pairs.
        # 3. Returning num_samples valid pairs.
        # This might involve rejection sampling or more sophisticated constrained decoding/sampling.

        # --- Implementation for filtering candidate_pairs ---
        if 'candidate_pairs' not in kwargs:
            # For now, this implementation only supports filtering provided candidates.
            # A more advanced sampler might generate candidates internally.
            logging.error("ConstrainedHaplotypeSampler requires 'candidate_pairs' in kwargs for current implementation.")
            return [] # Or raise an error

        candidate_pairs = kwargs['candidate_pairs']
        if not isinstance(candidate_pairs, list):
             raise TypeError("'candidate_pairs' must be a list of haplotype pairs.")

        valid_pairs = []
        for hap1, hap2 in candidate_pairs:
            # Assuming genotype_info is the genotype list [allele1, allele2] for this simple case
            if self.compatibility_checker.check(genotype_info, hap1, hap2):
                valid_pairs.append((hap1, hap2))

        # Return the requested number of samples, up to the number of valid pairs found.
        num_valid = len(valid_pairs)
        num_to_return = min(num_samples, num_valid)

        # For simplicity and test reproducibility, return the first 'num_to_return' valid pairs.
        # A more sophisticated sampler might randomly sample from valid_pairs.
        return valid_pairs[:num_to_return]
