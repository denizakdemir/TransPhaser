import torch
import torch.nn as nn
import math # Import math for calculations

class ELBOLoss(nn.Module):
    """
    Calculates the Evidence Lower Bound (ELBO) loss for the HLA phasing model.
    ELBO = E_q[log p(h|c)] - KL(q(h|g, c) || p(h|c))
    Maximizing ELBO is equivalent to minimizing -ELBO.
    This class computes -ELBO.
    """
    def __init__(self, kl_weight=1.0, reconstruction_weight=1.0):
        """
        Initializes the ELBOLoss module.

        Args:
            kl_weight (float): Weight for the KL divergence term in the loss.
                               Can be used for KL annealing. Defaults to 1.0.
            reconstruction_weight (float): Weight for the reconstruction term
                                           (log p(h|c)). Defaults to 1.0.
        """
        super().__init__()
        if not isinstance(kl_weight, (int, float)) or kl_weight < 0:
            raise ValueError("kl_weight must be a non-negative number.")
        if not isinstance(reconstruction_weight, (int, float)) or reconstruction_weight < 0:
             raise ValueError("reconstruction_weight must be a non-negative number.")

        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        # Removed placeholder print

    def forward(self, model_output):
        """
        Calculates the negative ELBO loss.

        Args:
            model_output (dict): A dictionary containing the necessary components
                                 calculated during the model's forward pass. Expected keys:
                                 - 'reconstruction_log_prob': log p(h|c) term (E_q[log p(h|c)])
                                 - 'kl_divergence': KL(q(h|g, c) || p(h|c)) term

        Returns:
            torch.Tensor: The calculated negative ELBO loss (scalar).
        """
        # Validate input keys
        if 'reconstruction_log_prob' not in model_output:
            raise KeyError("Missing 'reconstruction_log_prob' in model_output.")
        if 'kl_divergence' not in model_output:
            raise KeyError("Missing 'kl_divergence' in model_output.")

        reconstruction_log_prob = model_output['reconstruction_log_prob']
        kl_divergence = model_output['kl_divergence']

        # ELBO = E_q[log p(h|c)] - KL(q||p)
        # We want to maximize ELBO, which means minimizing -ELBO
        # -ELBO = - E_q[log p(h|c)] + KL(q||p)
        # Apply weights
        negative_elbo = -self.reconstruction_weight * reconstruction_log_prob + self.kl_weight * kl_divergence

        # Typically, the loss is averaged over the batch
        return negative_elbo.mean()

    def update_kl_weight(self, new_weight):
        """Allows updating the KL weight, e.g., for annealing."""
        if not isinstance(new_weight, (int, float)) or new_weight < 0:
            raise ValueError("new_weight must be a non-negative number.")
        # Removed print statement, logging handled in trainer
        # print(f"Updating KL weight from {self.kl_weight:.4f} to {new_weight:.4f}")
        self.kl_weight = new_weight


class KLAnnealingScheduler:
    """
    Manages the KL annealing schedule for the ELBO loss.
    Supports different annealing types like linear, sigmoid, cyclical.
    """
    SUPPORTED_TYPES = ['linear', 'sigmoid', 'cyclical'] # Add more as needed

    def __init__(self, anneal_type='linear', max_weight=1.0, total_steps=10000, cycles=1, sigmoid_k=10):
        """
        Initializes the KLAnnealingScheduler.

        Args:
            anneal_type (str): Type of annealing ('linear', 'sigmoid', 'cyclical'). Defaults to 'linear'.
            max_weight (float): The maximum weight for the KL term (beta). Defaults to 1.0.
            total_steps (int): The total number of steps over which annealing occurs
                               (for linear/sigmoid) or the length of half a cycle (for cyclical).
                               Defaults to 10000.
            cycles (int): Number of cycles for cyclical annealing. Defaults to 1.
            sigmoid_k (float): Steepness parameter for sigmoid annealing. Defaults to 10.
        """
        if anneal_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported anneal_type: {anneal_type}. Supported: {self.SUPPORTED_TYPES}")
        if not isinstance(max_weight, (int, float)) or max_weight <= 0:
            raise ValueError("max_weight must be a positive number.")
        if not isinstance(total_steps, int) or total_steps <= 0:
            raise ValueError("total_steps must be a positive integer.")
        if not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer.")
        if not isinstance(sigmoid_k, (int, float)):
             raise ValueError("sigmoid_k must be a number.")

        self.anneal_type = anneal_type
        self.max_weight = float(max_weight)
        self.total_steps = total_steps
        self.cycles = cycles
        self.sigmoid_k = float(sigmoid_k)

        self.current_step = 0
        self.current_weight = self._calculate_weight() # Calculate initial weight
        # Removed placeholder print

    def step(self):
        """Advances the scheduler by one step and updates the current weight."""
        self.current_step += 1
        self.current_weight = self._calculate_weight()
        return self.current_weight

    def get_weight(self):
        """Returns the current KL weight."""
        return self.current_weight

    def _calculate_weight(self):
        """Calculates the KL weight based on the current step and anneal type."""
        step = self.current_step
        total = self.total_steps

        if self.anneal_type == 'linear':
            # Linearly increase from 0 to max_weight over total_steps
            weight = min(self.max_weight, self.max_weight * (step / total))
        elif self.anneal_type == 'sigmoid':
            # Sigmoid increase, centered around total_steps / 2
            # Adjust step relative to the center, scale by k
            exponent = self.sigmoid_k * (step - total / 2) / total
            weight = self.max_weight / (1 + math.exp(-exponent))
        elif self.anneal_type == 'cyclical':
            # Cyclical schedule based on total_steps as half-cycle length
            period = total * 2 # Full cycle length
            cycle_progress = (step % period) / period # Progress within the current full cycle (0 to 1)
            # Linear increase for the first half, decrease/stay max for second (adjust as needed)
            # Simple linear up/down within each cycle for now
            if cycle_progress < 0.5: # First half: ramp up
                 weight = self.max_weight * (cycle_progress * 2) # Scale progress to 0-1
            else: # Second half: ramp down (or stay max) - let's ramp down
                 weight = self.max_weight * (1 - (cycle_progress - 0.5) * 2)
            # Adjust number of cycles - this simple version doesn't explicitly use self.cycles yet
            # A common approach is to just repeat the ramp-up phase 'cycles' times over total_steps*cycles
            # Let's refine this if needed. For now, it's one cycle over total_steps*2.

        else:
            # Should not happen due to init check, but fallback
            weight = self.max_weight

        return max(0.0, weight) # Ensure weight is non-negative
