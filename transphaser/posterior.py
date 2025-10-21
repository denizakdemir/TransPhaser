import torch
import torch.nn as nn
import logging
from torch.distributions import Distribution, Normal, Independent

class HaplotypePosteriorDistribution(nn.Module):
    """
    Represents the approximate posterior distribution q(h|g, c) over haplotype pairs,
    parameterized by the output of the GenotypeEncoderTransformer.

    Uses a Gaussian distribution in the latent space. The encoder output is expected
    to contain both mean and log-variance parameters (size latent_dim * 2).
    """
    def __init__(self, latent_dim, num_loci, vocab_sizes):
        """
        Initializes the HaplotypePosteriorDistribution module.

        Args:
            latent_dim (int): The dimensionality of the latent space produced by the encoder.
            num_loci (int): Number of loci.
            vocab_sizes (dict): Dictionary mapping locus names to vocabulary sizes.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_loci = num_loci
        self.vocab_sizes = vocab_sizes

        # Using Gaussian distribution in latent space
        # Encoder output has size latent_dim * 2 (mean + log_var)
        # No additional layers needed - distribution parameters come directly from encoder
        logging.debug("HaplotypePosteriorDistribution initialized with Gaussian latent distribution.")

    def get_distribution(self, encoder_output):
        """
        Creates a PyTorch Distribution object based on the encoder's output.

        Args:
            encoder_output (torch.Tensor): The output tensor from the GenotypeEncoderTransformer.
                                           Shape (batch_size, latent_dim * 2).

        Returns:
            torch.distributions.Independent: Gaussian distribution with diagonal covariance.
        """
        if encoder_output.shape[-1] != self.latent_dim * 2:
            raise ValueError(
                f"Expected encoder_output last dim {self.latent_dim * 2}, "
                f"got {encoder_output.shape[-1]}"
            )

        # Split encoder output into mean and log-variance
        mean, log_var = torch.chunk(encoder_output, 2, dim=-1)
        std_dev = torch.exp(0.5 * log_var)

        # Create independent Normal distribution for each latent dimension
        # Independent wraps Normal to treat latent_dim as event shape
        base_dist = Normal(mean, std_dev)
        posterior = Independent(base_dist, reinterpreted_batch_ndims=1)
        return posterior

    def sample(self, encoder_output, sample_shape=torch.Size()):
        """
        Samples from the posterior distribution q(h|g, c) using reparameterization.

        Args:
            encoder_output (torch.Tensor): Output from the encoder.
            sample_shape (torch.Size): The shape of samples to draw.

        Returns:
            torch.Tensor: Sampled latent variables. Shape (*sample_shape, batch_size, latent_dim).
        """
        posterior = self.get_distribution(encoder_output)
        # Use rsample for reparameterization trick (enables gradients)
        return posterior.rsample(sample_shape)

    def log_prob(self, encoder_output, value):
        """
        Calculates the log probability log q(value | g, c) under the posterior.

        Args:
            encoder_output (torch.Tensor): Output from the encoder used to parameterize q.
            value (torch.Tensor): The value (latent variable) to calculate the log probability of.
                                  Shape should match (batch_size, latent_dim).

        Returns:
            torch.Tensor: The log probability for each sample. Shape (batch_size,).
        """
        posterior = self.get_distribution(encoder_output)
        return posterior.log_prob(value)
