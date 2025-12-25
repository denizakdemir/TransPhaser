"""
TransPhaser: Neural Expectation-Maximization for HLA Phasing

This module implements the TransPhaser architecture (formerly NeuralEEM),
which combines:

1. Neural Proposal Network (Amortized E-step)
   - Proposes top-K candidate haplotype pairs for each genotype
   - Replaces expensive posterior enumeration

2. Conditional Haplotype Prior with Embeddings
   - π(h | x) via learned embeddings + covariates
   - Provides principled smoothing for rare haplotypes

3. HWE Haplotype Pair Prior
   - P(h1, h2 | x) with Hardy-Weinberg equilibrium
   - Identifiability through generative structure

4. Constrained Emission Model
   - P(G | h1, h2) with strict compatibility constraints
   - Calibrated noise for lab/site heterogeneity

5. Truncated EM Training
   - Exact scoring on candidate set (top-K)
   - Gradient-based M-step for neural parameters
   - Proposal distillation for improved amortization

Key insight: Neural nets propose, probabilistic model decides.
This keeps the system grounded in likelihood maximization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Optional, List, Tuple


class HaplotypeEmbedding(nn.Module):
    """
    Embeds a multi-locus haplotype into a fixed-dimensional vector.
    
    e(h) = Pool(E_1[a_1], ..., E_L[a_L])
    
    where E_l is the embedding for locus l.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.vocab_sizes = config["vocab_sizes"]
        self.embedding_dim = config["embedding_dim"]
        self.num_loci = config["num_loci"]
        self.loci_order = config.get("loci_order", sorted(list(self.vocab_sizes.keys())))
        
        # Per-locus allele embeddings
        self.allele_embeddings = nn.ModuleDict({
            locus: nn.Embedding(vocab_size, self.embedding_dim, padding_idx=config.get("padding_idx", 0))
            for locus, vocab_size in self.vocab_sizes.items()
        })
        
        # Pooling layer: concatenate + project (simple and effective)
        # Alternative: small transformer over loci (6-7 tokens)
        self.pooling = nn.Sequential(
            nn.Linear(self.embedding_dim * self.num_loci, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.pooling:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, haplotypes: torch.Tensor) -> torch.Tensor:
        """
        Embed haplotype sequences.
        
        Args:
            haplotypes: Token indices (batch, num_loci) or (batch, k, num_loci)
        
        Returns:
            Embeddings: (batch, embedding_dim) or (batch, k, embedding_dim)
        """
        has_candidates = haplotypes.dim() == 3
        
        if has_candidates:
            batch_size, k, num_loci = haplotypes.shape
            # Flatten for embedding lookup
            haplotypes_flat = haplotypes.view(batch_size * k, num_loci)
        else:
            batch_size, num_loci = haplotypes.shape
            haplotypes_flat = haplotypes
        
        # Get embeddings for each locus
        embeddings = []
        for i, locus in enumerate(self.loci_order):
            locus_tokens = haplotypes_flat[:, i]  # (batch*k,) or (batch,)
            # Clamp to valid range
            locus_tokens = locus_tokens.clamp(0, self.vocab_sizes[locus] - 1)
            embeddings.append(self.allele_embeddings[locus](locus_tokens))
        
        # Concatenate and pool
        concat_embed = torch.cat(embeddings, dim=-1)  # (batch*k, embedding_dim * num_loci)
        pooled = self.pooling(concat_embed)  # (batch*k, embedding_dim)
        
        if has_candidates:
            return pooled.view(batch_size, k, self.embedding_dim)
        return pooled


class ConditionalHaplotypePrior(nn.Module):
    """
    Neural conditional prior: π(h | x) = softmax(s(e(h), x))
    
    The prior is conditioned on covariates (ancestry, age, etc.)
    and uses haplotype embeddings for principled smoothing.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.embedding_dim = config["embedding_dim"]
        self.covariate_dim = config.get("covariate_dim", 0)
        
        # Haplotype embedding
        self.haplotype_embedding = HaplotypeEmbedding(config)
        
        # Scoring network: s(e(h), x)
        input_dim = self.embedding_dim + self.covariate_dim
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.scorer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, candidates: torch.Tensor, covariates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute log-probabilities for candidate haplotypes.
        
        Args:
            candidates: Candidate haplotype tokens (batch, k, num_loci)
            covariates: Covariate tensor (batch, covariate_dim) or None
        
        Returns:
            log_probs: Log-probabilities (batch, k) that sum to 1
        """
        batch_size, k, num_loci = candidates.shape
        
        # Get haplotype embeddings
        h_embed = self.haplotype_embedding(candidates)  # (batch, k, embedding_dim)
        
        # Expand covariates to match candidates
        if covariates is not None and self.covariate_dim > 0:
            cov_expanded = covariates.unsqueeze(1).expand(-1, k, -1)  # (batch, k, cov_dim)
            combined = torch.cat([h_embed, cov_expanded], dim=-1)  # (batch, k, embed + cov)
        else:
            combined = h_embed
        
        # Score each candidate
        scores = self.scorer(combined).squeeze(-1)  # (batch, k)
        
        # Return log-softmax (normalized log-probabilities)
        log_probs = F.log_softmax(scores, dim=-1)
        
        return log_probs


class HWEHaplotypePairPrior(nn.Module):
    """
    Hardy-Weinberg Equilibrium prior for haplotype pairs.
    
    P(h1, h2 | x) = 
        2 * π(h1|x) * π(h2|x)  if h1 ≠ h2 (heterozygous)
        π(h|x)²                 if h1 = h2 (homozygous)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.embedding_dim = config["embedding_dim"]
    
    def compute_pair_log_prob(
        self, 
        log_pi_h1: torch.Tensor, 
        log_pi_h2: torch.Tensor, 
        is_homozygous: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-probability of haplotype pairs under HWE.
        
        Args:
            log_pi_h1: Log-probability of h1 under π (batch, k)
            log_pi_h2: Log-probability of h2 under π (batch, k)
            is_homozygous: Boolean mask for h1 == h2 (batch, k)
        
        Returns:
            log_pair_prob: Log P(h1, h2 | x) (batch, k)
        """
        # Heterozygous: log(2) + log π(h1) + log π(h2)
        log_het = math.log(2.0) + log_pi_h1 + log_pi_h2
        
        # Homozygous: 2 * log π(h)
        log_hom = 2 * log_pi_h1  # h1 == h2, so use either
        
        # Select based on homozygosity
        log_pair_prob = torch.where(is_homozygous, log_hom, log_het)
        
        return log_pair_prob


class ConstrainedEmissionModel(nn.Module):
    """
    Emission model: P(G | h1, h2) = Compat(G; h1, h2) * Noise(G, x)
    
    - Compat ∈ {0, 1}: Hard constraint that alleles are compatible
    - Noise: Calibrated term for lab/site heterogeneity
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.num_loci = config["num_loci"]
        self.loci_order = config.get("loci_order", [f"locus_{i}" for i in range(self.num_loci)])
        self.vocab_sizes = config["vocab_sizes"]
        
        # Optional noise model (can be disabled for strict compatibility)
        self.use_noise = config.get("use_emission_noise", True)
        if self.use_noise:
            self.noise_logit = nn.Parameter(torch.tensor(0.0))  # Learned noise level
    
    def forward(
        self, 
        genotypes: torch.Tensor, 
        h1: torch.Tensor, 
        h2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute emission log-probability.
        
        Args:
            genotypes: Genotype tokens (batch, num_loci, 2) - two alleles per locus
            h1: Haplotype 1 tokens (batch, num_loci)
            h2: Haplotype 2 tokens (batch, num_loci)
        
        Returns:
            log_emission: Log P(G | h1, h2) per sample (batch,)
        """
        batch_size = h1.shape[0]
        
        # Check compatibility at each locus
        # For each locus: {h1[l], h2[l]} should match {G[l, 0], G[l, 1]} as multisets
        
        # Sorted genotype alleles
        g_sorted, _ = torch.sort(genotypes, dim=-1)  # (batch, num_loci, 2)
        
        # Sorted haplotype alleles
        h_stacked = torch.stack([h1, h2], dim=-1)  # (batch, num_loci, 2)
        h_sorted, _ = torch.sort(h_stacked, dim=-1)  # (batch, num_loci, 2)
        
        # Check if sorted alleles match
        compatible = (g_sorted == h_sorted).all(dim=-1)  # (batch, num_loci)
        
        # All loci must be compatible
        all_compatible = compatible.all(dim=-1)  # (batch,)
        
        # Log-probability: log(1) = 0 if compatible, log(0) = -inf if not
        # Use a small epsilon for numerical stability
        if self.use_noise:
            # Add small noise probability for non-compatible pairs
            noise_prob = torch.sigmoid(self.noise_logit) * 1e-6
            log_emission = torch.where(
                all_compatible,
                torch.zeros_like(all_compatible, dtype=torch.float),
                torch.log(noise_prob + 1e-10)
            )
        else:
            log_emission = torch.where(
                all_compatible,
                torch.zeros_like(all_compatible, dtype=torch.float),
                torch.full_like(all_compatible, -1e10, dtype=torch.float)
            )
        
        return log_emission


class ProposalNetwork(nn.Module):
    """
    Transformer-based Constrained Proposal Network: q(h1, h2 | G, x)
    
    Uses self-attention to model CROSS-LOCUS DEPENDENCIES (linkage disequilibrium).
    This is critical for learning which alleles at different loci tend to 
    co-occur on the same haplotype.
    
    CRITICAL: Proposals are CONSTRAINED to the compatible set C(G).
    For each locus, we can only assign alleles that appear in the genotype.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.num_loci = config["num_loci"]
        self.embedding_dim = config["embedding_dim"]
        self.covariate_dim = config.get("covariate_dim", 0)
        self.top_k = config.get("top_k", 16)
        self.vocab_sizes = config["vocab_sizes"]
        self.loci_order = config.get("loci_order", sorted(list(self.vocab_sizes.keys())))
        num_heads = config.get("num_heads", 4)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout", 0.1)
        
        # Allele embeddings (per locus)
        max_vocab = max(self.vocab_sizes.values())
        self.allele_embedding = nn.Embedding(max_vocab + 10, self.embedding_dim, padding_idx=0)
        
        # Locus positional embeddings
        self.locus_embedding = nn.Embedding(self.num_loci, self.embedding_dim)
        
        # Combine allele pairs per locus: (allele1_embed + allele2_embed) -> locus representation
        self.pair_combiner = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        
        # Transformer encoder to model cross-locus dependencies
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=num_heads,
            dim_feedforward=self.embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Covariate projection
        if self.covariate_dim > 0:
            self.covariate_proj = nn.Linear(self.covariate_dim, self.embedding_dim)
        
        # Per-locus phasing predictor: output logit for "first allele goes to h1"
        self.phasing_head = nn.Linear(self.embedding_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.pair_combiner.weight)
        nn.init.zeros_(self.pair_combiner.bias)
        nn.init.xavier_uniform_(self.phasing_head.weight)
        nn.init.zeros_(self.phasing_head.bias)
        if hasattr(self, 'covariate_proj'):
            nn.init.xavier_uniform_(self.covariate_proj.weight)
            nn.init.zeros_(self.covariate_proj.bias)
    
    def forward(
        self, 
        genotypes: torch.Tensor, 
        covariates: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propose top-K candidate haplotype pairs with cross-locus context.
        
        Args:
            genotypes: Genotype tokens (batch, num_loci * 2)
            covariates: Covariate tensor (batch, covariate_dim)
        
        Returns:
            candidates: (batch, k, num_loci, 2) - k pairs of haplotypes
            log_q: (batch, k) - proposal log-probabilities
        """
        batch_size = genotypes.shape[0]
        device = genotypes.device
        
        # Reshape genotypes to (batch, num_loci, 2)
        genotypes_reshaped = genotypes.view(batch_size, self.num_loci, 2)
        
        # Embed each allele
        genotypes_clamped = genotypes_reshaped.clamp(0, self.allele_embedding.num_embeddings - 1)
        allele1_embed = self.allele_embedding(genotypes_clamped[:, :, 0])  # (batch, num_loci, embed)
        allele2_embed = self.allele_embedding(genotypes_clamped[:, :, 1])  # (batch, num_loci, embed)
        
        # Combine allele pairs
        pair_embed = torch.cat([allele1_embed, allele2_embed], dim=-1)  # (batch, num_loci, embed*2)
        locus_repr = self.pair_combiner(pair_embed)  # (batch, num_loci, embed)
        
        # Add locus positional embeddings
        locus_ids = torch.arange(self.num_loci, device=device).unsqueeze(0).expand(batch_size, -1)
        locus_repr = locus_repr + self.locus_embedding(locus_ids)
        
        # Add covariate information as a context token
        if covariates is not None and self.covariate_dim > 0:
            cov_embed = self.covariate_proj(covariates).unsqueeze(1)  # (batch, 1, embed)
            # Broadcast to all loci
            locus_repr = locus_repr + cov_embed
        
        # Apply transformer to capture cross-locus dependencies
        locus_context = self.transformer(locus_repr)  # (batch, num_loci, embed)
        
        # Predict phasing probability per locus
        phasing_logits = self.phasing_head(locus_context).squeeze(-1)  # (batch, num_loci)
        phasing_probs = torch.sigmoid(phasing_logits)  # P(h1 gets first allele)
        
        # Generate K candidates with diverse phasing configurations
        candidates = torch.zeros(batch_size, self.top_k, self.num_loci, 2, 
                                dtype=torch.long, device=device)
        log_q = torch.zeros(batch_size, self.top_k, device=device)
        
        # Candidate 0: MAP phasing
        # Candidates 1 to num_loci: flip one locus each
        # Remaining: sample from distribution
        for k_idx in range(self.top_k):
            if k_idx == 0:
                # MAP phasing
                phase_choice = (phasing_probs > 0.5).long()
            elif k_idx <= self.num_loci:
                # Flip locus (k_idx - 1) from MAP
                phase_choice = (phasing_probs > 0.5).long()
                flip_locus = k_idx - 1
                phase_choice[:, flip_locus] = 1 - phase_choice[:, flip_locus]
            else:
                # Sample from distribution
                phase_choice = torch.bernoulli(phasing_probs).long()
            
            # Build haplotypes from phase choices
            for locus_idx in range(self.num_loci):
                allele1 = genotypes_reshaped[:, locus_idx, 0]
                allele2 = genotypes_reshaped[:, locus_idx, 1]
                phase = phase_choice[:, locus_idx]
                
                h1_allele = torch.where(phase == 1, allele1, allele2)
                h2_allele = torch.where(phase == 1, allele2, allele1)
                
                candidates[:, k_idx, locus_idx, 0] = h1_allele
                candidates[:, k_idx, locus_idx, 1] = h2_allele
            
            # Compute log-probability
            phase_float = phase_choice.float()
            log_prob = (phase_float * torch.log(phasing_probs + 1e-8) + 
                       (1 - phase_float) * torch.log(1 - phasing_probs + 1e-8))
            log_q[:, k_idx] = log_prob.sum(dim=-1)
        
        # Return phasing_logits for supervised training
        return candidates, log_q, phasing_logits


class TransPhaser(nn.Module):
    """
    TransPhaser: Neural Expectation-Maximization for HLA Phasing
    
    Combines:
    1. Proposal network for amortized E-step
    2. Conditional prior π(h | x)
    3. HWE pair prior P(h1, h2 | x)
    4. Constrained emission P(G | h1, h2)
    
    Training objective: Maximize marginal likelihood over candidate set.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.num_loci = config["num_loci"]
        self.top_k = config.get("top_k", 16)
        self.loci_order = config.get("loci_order", sorted(list(config["vocab_sizes"].keys())))
        
        # Components
        self.proposal = ProposalNetwork(config)
        self.prior = ConditionalHaplotypePrior(config)
        self.hwe_prior = HWEHaplotypePairPrior(config)
        self.emission = ConstrainedEmissionModel(config)
        
        logging.info(f"TransPhaser initialized with top_k={self.top_k}, num_loci={self.num_loci}")
    
    def _genotype_to_pairs(self, genotype_tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert flattened genotype to (num_loci, 2) format.
        
        Args:
            genotype_tokens: (batch, num_loci * 2)
        
        Returns:
            genotypes: (batch, num_loci, 2)
        """
        batch_size = genotype_tokens.shape[0]
        return genotype_tokens.view(batch_size, self.num_loci, 2)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass: Propose candidates, compute responsibilities.
        
        Args:
            batch: Dictionary containing:
                - genotype_tokens: (batch, num_loci * 2)
                - covariates: (batch, covariate_dim) [optional]
        
        Returns:
            Dictionary containing:
                - responsibilities: (batch, k) - posterior over candidates
                - h1_tokens: (batch, num_loci) - predicted h1
                - h2_tokens: (batch, num_loci) - predicted h2
                - log_likelihood: (batch,) - marginal log-likelihood
                - candidates: (batch, k, num_loci, 2) - candidate pairs
        """
        genotype_tokens = batch["genotype_tokens"]
        covariates = batch.get("covariates", None)
        batch_size = genotype_tokens.shape[0]
        device = genotype_tokens.device
        
        # Convert to (batch, num_loci, 2) format
        genotypes = self._genotype_to_pairs(genotype_tokens)
        
        # 1. Propose candidates
        candidates, log_q, phasing_logits = self.proposal(genotype_tokens, covariates)
        # candidates: (batch, k, num_loci, 2)
        # log_q: (batch, k)
        # phasing_logits: (batch, num_loci) - for supervised training
        
        # 2. Extract h1 and h2 from candidates
        h1_candidates = candidates[:, :, :, 0]  # (batch, k, num_loci)
        h2_candidates = candidates[:, :, :, 1]  # (batch, k, num_loci)
        
        # 3. Compute prior log-probabilities
        log_pi_h1 = self.prior(h1_candidates, covariates)  # (batch, k)
        log_pi_h2 = self.prior(h2_candidates, covariates)  # (batch, k)
        
        # Check for homozygous pairs
        is_homozygous = (h1_candidates == h2_candidates).all(dim=-1)  # (batch, k)
        
        # HWE pair prior
        log_pair_prior = self.hwe_prior.compute_pair_log_prob(
            log_pi_h1, log_pi_h2, is_homozygous
        )  # (batch, k)
        
        # 4. Compute emission log-probabilities
        log_emissions = []
        for k_idx in range(self.top_k):
            h1 = h1_candidates[:, k_idx, :]  # (batch, num_loci)
            h2 = h2_candidates[:, k_idx, :]  # (batch, num_loci)
            log_emit = self.emission(genotypes, h1, h2)  # (batch,)
            log_emissions.append(log_emit)
        log_emission = torch.stack(log_emissions, dim=-1)  # (batch, k)
        
        # 5. Compute unnormalized log-weights (E-step)
        # w_c ∝ P(h1, h2 | x) * P(G | h1, h2)
        log_weights = log_pair_prior + log_emission  # (batch, k)
        
        # Normalize to get responsibilities
        log_normalizer = torch.logsumexp(log_weights, dim=-1, keepdim=True)  # (batch, 1)
        log_responsibilities = log_weights - log_normalizer  # (batch, k)
        responsibilities = torch.exp(log_responsibilities)  # (batch, k)
        
        # 6. Select best candidate (MAP prediction)
        best_idx = log_weights.argmax(dim=-1)  # (batch,)
        batch_indices = torch.arange(batch_size, device=device)
        h1_pred = h1_candidates[batch_indices, best_idx, :]  # (batch, num_loci)
        h2_pred = h2_candidates[batch_indices, best_idx, :]  # (batch, num_loci)
        
        # Marginal log-likelihood
        log_likelihood = log_normalizer.squeeze(-1)  # (batch,)
        
        return {
            "responsibilities": responsibilities,
            "log_responsibilities": log_responsibilities,
            "h1_tokens": h1_pred,
            "h2_tokens": h2_pred,
            "log_likelihood": log_likelihood,
            "candidates": candidates,
            "log_pair_prior": log_pair_prior,
            "log_emission": log_emission,
            "log_q": log_q,  # For proposal distillation
            "phasing_logits": phasing_logits,  # For supervised training
        }
    
    @torch.no_grad()
    def predict_haplotypes(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict haplotypes for a batch.
        
        Args:
            batch: Dictionary with genotype_tokens and covariates
        
        Returns:
            h1: (batch, num_loci) - predicted haplotype 1
            h2: (batch, num_loci) - predicted haplotype 2
        """
        output = self.forward(batch)
        return output["h1_tokens"], output["h2_tokens"]


class TransPhaserLoss(nn.Module):
    """
    Loss function for TransPhaser training.
    
    Combines:
    1. Negative log-likelihood (marginal)
    2. Proposal distillation (KL divergence)
    3. Entropy regularization
    4. Supervised phasing loss (optional, when ground truth is available)
    """
    
    def __init__(
        self, 
        proposal_weight: float = 0.1,
        entropy_weight: float = 0.01,
        supervised_weight: float = 1.0,
    ):
        super().__init__()
        self.proposal_weight = proposal_weight
        self.entropy_weight = entropy_weight
        self.supervised_weight = supervised_weight
    
    def forward(
        self, 
        output: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute TransPhaser loss.
        
        Args:
            output: Model output dictionary
            batch: Input batch dictionary
        
        Returns:
            loss: Scalar loss value
        """
        # 1. Negative marginal log-likelihood
        nll_loss = -output["log_likelihood"].mean()
        
        # 2. Proposal distillation: KL(r || q)
        # Encourage proposal to match responsibilities
        log_r = output["log_responsibilities"]  # (batch, k)
        log_q = output["log_q"]  # (batch, k)
        
        # KL divergence: sum_c r_c * (log r_c - log q_c)
        r = output["responsibilities"]  # (batch, k)
        kl_div = (r * (log_r - log_q)).sum(dim=-1).mean()
        
        # 3. Entropy regularization (encourage diverse predictions)
        entropy = -(r * log_r).sum(dim=-1).mean()
        
        # 4. Supervised loss using BCE on phasing_logits (differentiable)
        supervised_loss = torch.tensor(0.0, device=nll_loss.device)
        
        if "target_h1_tokens" in batch and "target_h2_tokens" in batch and "phasing_logits" in output:
            # Get phasing logits and input genotypes
            phasing_logits = output["phasing_logits"]  # (batch, num_loci)
            genotype_tokens = batch["genotype_tokens"]  # (batch, num_loci * 2)
            
            # Ground truth includes BOS at [0], alleles at [1:num_loci+1]
            target_h1 = batch["target_h1_tokens"]  # (batch, num_loci + 2)
            target_h2 = batch["target_h2_tokens"]  # (batch, num_loci + 2)
            
            num_loci = phasing_logits.shape[1]
            batch_size = phasing_logits.shape[0]
            
            # Extract allele tokens (skip BOS at index 0)
            target_h1_alleles = target_h1[:, 1:num_loci+1]  # (batch, num_loci)
            target_h2_alleles = target_h2[:, 1:num_loci+1]  # (batch, num_loci)
            
            # Reshape genotypes to (batch, num_loci, 2)
            genotypes = genotype_tokens.view(batch_size, num_loci, 2)
            allele1 = genotypes[:, :, 0]  # (batch, num_loci)
            allele2 = genotypes[:, :, 1]  # (batch, num_loci)
            
            # Determine target phase for DIRECT ordering (h1 -> target_h1, h2 -> target_h2)
            # phase_target = 1 means h1 should get allele1
            # So we check: does target_h1 match allele1 at each locus?
            direct_phase = (target_h1_alleles == allele1).float()  # (batch, num_loci)
            
            # Similarly for SWAPPED ordering (h1 -> target_h2, h2 -> target_h1)
            swapped_phase = (target_h2_alleles == allele1).float()  # (batch, num_loci)
            
            # Compute BCE for both orderings
            direct_bce = F.binary_cross_entropy_with_logits(
                phasing_logits, direct_phase, reduction='none'
            ).sum(dim=-1)  # (batch,)
            
            swapped_bce = F.binary_cross_entropy_with_logits(
                phasing_logits, swapped_phase, reduction='none'
            ).sum(dim=-1)  # (batch,)
            
            # Take minimum (phase ambiguity)
            supervised_loss = torch.minimum(direct_bce, swapped_bce).mean()
        
        # Total loss
        loss = (nll_loss + 
                self.proposal_weight * kl_div - 
                self.entropy_weight * entropy +
                self.supervised_weight * supervised_loss)
        
        return loss
