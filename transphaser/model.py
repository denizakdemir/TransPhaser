import torch
import torch.nn as nn
import logging # Add missing import

# Import actual transformer implementations
from transphaser.encoder import GenotypeEncoderTransformer # Corrected import path
from transphaser.decoder import HaplotypeDecoderTransformer # Corrected import path

class HLAPhasingModel(nn.Module):
    """
    Main probabilistic model for HLA phasing, combining an encoder and a decoder.
    Implements the Evidence Lower Bound (ELBO) objective.
    """
    # Add tokenizer to __init__ signature
    def __init__(self, num_loci, allele_vocabularies, covariate_dim, tokenizer, encoder_config=None, decoder_config=None):
        """
        Initializes the HLAPhasingModel.

        Args:
            num_loci (int): The number of HLA loci being phased (k).
            allele_vocabularies (dict): A dictionary mapping locus names to their
                                        vocabulary dictionaries (allele -> token_id).
            covariate_dim (int): The dimensionality of the encoded covariate vector.
            tokenizer: An initialized AlleleTokenizer instance. # Added tokenizer arg
            encoder_config (dict, optional): Configuration dictionary for the encoder transformer. Defaults to None.
            decoder_config (dict, optional): Configuration dictionary for the decoder transformer. Defaults to None.
        """
        super().__init__()

        self.num_loci = num_loci
        self.allele_vocabularies = allele_vocabularies
        self.covariate_dim = covariate_dim
        self.tokenizer = tokenizer # Store tokenizer

        # Prepare encoder config
        self.encoder_config = encoder_config if encoder_config else {}
        # Ensure vocab_sizes is in encoder_config, derived from allele_vocabularies
        if "vocab_sizes" not in self.encoder_config:
             self.encoder_config["vocab_sizes"] = {locus: len(vocab) for locus, vocab in allele_vocabularies.items()}
        # Ensure num_loci is in encoder_config
        if "num_loci" not in self.encoder_config:
            self.encoder_config["num_loci"] = self.num_loci
        # Ensure covariate_dim is in encoder_config (if applicable)
        if "covariate_dim" not in self.encoder_config:
             self.encoder_config["covariate_dim"] = self.covariate_dim
        # Ensure tokenizer is available if needed by encoder (though likely not)
        # self.encoder_config['tokenizer'] = self.tokenizer

        # Prepare decoder config
        self.decoder_config = decoder_config if decoder_config else {}
        # Ensure necessary keys are in decoder_config
        if "vocab_sizes" not in self.decoder_config:
             self.decoder_config["vocab_sizes"] = {locus: len(vocab) for locus, vocab in allele_vocabularies.items()}
        if "num_loci" not in self.decoder_config:
            self.decoder_config["num_loci"] = self.num_loci
        if "covariate_dim" not in self.decoder_config:
             self.decoder_config["covariate_dim"] = self.covariate_dim
        self.decoder_config['tokenizer'] = self.tokenizer # Ensure tokenizer is in decoder config

        # Initialize actual encoder and decoder models
        # Pass the prepared config dictionaries
        self.encoder = GenotypeEncoderTransformer(self.encoder_config)
        self.decoder = HaplotypeDecoderTransformer(self.decoder_config)
        logging.info("HLAPhasingModel initialized with Encoder and Decoder.")
        # Store latent_dim for convenience
        self.latent_dim = self.encoder.latent_dim # Assuming encoder exposes latent_dim

        # Apply custom weight initialization
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Applies Xavier initialization to linear layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Optionally initialize embeddings differently, e.g., standard normal
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        """
        Performs a forward pass through the VAE model.

        Args:
            batch (dict): A batch dictionary from the HLADataset, containing:
                          'genotype_tokens': Input genotype tokens (batch, seq_len_encoder).
                          'covariates': Encoded covariates (batch, covariate_dim).
                          'target_haplotype_tokens': Target haplotype tokens for reconstruction
                                                     (batch, seq_len_decoder). Assumed to exist.
                          'target_locus_indices': Locus indices for target tokens (batch, seq_len_decoder). Assumed.
                          'attention_mask' (optional): Padding mask for encoder input.
                          'decoder_attention_mask' (optional): Padding mask for decoder input.

        Returns:
            dict: A dictionary containing:
                  'reconstruction_log_prob': Log probability of target haplotypes given z and c.
                  'kl_divergence': KL divergence between q(z|g,c) and p(z).
                  'logits': Raw output logits from the decoder (optional, for debugging/analysis).
        """
        genotype_tokens = batch['genotype_tokens']
        covariates = batch.get('covariates') # Use .get for optional covariates
        attention_mask = batch.get('attention_mask') # Encoder padding mask
        target_tokens = batch['target_haplotype_tokens'] # Assumed target tokens
        target_locus_indices = batch['target_locus_indices'] # Assumed locus indices
        decoder_attention_mask = batch.get('decoder_attention_mask') # Decoder padding mask

        # 1. Encode genotype and covariates to get parameters of q(z|g, c)
        # Encoder output shape: (batch_size, latent_dim * 2)
        latent_params = self.encoder(genotype_tokens, covariates=covariates, attention_mask=attention_mask)
        mu = latent_params[:, :self.latent_dim]
        log_var = latent_params[:, self.latent_dim:]

        # --- DEBUG: Check mu and log_var immediately after encoder ---
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            logging.warning(f"NaN or Inf detected in mu from encoder!")
        if torch.isnan(log_var).any() or torch.isinf(log_var).any():
            logging.warning(f"NaN or Inf detected in log_var from encoder!")
        # --- End DEBUG ---

        # --- Add log_var clipping for stability ---
        log_var = torch.clamp(log_var, min=-10, max=10) # Keep log_var clamping
        # --- End clipping ---

        # --- Re-introduce mu clipping for stability ---
        mu = torch.clamp(mu, min=-10, max=10) # Clamp mu as well
        # --- End clipping ---

        # 2. Sample latent variable z from q(z|g, c) using reparameterization trick
        z = self.reparameterize(mu, log_var) # Shape: (batch_size, latent_dim)

        # --- Add z clamping for stability ---
        z = torch.clamp(z, min=-10, max=10) # Clamp sampled latent variable
        # --- End clamping ---

        # 3. Calculate KL divergence: KL(q(z|g, c) || p(z)) where p(z) = N(0, I)
        # Formula: 0.5 * sum(1 + log_var - mu^2 - exp(log_var)) per sample
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        # kl_div shape: (batch_size,)

        # --- DEBUG: Check for NaN/Inf in KL divergence ---
        if torch.isnan(kl_div).any() or torch.isinf(kl_div).any():
            logging.warning(f"NaN or Inf detected in KL divergence!")
        # --- End DEBUG ---

        # --- DEBUG: Check for NaN/Inf in sampled z ---
        if torch.isnan(z).any() or torch.isinf(z).any():
            logging.warning(f"NaN or Inf detected in sampled latent variable z!")
        # --- End DEBUG ---


        # 4. Decode z and covariates to reconstruct target haplotypes
        # Prepare inputs for the decoder based on the updated decoder logic
        # Input: [BOS, A1, ..., Ak] (Shape: batch, num_loci + 1)
        # Target: [A1, ..., Ak, EOS] (Shape: batch, num_loci + 1)
        decoder_input_tokens = target_tokens[:, :-1] # Includes BOS, excludes EOS
        decoder_target_tokens = target_tokens[:, 1:]  # Excludes BOS, includes EOS
        decoder_padding_mask = decoder_attention_mask[:, :-1] if decoder_attention_mask is not None else None # Mask for input seq

        # Decoder returns a dict: {locus_name: logits (batch, vocab_size)}
        # The logits dict corresponds to predictions for A1...Ak based on input BOS...A(k-1)
        all_logits = self.decoder(
            input_tokens=decoder_input_tokens,      # Shape (batch, num_loci + 1)
            locus_indices=None, # Pass None as it's not used by decoder anymore
            covariates=covariates,
            latent_variable=z,
            attention_mask=decoder_padding_mask     # Shape (batch, num_loci + 1)
        )
        # Output `all_logits` is dict: {locus: (batch, vocab_size)}
        # all_logits[locus_i] contains logits for predicting allele i+1

        # --- DEBUG: Check for NaN/Inf in logits ---
        for locus_name_check, locus_logits_check in all_logits.items():
            if torch.isnan(locus_logits_check).any() or torch.isinf(locus_logits_check).any():
                logging.warning(f"NaN or Inf detected in logits for locus {locus_name_check} before loss calculation!")
        # --- End DEBUG ---

        # 5. Calculate reconstruction log probability: log p(h|z, c)
        # Sum cross-entropy loss across all loci + EOS token
        total_neg_log_prob = torch.zeros(genotype_tokens.size(0), device=genotype_tokens.device)
        # Get pad_token_id from the stored tokenizer
        pad_token_id = self.tokenizer.special_tokens.get("PAD", 0) # Correctly get pad_id from tokenizer

        for i, locus_name in enumerate(self.decoder.loci_order):
            locus_logits = all_logits[locus_name] # Logits for predicting allele i+1

            # --- Add logit clamping for stability ---
            locus_logits = torch.clamp(locus_logits, min=-10, max=10) # Clamp logits
            # --- End clamping ---

            # Target is the actual allele token for locus i+1 (index i+1 in original target, index i in shifted target)
            locus_targets = decoder_target_tokens[:, i] # Shape: (batch,)

            # Calculate cross-entropy loss for this locus prediction
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
            locus_neg_log_prob = loss_fn(locus_logits, locus_targets) # Shape: (batch,)

            # Mask out padding in the target if necessary (though typically target shouldn't be padded here)
            # target_mask = (locus_targets != pad_token_id)
            # locus_neg_log_prob = locus_neg_log_prob * target_mask

            total_neg_log_prob += locus_neg_log_prob

        # reconstruction_log_prob is the negative of the summed negative log probabilities
        reconstruction_log_prob = -total_neg_log_prob # Shape: (batch_size,)

        return {
            'reconstruction_log_prob': reconstruction_log_prob, # E_q[log p(h|z,c)] term
            'kl_divergence': kl_div, # KL(q||p) term
            'logits': all_logits # Optional: return dict of logits per locus
        }

    # Add torch.no_grad() back for inference efficiency
    @torch.no_grad()
    def predict_haplotypes(self, batch, max_len=None, return_logits=False):
        """
        Predicts haplotype sequences autoregressively using the trained model.

        Args:
            batch (dict): A batch containing 'genotype_tokens' and optionally 'covariates'.
            max_len (int, optional): Maximum length of the generated sequence (number of loci).
                                     Defaults to self.num_loci.
            return_logits (bool, optional): If True, also returns the masked logits used
                                            at each prediction step. Defaults to False.

        Returns:
            torch.Tensor or tuple:
                If return_logits is False:
                    Tensor of predicted haplotype token sequences (excluding BOS/EOS).
                    Shape (batch_size, num_loci).
                If return_logits is True:
                    A tuple containing:
                    - Predicted sequence tensor (batch_size, num_loci)
                    - List of masked logits tensors, one for each step.
                      Each tensor shape (batch_size, vocab_size).
        """
        self.eval() # Ensure model is in eval mode
        genotype_tokens = batch['genotype_tokens'].to(self.encoder.positional_embedding.weight.device) # Move to model device
        covariates = batch.get('covariates')
        if covariates is not None:
            covariates = covariates.to(genotype_tokens.device)

        batch_size = genotype_tokens.size(0)
        device = genotype_tokens.device
        max_len = max_len if max_len is not None else self.num_loci

        # 1. Encode to get latent variable (use mean for deterministic prediction)
        latent_params = self.encoder(genotype_tokens, covariates=covariates)
        mu = latent_params[:, :self.latent_dim]
        # z = self.reparameterize(mu, log_var) # Use mu for deterministic prediction
        z = mu

        # 2. Autoregressive decoding
        # Start with BOS token
        decoder_input_tokens = torch.full((batch_size, 1),
                                          self.tokenizer.special_tokens.get("BOS", 2),
                                          dtype=torch.long, device=device)
        predicted_sequence = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        all_step_logits = [] if return_logits else None

        for step in range(max_len):
            # Prepare decoder input for this step
            current_input_len = decoder_input_tokens.size(1)
            # Locus indices are not needed by current decoder forward
            # locus_indices_step = torch.arange(current_input_len, device=device).unsqueeze(0).repeat(batch_size, 1)

            # Get logits from decoder
            logits_dict = self.decoder(
                input_tokens=decoder_input_tokens,
                locus_indices=None, # Not used
                covariates=covariates,
                latent_variable=z # Revert: Pass z during prediction
                # No attention mask needed for generation usually, unless handling padding within generation
            )

            # Get logits for the current locus being predicted
            current_locus_name = self.decoder.loci_order[step]
            step_logits = logits_dict[current_locus_name] # Shape: (batch_size, vocab_size)

            # --- Prevent predicting PAD and UNK tokens ---
            pad_token_id = self.tokenizer.special_tokens.get("PAD", 0)
            unk_token_id = self.tokenizer.special_tokens.get("UNK", 1) # Assuming UNK is 1
            bos_token_id = self.tokenizer.special_tokens.get("BOS", 2) # Assuming BOS is 2
            eos_token_id = self.tokenizer.special_tokens.get("EOS", 3) # Assuming EOS is 3
            step_logits[:, pad_token_id] = -float('inf')
            step_logits[:, unk_token_id] = -float('inf')
            step_logits[:, bos_token_id] = -float('inf') # Prevent predicting BOS
            step_logits[:, eos_token_id] = -float('inf') # Prevent predicting EOS
            # --- End prevention ---

            # --- Enforce Genotype Compatibility ---
            # Get the two valid allele tokens for the current locus from the input genotype
            locus_genotype_token1 = genotype_tokens[:, step * 2]     # Shape: (batch_size,)
            locus_genotype_token2 = genotype_tokens[:, step * 2 + 1] # Shape: (batch_size,)

            # Create a compatibility mask (batch_size, vocab_size)
            # Initialize with False (invalid)
            vocab_size = step_logits.size(-1)
            compatibility_mask = torch.zeros_like(step_logits, dtype=torch.bool) # Use boolean mask

            # Set True for the valid tokens for each sample in the batch
            # Use scatter_ with dim=1 to efficiently set indices based on genotype tokens
            compatibility_mask.scatter_(1, locus_genotype_token1.unsqueeze(1), True)
            compatibility_mask.scatter_(1, locus_genotype_token2.unsqueeze(1), True)

            # Apply the compatibility mask to the logits (set invalid tokens to -inf)
            # Combine with the special token masking already done
            step_logits[~compatibility_mask] = -float('inf')
            # --- End Genotype Compatibility ---

            # Store logits if requested (store the masked logits)
            if return_logits:
                all_step_logits.append(step_logits.clone()) # Clone to avoid modification issues

            # Greedy sampling: take the most likely token from the doubly-masked logits
            predicted_token = torch.argmax(step_logits, dim=-1) # Shape: (batch_size,)

            # --- Debug: Check if predicted token is valid ---
            # for i in range(batch_size):
            #     p_tok = predicted_token[i].item()
            #     g_tok1 = locus_genotype_token1[i].item()
            #     g_tok2 = locus_genotype_token2[i].item()
            #     if p_tok != g_tok1 and p_tok != g_tok2:
            #         logging.warning(f"Prediction Step {step}, Sample {i}: Predicted token {p_tok} is not in genotype ({g_tok1}, {g_tok2}) despite masking!")
            # --- End Debug ---


            # Store prediction
            predicted_sequence[:, step] = predicted_token

            # Prepare input for the next step
            decoder_input_tokens = torch.cat([decoder_input_tokens, predicted_token.unsqueeze(1)], dim=1)

            # Optional: Add EOS token handling if needed

        if return_logits:
            return predicted_sequence, all_step_logits
        else:
            return predicted_sequence # Shape: (batch_size, num_loci)
