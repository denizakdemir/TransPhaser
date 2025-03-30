import torch
import torch.nn as nn
import math
import logging
from typing import Dict, Optional, List

# Import sub-modules
from src.embeddings import AlleleEmbedding, LocusPositionalEmbedding # Import LocusPositionalEmbedding

class HaplotypeDecoderTransformer(nn.Module):
    """
    Decoder-only transformer model for autoregressively generating haplotypes,
    conditioned on covariates. Assumes input represents one haplotype sequence
    across loci (e.g., [A_h1, B_h1, C_h1,...]).
    """
    def __init__(self, config):
        """
        Initializes the HaplotypeDecoderTransformer.

        Args:
            config (dict): Configuration dictionary containing model hyperparameters
                           such as vocab_sizes (dict per locus), num_loci, embedding_dim,
                           num_heads, num_layers, ff_dim, dropout, covariate_dim etc.
                           'max_seq_len' in config should correspond to num_loci.
        """
        super().__init__()
        self.config = config

        # Extract key parameters from config
        self.vocab_sizes = config["vocab_sizes"] # Dict: locus -> vocab_size
        self.num_loci = config["num_loci"]
        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.ff_dim = config["ff_dim"]
        self.dropout_rate = config["dropout"]
        # Sequence length for decoder input during training is num_loci + 1 (BOS + k alleles)
        self.seq_len = self.num_loci + 1 # Adjusted for BOS token
        self.covariate_dim = config.get("covariate_dim", 0)
        self.latent_dim = config.get("latent_dim", 0) # Get latent dim for conditioning
        self.loci_order = config.get("loci_order", sorted(list(self.vocab_sizes.keys()))) # Get or infer locus order
        self.tokenizer = config["tokenizer"] # Expect tokenizer in config

        if len(self.loci_order) != self.num_loci:
             raise ValueError("Length of loci_order must match num_loci in config.")
        if set(self.loci_order) != set(self.vocab_sizes.keys()):
             raise ValueError("loci_order must contain the same keys as vocab_sizes.")

        # --- Initialize Sub-modules ---
        # 1. Allele Embeddings (per locus)
        self.allele_embedding = AlleleEmbedding(self.vocab_sizes, self.embedding_dim)
        # Add BOS token embedding? Or handle it in AlleleEmbedding? Assume AlleleEmbedding handles BOS/EOS/PAD/UNK per locus.

        # 2. Positional Embeddings (using LocusPositionalEmbedding)
        # Needs num_loci and embedding_dim. It handles the mapping internally.
        # Note: LocusPositionalEmbedding expects num_loci, not seq_len (which is num_loci + 1)
        self.positional_embedding = LocusPositionalEmbedding(self.num_loci, self.embedding_dim)

        # 3. Covariate Embedding/Projection (Optional)
        if self.covariate_dim > 0:
            self.covariate_projection = nn.Linear(self.covariate_dim, self.embedding_dim)
        else:
            self.covariate_projection = None

        # 3b. Latent Variable Projection (Optional)
        if self.latent_dim > 0:
            self.latent_projection = nn.Linear(self.latent_dim, self.embedding_dim)
        else:
            self.latent_projection = None

        # 4. Transformer Decoder Layers (using Encoder stack with causal mask)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_dim,
            dropout=self.dropout_rate, # Use original dropout rate
            batch_first=True,
            norm_first=True # Keep pre-normalization
            # activation='relu' # Default, so remove explicit setting
        )
        # Need LayerNorm instance if using default TransformerEncoder wrapper with norm_first=True
        decoder_norm = nn.LayerNorm(self.embedding_dim, eps=1e-3) # Keep increased epsilon
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers, norm=decoder_norm) # Add norm argument back

        # 5. Output Head (predict next allele token)
        #    Using separate heads per locus is more appropriate here.
        self.output_heads = nn.ModuleDict({
            locus: nn.Linear(self.embedding_dim, size)
            for locus, size in self.vocab_sizes.items()
        })

        self.dropout = nn.Dropout(self.dropout_rate)

        # Add LayerNorm for input stabilization after adding conditioning
        self.input_norm = nn.LayerNorm(self.embedding_dim, eps=1e-3) # Increased epsilon

        logging.info("HaplotypeDecoderTransformer initialized with sub-modules (per-locus output heads).")


    def forward(self, input_tokens: torch.Tensor,
                locus_indices: torch.Tensor, # Not strictly needed if using standard positional embedding
                covariates: Optional[torch.Tensor] = None,
                latent_variable: Optional[torch.Tensor] = None, # Added latent variable z
                attention_mask: Optional[torch.Tensor] = None): # Padding mask (batch, seq_len = k+1)
        """
        Performs a forward pass through the decoder.

        Args:
            input_tokens (torch.Tensor): Tensor of input allele tokens including BOS.
                                         Shape (batch_size, seq_len = num_loci + 1).
                                         input_tokens[:, 0] is BOS.
                                         input_tokens[:, i+1] is token for locus i.
            locus_indices (torch.Tensor): Not used with standard positional embedding. Kept for compatibility? Remove later.
            covariates (torch.Tensor, optional): Tensor of encoded covariates. Shape (batch_size, covariate_dim).
            latent_variable (torch.Tensor, optional): Latent variable sample z. Shape (batch_size, latent_dim).
            attention_mask (torch.Tensor, optional): Mask for padding tokens. Shape (batch_size, seq_len). True indicates masked.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping locus name to output logits for that locus.
                                     Shape of each tensor: (batch_size, vocab_size_for_locus).
                                     Logits correspond to the prediction for the *next* token at each position.
        """
        batch_size, seq_len = input_tokens.shape # seq_len should be num_loci + 1
        device = input_tokens.device

        # Removed fixed sequence length check to allow for variable length during generation
        # if seq_len != self.seq_len:
        #      raise ValueError(f"Input seq_len ({seq_len}) doesn't match expected decoder seq_len ({self.seq_len}).")

        # --- Input Embeddings ---
        # 1. Allele Embeddings: Manual lookup, handling BOS separately
        allele_embed = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        # Get BOS embedding - Get ID from the stored tokenizer
        bos_token_id = self.tokenizer.special_tokens.get("BOS", 2)
        # Use the first locus's embedder for BOS for simplicity
        first_locus = self.loci_order[0]
        bos_embeds = self.allele_embedding.locus_embedders[first_locus](
            torch.full((batch_size,), bos_token_id, dtype=torch.long, device=device)
        )
        allele_embed[:, 0, :] = bos_embeds

        # Get embeddings for actual allele tokens (positions 1 to seq_len-1)
        # The loop needs to go up to seq_len-1, accessing locus name by index i
        for i in range(seq_len - 1): # Iterate based on actual input length (excluding BOS)
            locus_name = self.loci_order[i] # Get locus name for this position
            locus_tokens = input_tokens[:, i+1] # Get tokens for locus i (at position i+1)
            embeds = self.allele_embedding.locus_embedders[locus_name](locus_tokens)
            allele_embed[:, i+1, :] = embeds # Corrected indentation

        # 2. Positional Embeddings (using LocusPositionalEmbedding for loci only)
        # Create positional embeddings tensor, initialized to zeros
        pos_embed = torch.zeros_like(allele_embed) # (batch, current_seq_len, embed_dim)

        # Determine how many locus positional embeddings to apply based on current sequence length
        # We apply embeddings to positions 1 up to min(current_seq_len - 1, num_loci)
        num_loci_in_sequence = seq_len - 1 # Number of actual loci tokens present (excluding BOS)
        num_embeddings_to_apply = min(num_loci_in_sequence, self.num_loci)

        if num_embeddings_to_apply > 0:
            # Get the required positional embeddings (indices 0 to num_embeddings_to_apply - 1)
            locus_pos_indices = torch.arange(0, num_embeddings_to_apply, dtype=torch.long, device=device)
            locus_pos_embeds = self.positional_embedding(locus_pos_indices) # Shape (num_embeddings_to_apply, embed_dim)

            # Add these embeddings to the corresponding positions in pos_embed (positions 1 to num_embeddings_to_apply)
            # Slice target: pos_embed[:, 1 : 1 + num_embeddings_to_apply, :]
            # Slice source: locus_pos_embeds.unsqueeze(0) # Shape (1, num_embeddings_to_apply, embed_dim)
            pos_embed[:, 1 : 1 + num_embeddings_to_apply, :] = locus_pos_embeds.unsqueeze(0)
        # Positions beyond num_embeddings_to_apply + 1 (e.g., EOS) and position 0 (BOS) will have zero positional embedding.

        # Combine base embeddings
        x = allele_embed + pos_embed
        # Apply input norm *before* adding conditioning and dropout
        x = self.input_norm(x)
        x = self.dropout(x) # Apply dropout after base embeddings + norm

        # 3. Add Covariates
        if covariates is not None and self.covariate_projection is not None:
            cov_embed = self.covariate_projection(covariates).unsqueeze(1) # (batch, 1, embed_dim)
            # --- Add clamping for stability ---
            cov_embed = torch.clamp(cov_embed, min=-10, max=10)
            # --- End clamping ---
            x = x + cov_embed # Add to all positions

        # 3b. Add Latent Variable
        if latent_variable is not None and self.latent_projection is not None:
            z_embed = self.latent_projection(latent_variable).unsqueeze(1) # (batch, 1, embed_dim)
            # --- Add clamping for stability ---
            z_embed = torch.clamp(z_embed, min=-10, max=10)
            # --- End clamping ---
            x = x + z_embed # Add to all positions

        # LayerNorm was moved to before adding conditioning

        # 4. Generate Causal Mask based on current input seq_len
        causal_mask = self._generate_square_subsequent_mask(seq_len).to(device)

        # 5. Pass through Transformer Layers
        # Use causal_mask for the attention mask within the decoder stack
        # Use attention_mask (padding mask) for src_key_padding_mask
        output = self.transformer_layers(x, mask=causal_mask, src_key_padding_mask=attention_mask) # (batch, seq_len, embed_dim)

        # 6. Apply Output Heads (Per Locus)
        # Output corresponds to prediction for the *next* token.
        # output[:, i, :] corresponds to the hidden state after processing input token i (BOS or allele i).
        # We use output[:, i, :] to predict allele i+1.
        # The output sequence length matches the input sequence length (`seq_len`).
        all_logits = {}
        # The loop should go up to seq_len, using output from step i to predict locus i+1
        for i in range(seq_len): # Iterate based on input seq len (BOS + alleles processed so far)
             # Ensure we don't try to get logits for a locus beyond the defined number
             if i < self.num_loci:
                 locus_name = self.loci_order[i] # Get locus name for the allele being predicted (allele i+1)
                 # Use output from position i (after processing BOS or allele i) to predict allele i+1
                 locus_hidden_state = output[:, i, :] # Shape: (batch, embed_dim)
                 # --- Add hidden state clamping for stability ---
                 locus_hidden_state = torch.clamp(locus_hidden_state, min=-10, max=10) # Clamp hidden state
                 # --- End clamping ---
                 # REMOVED redundant application of final norm: self.transformer_layers.norm()
                 # Use the hidden state directly from the transformer layers
                 all_logits[locus_name] = self.output_heads[locus_name](locus_hidden_state) # Shape: (batch, vocab_size_for_locus)
             # else: # This would correspond to predicting EOS based on the last allele's hidden state
             #     pass # No specific EOS head defined here

        # Return the dictionary of logits per locus
        # Keys are locus names, values are logits for predicting that locus's allele
        # Keys are locus names, values are logits for predicting that locus's allele
        return all_logits


    # Helper method to generate causal mask
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
       """Generates a square causal mask for attending to previous positions."""
       mask = (torch.triu(torch.ones(sz, sz, dtype=torch.bool)) == 1).transpose(0, 1)
       # Fill with float('-inf') where mask is True (positions not allowed to attend)
       # Fill with float(0.0) where mask is False (positions allowed to attend)
       attn_mask = torch.zeros(sz, sz, dtype=torch.float)
       attn_mask.masked_fill_(mask, float('-inf'))
       return attn_mask
