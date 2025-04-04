import torch
import torch.nn as nn
import logging

# Import embedding classes
from transphaser.embeddings import AlleleEmbedding, LocusPositionalEmbedding

class GenotypeEncoderTransformer(nn.Module):
    """
    Encoder transformer model for processing unphased genotype data and covariates
    to produce latent representations for parameterizing the approximate posterior q(h|g,c).
    """
    def __init__(self, config):
        """
        Initializes the GenotypeEncoderTransformer.

        Args:
            config (dict): Configuration dictionary containing model hyperparameters
                           such as vocab_sizes, num_loci, embedding_dim, num_heads,
                           num_layers, ff_dim, dropout, covariate_dim, latent_dim etc.
        """
        super().__init__()
        
        # Update default configuration
        default_config = {
            'num_layers': 6,
            'num_heads': 8,
            'hidden_dim': 512,
            'ff_dim': 2048,
            'dropout': 0.1,
            'norm_first': True,  # Keep True for pre-normalization
            'batch_first': True,
            'enable_nested_tensor': False  # Set to False since we're using norm_first=True
        }
        
        # Merge with provided config
        self.config = {**default_config, **config}

        # Extract key parameters from config
        self.vocab_sizes = config["vocab_sizes"]
        self.num_loci = config["num_loci"]
        self.embedding_dim = config["embedding_dim"]
        self.input_seq_len = self.num_loci * 2
        self.covariate_dim = config.get("covariate_dim", 0)
        self.latent_dim = config.get("latent_dim", 64) # Get latent dim from config
        self.loci_order = config.get("loci_order", sorted(list(self.vocab_sizes.keys())))

        if len(self.loci_order) != self.num_loci:
             raise ValueError("Length of loci_order must match num_loci in config.")
        if set(self.loci_order) != set(self.vocab_sizes.keys()):
             raise ValueError("loci_order must contain the same keys as vocab_sizes.")

        # --- Initialize Sub-modules ---
        # 1. Input Embeddings:
        #    - Allele Embeddings (per locus): Use the AlleleEmbedding module.
        self.allele_embedding = AlleleEmbedding(self.vocab_sizes, self.embedding_dim)

        #    - Positional Embeddings: Standard absolute positional embedding for the flattened sequence.
        self.positional_embedding = nn.Embedding(self.input_seq_len, self.embedding_dim)

        #    - Type Embeddings (Optional): To distinguish allele 1 vs allele 2 at a locus.
        self.type_embedding = nn.Embedding(2, self.embedding_dim) # 0 for allele 1, 1 for allele 2

        # 2. Covariate Embedding/Projection (Optional)
        if self.covariate_dim > 0:
            self.covariate_projection = nn.Linear(self.covariate_dim, self.embedding_dim)
        else:
            self.covariate_projection = None

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.config['num_heads'],
            dim_feedforward=self.config['ff_dim'],
            dropout=self.config['dropout'],
            activation='gelu',
            batch_first=self.config['batch_first'],
            norm_first=self.config['norm_first']
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config['num_layers']
        )

        # 4. Output Head (to produce parameters for the posterior distribution)
        #    Projecting pooled output to mean and log_var for a Gaussian posterior.
        self.output_head = nn.Linear(self.embedding_dim, self.latent_dim * 2) # For mean + log_var

        self.dropout = nn.Dropout(self.config['dropout'])

        logging.info("GenotypeEncoderTransformer initialized with sub-modules.")

    def forward(self, genotype_tokens, covariates=None, attention_mask=None):
        """
        Performs a forward pass through the encoder.

        Args:
            genotype_tokens (torch.Tensor): Tensor of input genotype tokens,
                                            shape (batch_size, seq_len), where seq_len
                                            is 2 * num_loci. Assumes alleles are flattened
                                            (e.g., [A1, A2, B1, B2, ...]) and tokens
                                            correspond to the specific locus vocabulary.
            covariates (torch.Tensor, optional): Tensor of encoded covariates,
                                                 shape (batch_size, covariate_dim).
            attention_mask (torch.Tensor, optional): Mask to prevent attention to padding tokens.
                                                     Shape (batch_size, seq_len). True indicates masked.

        Returns:
            torch.Tensor: Parameters for the posterior distribution (e.g., mean and log_var).
                          Shape (batch_size, latent_dim * 2).
        """
        batch_size, seq_len = genotype_tokens.shape
        device = genotype_tokens.device

        if seq_len != self.input_seq_len:
            raise ValueError(f"Input sequence length ({seq_len}) does not match expected ({self.input_seq_len})")

        # --- Input Embeddings ---
        # 1. Allele Embeddings: Manual lookup using AlleleEmbedding module
        allele_embed = torch.zeros(batch_size, seq_len, self.embedding_dim, device=device)
        for i, locus in enumerate(self.loci_order):
            # Alleles for locus 'i' are at positions 2*i and 2*i+1
            pos1 = 2 * i
            pos2 = 2 * i + 1
            tokens1 = genotype_tokens[:, pos1] # Tokens for first allele of locus i
            tokens2 = genotype_tokens[:, pos2] # Tokens for second allele of locus i

            embeds1 = self.allele_embedding.locus_embedders[locus](tokens1) # (batch, embed_dim)
            embeds2 = self.allele_embedding.locus_embedders[locus](tokens2) # (batch, embed_dim)

            allele_embed[:, pos1, :] = embeds1
            allele_embed[:, pos2, :] = embeds2

        # 2. Positional Embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0) # (1, seq)
        pos_embed = self.positional_embedding(positions) # (1, seq, embed_dim)

        # 3. Type Embeddings (0 for first allele, 1 for second allele at each locus)
        type_ids = torch.tensor([0, 1] * self.num_loci, device=device).unsqueeze(0).repeat(batch_size, 1) # (batch, seq)
        type_embed = self.type_embedding(type_ids) # (batch, seq, embed_dim)

        # Combine embeddings
        x = allele_embed + pos_embed + type_embed
        x = self.dropout(x)

        # 4. Add Covariates (if provided) - Add to the sequence embedding
        if covariates is not None and self.covariate_projection is not None:
            cov_embed = self.covariate_projection(covariates).unsqueeze(1) # (batch, 1, embed_dim)
            x = x + cov_embed # Add to all positions

        # 5. Pass through Transformer Encoder Layers
        # src_key_padding_mask should be True for padded tokens
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=attention_mask) # (batch, seq, embed_dim)

        # 6. Process output (e.g., mean pooling over non-padded tokens)
        if attention_mask is None:
            pooled_output = encoder_output.mean(dim=1) # Mean pool across sequence length
        else:
            # Masked mean pooling (avoid averaging padded positions)
            padding_mask_inv = ~attention_mask # Invert mask (True for non-padded)
            masked_output = encoder_output * padding_mask_inv.unsqueeze(-1) # Zero out padded embeddings
            summed_output = masked_output.sum(dim=1)
            num_non_padded = padding_mask_inv.sum(dim=1, keepdim=True) # (batch, 1)
            # Avoid division by zero if a sample is fully padded (shouldn't happen with valid input)
            pooled_output = summed_output / num_non_padded.clamp(min=1e-9)

        # 7. Apply Output Head
        posterior_params = self.output_head(pooled_output) # (batch, latent_dim * 2)

        return posterior_params
