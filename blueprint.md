# Transphaser Codebase Blueprint

## Introduction

This document provides a comprehensive blueprint of the Transphaser codebase. The goal of this project is to perform HLA phasing using a deep learning model. The architecture is based on a Variational Autoencoder (VAE) with a Transformer-based encoder and decoder. This document is intended to be a guide for understanding the codebase and for reconstructing it from scratch if needed.

## Directory Structure

```
.
├── CODE_REVIEW_FINDINGS.md
├── README.md
├── comprehensive_output/
├── description.md
├── examples/
├── pyproject.toml
├── requirements.txt
├── tests/
└── transphaser/
```

- **`transphaser/`**: The core source code for the HLA phasing model.
- **`tests/`**: Unit tests for the components in `transphaser/`.
- **`examples/`**: Example scripts and notebooks demonstrating how to use the `transphaser` library.
- **`comprehensive_output/`**: Directory for storing comprehensive evaluation reports and results.
- **`pyproject.toml`**: Project metadata and dependencies.
- **`requirements.txt`**: Project dependencies.

## Core Components (`transphaser/`)

This directory contains the core logic of the HLA phasing model.

### `config.py`

This file defines the configuration for the model and training process using Pydantic for validation and type hinting.

-   **`DataConfig`**: Defines data-related paths and parameters.
    -   `locus_columns`: List of strings for HLA loci columns.
    -   `covariate_columns`, `categorical_covariate_columns`, `numerical_covariate_columns`: Lists of strings for different types of covariates.
    -   `validation_split_ratio`: Float for splitting data.
    -   `unphased_data_path`, `phased_data_path`: Optional strings for data file paths.
-   **`ModelEncoderConfig`** & **`ModelDecoderConfig`**: Define hyperparameters for their respective Transformer modules.
    -   `hidden_dim`, `num_layers`, `dropout`.
-   **`ModelConfig`**: Defines the main model's architecture.
    -   `embedding_dim`, `num_heads`, `latent_dim`.
    -   `encoder`: Nested `ModelEncoderConfig`.
    -   `decoder`: Nested `ModelDecoderConfig`.
-   **`TrainingConfig`**: Defines all training parameters.
    -   `learning_rate`, `batch_size`, `epochs`, `optimizer`.
    -   `lr_scheduler`: Optional nested `TrainingLRSchedulerConfig`.
    -   `gradient_accumulation_steps`: Integer for accumulating gradients.
    -   `kl_annealing_type`, `kl_annealing_epochs`, `kl_annealing_max_weight`, `reconstruction_weight`: Parameters for KL annealing in the VAE loss.
    -   `checkpoint_dir`, `checkpoint_frequency`, `final_model_filename`: Checkpointing settings.
    -   `early_stopping_patience`, `log_interval`: Early stopping and logging settings.
-   **`HLAPhasingConfig`**: The main configuration class.
    -   It aggregates all other configuration objects (`DataConfig`, `ModelConfig`, `TrainingConfig`, etc.).
    -   The `__init__` method can load settings from a JSON file and override them with keyword arguments.
    -   A `save` method serializes the current configuration to a JSON file.

### `embeddings.py`

This file defines the specialized embedding layers for the model.

-   **`LocusPositionalEmbedding(nn.Module)`**: Encodes the position of each locus.
    -   `__init__(self, num_loci, embedding_dim)`: Initializes a standard `torch.nn.Embedding` layer of size `(num_loci, embedding_dim)`.
    -   `forward(self, locus_indices)`: Takes a tensor of locus indices `(...,)` and returns their corresponding embedding vectors `(..., embedding_dim)`.
-   **`AlleleEmbedding(nn.Module)`**: Creates separate embedding spaces for the alleles of each locus.
    -   `__init__(self, vocab_sizes, embedding_dim)`: Takes a dictionary `vocab_sizes` mapping locus names to their vocabulary size. It creates a `torch.nn.ModuleDict` containing a separate `nn.Embedding` layer for each locus.
    -   `forward(self, allele_tokens_per_locus)`: Takes a dictionary mapping locus names to allele token tensors. It retrieves the appropriate embedding layer for each locus and returns a dictionary of the resulting embedding tensors.

### `encoder.py`

This file defines the `GenotypeEncoderTransformer`, which maps unphased genotypes to a latent space.

-   **`GenotypeEncoderTransformer(nn.Module)`**:
    -   **Initialization (`__init__`)**:
        -   `self.allele_embedding`: An `AlleleEmbedding` module.
        -   `self.positional_embedding`: A `nn.Embedding` for the absolute position of each allele in the flattened input sequence (`num_loci * 2`, `embedding_dim`).
        -   `self.type_embedding`: A `nn.Embedding` of size `(2, embedding_dim)` to create a vector for "allele 1" vs. "allele 2" at each locus.
        -   `self.covariate_projection`: An optional `nn.Linear` layer to project covariates to `embedding_dim`.
        -   `self.transformer_encoder`: A `nn.TransformerEncoder` stack. Each `nn.TransformerEncoderLayer` uses GELU activation, pre-layer normalization (`norm_first=True`), and dropout.
        -   `self.output_head`: A `nn.Linear` layer that maps the final pooled representation to the latent space parameters (`embedding_dim` -> `latent_dim * 2` for mean and log-variance).
    -   **Forward Pass (`forward`)**:
        1.  **Input Embeddings**: The final input embedding for each token is the sum of three parts:
            1.  **Allele Embedding**: Retrieved from `self.allele_embedding` based on the locus and allele token.
            2.  **Positional Embedding**: Based on the token's absolute position (0 to `2*k-1`).
            3.  **Type Embedding**: Based on whether the token is the first or second allele for its locus.
        2.  **Covariates**: If covariates are provided, they are projected by `self.covariate_projection` and added to every position in the sequence embedding.
        3.  **Transformer**: The combined embeddings are passed through the `self.transformer_encoder`.
        4.  **Pooling**: The output sequence is pooled into a single vector using masked mean pooling (to ignore padding).
        5.  **Output Head**: The pooled vector is passed through `self.output_head` to produce the mean (`mu`) and log-variance (`log_var`) of the latent distribution.

### `decoder.py`

This file defines the `HaplotypeDecoderTransformer`, which autoregressively generates a phased haplotype from a latent variable.

-   **`HaplotypeDecoderTransformer(nn.Module)`**:
    -   **Initialization (`__init__`)**:
        -   `self.allele_embedding`: An `AlleleEmbedding` module.
        -   `self.positional_embedding`: A `LocusPositionalEmbedding` module.
        -   `self.latent_projection` & `self.covariate_projection`: Optional `nn.Linear` layers to project the latent variable and covariates to `embedding_dim`.
        -   `self.transformer_layers`: A `nn.TransformerEncoder` stack, which functions as a decoder because it is always used with a causal (look-ahead) mask.
        -   `self.output_heads`: A `nn.ModuleDict` containing a separate `nn.Linear` output layer for each locus. Each layer maps `embedding_dim` to that locus's specific vocabulary size.
    -   **Forward Pass (`forward`)**:
        1.  **Masking**: A causal mask is generated to prevent attention to future tokens.
        2.  **Input Embeddings**: The input embedding is the sum of the allele embedding and the locus-positional embedding. The BOS token is handled separately.
        3.  **Conditioning**: The latent variable and covariates are projected and added to every position in the sequence embedding.
        4.  **Transformer**: The conditioned embeddings are passed through the `self.transformer_layers` with the causal mask applied.
        5.  **Output Heads**: The transformer's output hidden state at each position `i` is used to predict the token for the *next* position `i+1`. The hidden state at position `i` is fed into the locus-specific output head corresponding to locus `i` to produce logits over that locus's vocabulary.
        6.  **Return**: The function returns a dictionary mapping each locus name to its corresponding output logits.

### `model.py`

This file defines the main `HLAPhasingModel`, which encapsulates the VAE architecture.

-   **`HLAPhasingModel(nn.Module)`**:
    -   `__init__(...)`: Initializes the `GenotypeEncoderTransformer` and `HaplotypeDecoderTransformer` modules.
    -   `reparameterize(self, mu, log_var)`: Performs the reparameterization trick: `z = mu + eps * std`, where `std = torch.exp(0.5 * log_var)`.
    -   **Forward Pass (`forward`)**:
        1.  The `encoder` produces `mu` and `log_var` from the input genotype.
        2.  `mu` and `log_var` are clamped for numerical stability.
        3.  A latent variable `z` is sampled using the `reparameterize` method.
        4.  The **KL Divergence** `KL(q(z|g) || p(z))` is calculated analytically using `mu` and `log_var` against a standard normal prior `p(z) = N(0, I)`. The formula is `-0.5 * sum(1 + log_var - mu^2 - exp(log_var))`.
        5.  The `decoder` receives `z`, covariates, and the target haplotype (for teacher forcing) to produce output `logits`.
        6.  The **Reconstruction Log Probability** `log p(h|z)` is calculated by applying a cross-entropy loss between the decoder's `logits` and the target haplotype tokens for each locus and summing the results.
        7.  Returns a dictionary containing `reconstruction_log_prob`, `kl_divergence`, and the raw `logits`.
    -   `predict_haplotypes(...)`: Generates a haplotype sequence via autoregressive decoding.
        1.  Encodes the genotype to get `mu` (and `log_var`, which is ignored). `z` is set to `mu` for deterministic prediction.
        2.  Starts the generation with a BOS (Beginning-Of-Sequence) token.
        3.  Iteratively feeds the currently generated sequence into the decoder to predict the next token.
        4.  At each step, logits are masked to only allow alleles present in the original input genotype, preventing invalid predictions.
        5.  Greedy sampling is used to select the token with the highest probability at each step.

### `loss.py`

This file defines the loss function for training the VAE.

-   **`ELBOLoss(nn.Module)`**:
    -   `forward(self, model_output)`: Calculates the final training loss, which is a weighted sum of the reconstruction loss and the KL divergence. The objective is to minimize the negative ELBO:
        `Loss = - (reconstruction_log_prob - kl_weight * kl_divergence)`
        `Loss = (-reconstruction_log_prob) + (kl_weight * kl_divergence)`
-   **`KLAnnealingScheduler`**:
    -   Implements a scheduler to gradually increase the `kl_weight` in the `ELBOLoss` from 0 to a maximum value over a set number of training steps. This helps the model learn to reconstruct well before being heavily penalized for the KL divergence term.
    -   Supports `linear`, `sigmoid`, and `cyclical` annealing schedules.

### `trainer.py`

This file defines the `HLAPhasingTrainer` class to manage the training and evaluation loop.

-   **`HLAPhasingTrainer`**:
    -   `__init__(...)`: Stores the model, data loaders, optimizer, loss function, schedulers, and other training configuration.
    -   **`train_epoch(self)`**:
        -   Iterates over the `train_loader`.
        -   For each batch, it performs a forward pass, calculates the loss, and calls `loss.backward()`.
        -   Implements gradient accumulation by only calling `optimizer.step()` and `optimizer.zero_grad()` every `grad_accumulation_steps` batches.
        -   Updates the learning rate and KL annealing schedulers after each optimizer step.
    -   **`evaluate(self)`**:
        -   Iterates over the `val_loader` with gradients disabled (`torch.no_grad()`).
        -   Calculates and returns the average validation loss.
    -   **`train(self)`**:
        -   The main training loop that runs for a specified number of `epochs`.
        -   Calls `train_epoch` and `evaluate` in each epoch.
        -   Implements checkpointing (saving the model periodically and whenever a new best validation loss is achieved).
        -   Implements early stopping (stops training if validation loss does not improve for a given number of epochs).
        -   Saves the final model at the end of training.
