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
-   **`ModelEncoderConfig`** & **`ModelDecoderConfig`**: Define hyperparameters for their respective Transformer modules.
-   **`ModelConfig`**: Defines the main model's architecture.
-   **`TrainingConfig`**: Defines all training parameters.
-   **`HLAPhasingConfig`**: The main configuration class.

### `data_preprocessing.py`

This file contains classes for parsing, tokenizing, and encoding the data.

-   **`GenotypeDataParser`**: Parses genotype data from a pandas DataFrame.
-   **`AlleleTokenizer`**: Manages vocabularies for HLA alleles.
    -   `build_vocabulary_from_dataframe(df, locus_columns)`: Builds vocabularies for multiple loci directly from a DataFrame.
-   **`CovariateEncoder`**: Encodes categorical and numerical covariates.
-   **`HLADataset`**: A PyTorch Dataset for HLA phasing data.

### `embeddings.py`

This file defines the specialized embedding layers for the model.

-   **`LocusPositionalEmbedding(nn.Module)`**: Encodes the position of each locus.
-   **`AlleleEmbedding(nn.Module)`**: Creates separate embedding spaces for the alleles of each locus.

### `encoder.py`

This file defines the `GenotypeEncoderTransformer`, which maps unphased genotypes to a latent space.

-   **`GenotypeEncoderTransformer(nn.Module)`**: A Transformer-based encoder.

### `decoder.py`

This file defines the `HaplotypeDecoderTransformer`, which autoregressively generates a phased haplotype from a latent variable.

-   **`HaplotypeDecoderTransformer(nn.Module)`**: A Transformer-based decoder.

### `model.py`

This file defines the main `HLAPhasingModel`, which encapsulates the VAE architecture.

-   **`HLAPhasingModel(nn.Module)`**: The main VAE model.
    -   `predict_haplotypes(...)`: Generates a haplotype sequence via autoregressive decoding.

### `loss.py`

This file defines the loss function for training the VAE.

-   **`ELBOLoss(nn.Module)`**: Calculates the Evidence Lower Bound (ELBO) loss.
-   **`KLAnnealingScheduler`**: Implements a scheduler to gradually increase the `kl_weight`.

### `trainer.py`

This file defines the `HLAPhasingTrainer` class to manage the training and evaluation loop.

-   **`HLAPhasingTrainer`**: Manages the training and evaluation loop.

### `runner.py`

This file defines the `HLAPhasingRunner` class, which orchestrates the entire workflow.

-   **`HLAPhasingRunner`**:
    -   `__init__(self, config)`: Initializes the runner with a configuration object.
    -   `run(self)`: Executes the full workflow, including data loading, preprocessing, training, prediction, and evaluation.
    -   `train(self)`: Runs the training and validation loops.
    -   `predict(self)`: Runs the prediction and evaluation loops.
