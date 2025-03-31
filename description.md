# TransPhaser: Transformer-Based HLA Phasing Suite

TransPhaser is a Python library designed for HLA haplotype phasing using transformer-based models. It leverages variational inference to learn the relationship between unphased genotypes, covariates, and phased haplotypes.

## Project Structure

The project is organized into several key modules:

-   **`transphaser/`**: Contains the core library code.
    -   `config.py`: Defines Pydantic models for configuration (`HLAPhasingConfig`, `DataConfig`, etc.).
    -   `data_preprocessing.py`: Handles loading, parsing, tokenizing, and encoding input data (`GenotypeDataParser`, `AlleleTokenizer`, `CovariateEncoder`, `HLADataset`).
    -   `embeddings.py`: Implements allele and positional embedding layers (`AlleleEmbedding`, `LocusPositionalEmbedding`).
    -   `encoder.py`: Defines the genotype encoder transformer (`GenotypeEncoderTransformer`).
    -   `decoder.py`: Defines the haplotype decoder transformer (`HaplotypeDecoderTransformer`).
    -   `model.py`: Integrates the encoder and decoder into the main `HLAPhasingModel`.
    -   `posterior.py`: Defines the approximate posterior distribution (`HaplotypePosteriorDistribution`).
    -   `loss.py`: Implements the ELBO loss function (`ELBOLoss`) and KL annealing (`KLAnnealingScheduler`).
    -   `autoregressive.py`: Handles the autoregressive decoding process (`AutoregressiveHaplotypeDecoder`).
    -   `compatibility.py`: Checks haplotype compatibility with genotypes (`HaplotypeCompatibilityChecker`).
    -   `missing_data.py`: Includes components for handling missing data (`MissingDataMarginalizer`, `AlleleImputer`).
    -   `samplers.py`: (Intended for different sampling strategies during generation - currently may contain basic implementations or placeholders).
    -   `trainer.py`: Manages the model training loop (`HLAPhasingTrainer`).
    -   `checkpoint.py`: Handles saving and loading model checkpoints (`CheckpointManager`).
    -   `evaluation.py`: Calculates evaluation metrics and estimates uncertainty (`HLAPhasingMetrics`, `PhasingUncertaintyEstimator`). Also includes placeholders for candidate ranking and visualization.
    -   `result_formatter.py`: Formats the phasing results (`PhasingResultFormatter`).
    -   `runner.py`: Provides a high-level interface to run the phasing process (`HLAPhasingRunner`).
    -   `performance_reporter.py`: Generates reports on model performance (`PerformanceReporter`).
-   **`examples/`**: Contains example scripts demonstrating usage.
    -   `generate_synthetic_data.py`: Script to create sample data.
    -   `run_phaser_example.py`: End-to-end example of training and evaluating the model.
    -   `data/`: Directory for example input data.
    -   `output/`: Directory for example output files (predictions, metrics).
-   **`tests/`**: Contains unit tests for various components.
-   **`pyproject.toml`**: Project configuration and dependencies.
-   **`README.md`**: Main project README file.
-   **`description.md`**: This file, providing an overview of the project structure.

## Core Components

1.  **Data Handling (`data_preprocessing.py`, `config.py`)**: Parses input genotypes (various formats), tokenizes alleles per locus, encodes covariates, and prepares data batches using PyTorch Datasets and DataLoaders. Configuration is managed via Pydantic models.
2.  **Model Architecture (`encoder.py`, `decoder.py`, `model.py`, `embeddings.py`)**: Implements a variational autoencoder structure with transformer-based encoder and decoder. Uses specialized embeddings for alleles and locus positions.
3.  **Variational Inference (`posterior.py`, `loss.py`)**: Defines the approximate posterior distribution and calculates the Evidence Lower Bound (ELBO) for training, including KL annealing.
4.  **Haplotype Generation (`autoregressive.py`, `compatibility.py`, `samplers.py`)**: Performs autoregressive decoding to generate haplotypes, ensuring compatibility with the input genotype.
5.  **Training (`trainer.py`, `checkpoint.py`)**: Manages the training loop, optimization, validation, and checkpointing.
6.  **Evaluation (`evaluation.py`, `result_formatter.py`, `performance_reporter.py`)**: Calculates phasing accuracy, Hamming distance, switch error rate, estimates uncertainty, formats results, and reports performance.
7.  **Missing Data (`missing_data.py`)**: Provides tools for marginalizing over or imputing missing allele data.
8.  **Execution (`runner.py`, `examples/run_phaser_example.py`)**: High-level runner class and example scripts orchestrate the workflow from data loading to evaluation.

This structure allows for modular development, testing, and extension of the HLA phasing capabilities.
