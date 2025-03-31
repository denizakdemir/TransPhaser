# TransPhaser: Transformer-Based HLA Phasing Suite

TransPhaser is a Python library designed for HLA haplotype phasing using transformer-based models. It leverages variational inference to learn the complex relationship between unphased multi-locus genotypes, optional covariates (like population or age), and the resulting phased haplotypes. The core idea is to model the probability distribution of haplotypes given the genotype and covariates.

## Project Structure

The project is organized into several key modules to promote modularity and maintainability:

-   **`transphaser/`**: Contains the core library code implementing the phasing logic.
    -   `config.py`: Defines Pydantic models (`HLAPhasingConfig`, `DataConfig`, `ModelConfig`, `TrainingConfig`, etc.) for managing all configuration parameters. This ensures type safety and easy validation of settings related to data paths, model hyperparameters, and training options.
    -   `data_preprocessing.py`: Handles the crucial first step of preparing input data.
        -   `GenotypeDataParser`: Parses genotype information from input files (e.g., CSV), supporting various formats and validating input structure.
        -   `AlleleTokenizer`: Creates and manages vocabularies for HLA alleles, mapping allele strings to unique integer tokens for each locus separately. Includes handling of special tokens (PAD, UNK, BOS, EOS).
        -   `CovariateEncoder`: Encodes categorical (e.g., population) and numerical (e.g., age) covariates into numerical representations suitable for the model, using techniques like one-hot encoding and standardization.
        -   `HLADataset`: Implements a PyTorch `Dataset` class to efficiently load and batch preprocessed data (tokenized genotypes, encoded covariates, optional target haplotypes, sample IDs) for training and inference.
    -   `embeddings.py`: Implements specialized embedding layers crucial for transformers.
        -   `AlleleEmbedding`: Creates embedding vectors for allele tokens, potentially using separate embedding spaces for different loci.
        -   `LocusPositionalEmbedding`: Generates embeddings that capture both the position within the haplotype sequence and the identity of the HLA locus at that position.
    -   `encoder.py`: Defines the `GenotypeEncoderTransformer`, a bidirectional transformer model that processes the unphased genotype tokens and covariates to produce a latent representation summarizing the input information. This latent representation parameterizes the approximate posterior distribution.
    -   `decoder.py`: Defines the `HaplotypeDecoderTransformer`, a causal (autoregressive) transformer model that generates one haplotype sequence step-by-step (locus by locus), conditioned on the latent representation from the encoder and covariates.
    -   `model.py`: The central `HLAPhasingModel` class integrates the encoder, decoder, and embedding layers. It defines the overall forward pass for training (calculating ELBO components) and the prediction logic for inference.
    -   `posterior.py`: Defines the `HaplotypePosteriorDistribution` class, which represents the approximate posterior distribution q(h|g, c) parameterized by the encoder's output. It handles sampling from this distribution and calculating KL divergence.
    -   `loss.py`: Implements the loss function used for training.
        -   `ELBOLoss`: Calculates the Evidence Lower Bound, the objective function maximized during training. It combines the reconstruction loss (how well the decoder reconstructs haplotypes) and the KL divergence (regularization term measuring difference between posterior and prior).
        -   `KLAnnealingScheduler`: Manages the weight of the KL divergence term during training, often starting low and increasing ("annealing") to stabilize initial training.
    -   `autoregressive.py`: Contains the `AutoregressiveHaplotypeDecoder` class, which orchestrates the step-by-step generation of haplotype sequences using the decoder transformer, applying sampling strategies (like greedy search) and potentially enforcing constraints.
    -   `compatibility.py`: Implements the `HaplotypeCompatibilityChecker`, a utility to verify if a generated pair of haplotypes is consistent with the original unphased genotype input. This is crucial for ensuring valid outputs.
    -   `missing_data.py`: Includes components designed to handle missing alleles in the input genotype data, such as `MissingDataMarginalizer` (to average over possibilities) and `AlleleImputer` (to fill in missing values). (Functionality might be under development).
    -   `samplers.py`: Intended to house various sampling algorithms (e.g., beam search, nucleus sampling) that can be used during the autoregressive decoding process to generate diverse or high-probability haplotype candidates. (Currently may contain basic implementations).
    -   `trainer.py`: The `HLAPhasingTrainer` class encapsulates the entire training and validation loop, handling epoch iteration, batch processing, optimizer steps, gradient updates, loss calculation, metric logging, and interaction with the KL scheduler and checkpoint manager.
    -   `checkpoint.py`: The `CheckpointManager` class handles saving model state (weights, optimizer state, epoch number) during training and loading checkpoints to resume training or for inference. Supports saving the best model based on validation performance.
    -   `evaluation.py`: Provides tools for assessing phasing performance and understanding model predictions.
        -   `HLAPhasingMetrics`: Calculates standard phasing metrics like phasing accuracy, Hamming distance, and switch error rate by comparing predicted haplotypes to ground truth.
        -   `PhasingUncertaintyEstimator`: Estimates the model's confidence in its predictions, for example, by calculating the entropy of the decoder's output distribution.
        -   (Placeholders for `HaplotypeCandidateRanker` and `PhasingResultVisualizer` suggest future features for ranking multiple predictions and creating plots).
    -   `result_formatter.py`: The `PhasingResultFormatter` class takes the raw model output (predicted haplotypes, scores, uncertainty) and formats it into a user-friendly structure, often JSON or CSV.
    -   `runner.py`: The `HLAPhasingRunner` class provides a high-level interface to simplify running the entire phasing pipeline (data loading, preprocessing, model loading, prediction/training, evaluation, result saving) from a single configuration.
    -   `performance_reporter.py`: The `PerformanceReporter` class generates summary reports and potentially plots detailing the model's performance during training and evaluation.
-   **`examples/`**: Contains example scripts demonstrating how to use the library.
    -   `generate_synthetic_data.py`: A utility script to create synthetic genotype and haplotype data for testing and demonstration purposes.
    -   `run_phaser_example.py`: An end-to-end script showing the typical workflow: loading data, configuring the model, training, predicting, evaluating, and saving results.
    -   `data/`: Directory containing example input CSV files generated by `generate_synthetic_data.py`.
    -   `output/`: Directory where the example script saves its outputs (model checkpoints, predictions CSV, evaluation metrics text file, loss curves plot).
-   **`tests/`**: Contains unit tests (using a framework like `pytest`) for individual modules and functions to ensure correctness and prevent regressions.
-   **`pyproject.toml`**: Project build system configuration (using Poetry or similar), specifying dependencies, package metadata, and tool settings (like linters or formatters).
-   **`README.md`**: The main project README file, typically containing installation instructions, basic usage examples, and contribution guidelines.
-   **`description.md`**: This file, providing a detailed overview of the project's architecture, modules, and components.

## Core Components Explained

1.  **Data Handling (`data_preprocessing.py`, `config.py`)**: This is the foundation. It takes raw input data (like CSV files with genotypes and covariates) and transforms it into numerical tensors that the PyTorch model can understand. Key steps include parsing different genotype formats, creating consistent integer representations (tokens) for alleles specific to each locus, encoding covariates, and organizing everything into efficient batches using `HLADataset` and `DataLoader`. Configuration (`config.py`) drives this process.
2.  **Model Architecture (`encoder.py`, `decoder.py`, `model.py`, `embeddings.py`)**: This defines the neural network. It uses a Variational Autoencoder (VAE) approach. The `GenotypeEncoderTransformer` reads the unphased genotype and covariates and compresses the information into a latent vector. The `HaplotypeDecoderTransformer` takes this latent vector and autoregressively generates a likely haplotype sequence, one allele at a time. Specialized `AlleleEmbedding` and `LocusPositionalEmbedding` layers help the transformers understand the unique nature of HLA data. The `HLAPhasingModel` ties these parts together.
3.  **Variational Inference (`posterior.py`, `loss.py`)**: This is the mathematical core of the VAE. The encoder defines an *approximate posterior* distribution q(h|g, c) over haplotypes given the input. The `ELBOLoss` function is optimized during training. It encourages the decoder to reconstruct valid haplotypes (reconstruction term) while keeping the approximate posterior close to a prior distribution (KL divergence term, managed by `KLAnnealingScheduler`).
4.  **Haplotype Generation (`autoregressive.py`, `compatibility.py`, `samplers.py`)**: During inference (prediction), the `AutoregressiveHaplotypeDecoder` uses the trained decoder transformer to predict the most likely haplotype sequence step-by-step. The `HaplotypeCompatibilityChecker` ensures that the generated pair of haplotypes could actually produce the input genotype. Different `samplers` can be employed to explore various possible haplotypes beyond the single most likely one.
5.  **Training (`trainer.py`, `checkpoint.py`)**: The `HLAPhasingTrainer` orchestrates the learning process. It iterates through the training data, feeds batches to the model, calculates the `ELBOLoss`, computes gradients, updates model weights using an optimizer (like Adam), validates performance on a separate dataset, and saves progress (`CheckpointManager`) periodically and when performance improves.
6.  **Evaluation (`evaluation.py`, `result_formatter.py`, `performance_reporter.py`)**: After training or prediction, these components assess the quality of the results. `HLAPhasingMetrics` compares predictions to known true haplotypes (if available) using standard metrics. `PhasingUncertaintyEstimator` provides insights into the model's confidence. `PhasingResultFormatter` organizes the output neatly, and `PerformanceReporter` summarizes the findings.
7.  **Missing Data (`missing_data.py`)**: Real-world data often has missing alleles. This module provides strategies (like marginalization or imputation) to handle these gaps, allowing the model to still make predictions, albeit potentially with higher uncertainty.
8.  **Execution (`runner.py`, `examples/run_phaser_example.py`)**: These provide user-facing interfaces. The `HLAPhasingRunner` aims to simplify executing the entire pipeline with a single configuration file, while `run_phaser_example.py` serves as a concrete, step-by-step demonstration of how to use the library's components.

This modular structure allows for independent development and testing of each part, facilitating maintenance and future extensions to the TransPhaser library.
