# TransPhaser Examples

This directory contains example scripts demonstrating the usage of the TransPhaser library for HLA phasing.

## Synthetic Data Example

This example shows the end-to-end workflow using synthetically generated data:

1.  **Data Generation:** Create artificial HLA genotype and haplotype data.
2.  **Training:** Train the TransPhaser model on the unphased synthetic genotypes.
3.  **Prediction:** Use the trained model to predict haplotypes for unseen genotypes.
4.  **Evaluation:** Compare the predicted haplotypes against the ground truth phased data.

### Files

*   `generate_synthetic_data.py`: Script to generate synthetic HLA data based on defined allele/haplotype frequencies and covariates.
    *   Outputs CSV files to `examples/data/`:
        *   `synthetic_haplotypes_phased.csv`: Ground truth phased haplotypes.
        *   `synthetic_genotypes_unphased.csv`: Unphased genotypes (input for training/prediction).
        *   `synthetic_genotypes_unphased_missing.csv`: Unphased genotypes with simulated missing alleles.
*   `run_phaser_example.py`: Main script demonstrating the workflow:
    *   Loads the unphased synthetic data (`synthetic_genotypes_unphased.csv`).
    *   Sets up configuration (`HLAPhasingConfig`).
    *   Preprocesses data using `AlleleTokenizer`, `CovariateEncoder`, and `HLADataset`.
    *   Initializes the `HLAPhasingModel`.
    *   Initializes the `HLAPhasingTrainer` and runs a short training loop (or uses a pre-trained model if available).
    *   Performs haplotype prediction on a validation set using a sampler (e.g., `GreedySampler`).
    *   Evaluates the predictions against the ground truth (`synthetic_haplotypes_phased.csv`) using the `Evaluation` class.
    *   Outputs results to `examples/output/`:
        *   `phaser_model_example.pt`: Saved model checkpoint.
        *   `predictions.csv`: Predicted haplotypes.
        *   `evaluation_metrics.txt`: Calculated evaluation metrics (e.g., accuracy, switch error).
*   `data/`: Directory containing the generated synthetic data files.
*   `output/`: Directory where model checkpoints, predictions, and evaluation results are saved.

### How to Run

1.  **Generate Data:**
    ```bash
    python examples/generate_synthetic_data.py
    ```
    This will create the `examples/data/` directory and populate it with CSV files.

2.  **Run the Phaser Workflow:**
    ```bash
    python examples/run_phaser_example.py
    ```
    This will execute the training, prediction, and evaluation steps, saving results to `examples/output/`.

**Note:** The `run_phaser_example.py` script now uses implemented core components from `src/`.

## TODO / Known Issues

*   **Validation Loss NaN:** The validation loss currently becomes `NaN`. This indicates numerical instability during evaluation mode, potentially in the decoder or KL divergence calculation, requiring further debugging (e.g., hyperparameter tuning, architectural adjustments like adding more normalization, checking weight initializations).
*   **Prediction Logic:**
    *   The `HLAPhasingModel.predict_haplotypes` method currently only predicts one haplotype (H1) using greedy decoding.
    *   The example script derives H2 as a placeholder based on the input genotype. A proper implementation should predict H2, potentially ensuring consistency with the input genotype.
    *   More sophisticated sampling methods (e.g., beam search, nucleus sampling) could be implemented.
*   **Evaluation Metrics:** Only phasing accuracy is implemented in `HLAPhasingMetrics`. Hamming distance and Switch Error Rate calculations are needed.
*   **Advanced Trainer Features:** KL annealing, proper checkpointing (saving best model based on validation loss), and early stopping are not implemented in `HLAPhasingTrainer`.
*   **Other `src/evaluation.py` Classes:** `PhasingUncertaintyEstimator`, `HaplotypeCandidateRanker`, and `PhasingResultVisualizer` are still placeholders.
*   **Configuration:** The `HLAPhasingConfig` object currently doesn't store vocabulary/tokenizer information directly; this is handled manually in the example script. Consider adding fields like `vocab_sizes`, `allele_to_ix`, etc., to the config class for better integration.
