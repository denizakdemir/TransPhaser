# TransPhaser: Transformer-Based HLA Phasing Suite

TransPhaser is a suite for phasing HLA genotypes using transformer-based models. It leverages variational inference and generative models to predict haplotypes from unphased genotype data, potentially incorporating covariates and handling missing data.

## Features

*   **Transformer Architecture:** Utilizes encoder-decoder transformer models for both inferring latent representations and generating haplotypes.
*   **Variational Inference:** Employs an Evidence Lower Bound (ELBO) objective for training, learning an approximate posterior distribution over haplotypes.
*   **k-Locus Support:** Designed to handle phasing across an arbitrary number of HLA loci (k-locus).
*   **Covariate Integration:** Can incorporate covariate information (e.g., population, age) into the phasing process.
*   **Compatibility Enforcement:** Includes mechanisms to ensure generated haplotypes are compatible with the observed genotypes.
*   **Missing Data Handling:** Provides strategies for dealing with missing allele information.
*   **Modular Design:** Built with distinct components for data preprocessing, model architecture, training, evaluation, and configuration.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:denizakdemir/TransPhaser.git
    cd TransPhaser
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

An example workflow is provided in `examples/run_phaser_example.py`.

1.  **Prepare your data:**
    *   **Unphased Genotypes:** A CSV file with columns for individual IDs, HLA loci (e.g., 'HLA-A', 'HLA-B'), and any covariates. Genotypes should be represented as strings like 'A\*01:01/A\*02:01'. See `examples/data/synthetic_genotypes_unphased.csv`.
    *   **(Optional) Phased Haplotypes:** A CSV file with ground truth haplotypes for evaluation. See `examples/data/synthetic_haplotypes_phased.csv`.
2.  **Configure the example script:**
    *   Modify paths in `examples/run_phaser_example.py` (e.g., `DATA_DIR`, `OUTPUT_DIR`) if necessary.
    *   Adjust model and training parameters within the script as needed.
3.  **Run the example:**
    ```bash
    python examples/run_phaser_example.py
    ```

## Output

The example script will generate the following files in the specified output directory (`examples/output/` by default):

*   `phaser_model_example.pt`: The trained model checkpoint.
*   `predictions.csv`: Predicted haplotype pairs for the validation set.
*   `evaluation_metrics.txt`: Phasing accuracy metrics compared to the ground truth (if provided).
*   `loss_curves.png`: A plot showing training and validation loss over epochs.

## Testing

Unit tests are located in the `tests/` directory. You can run them using pytest:

```bash
pip install pytest  # If not already installed
pytest tests/
```

## Project Structure

*   `src/`: Core source code for the TransPhaser components.
*   `examples/`: Example scripts and data.
*   `tests/`: Unit tests.
*   `description.md`: Detailed technical implementation plan (developer reference).
*   `requirements.txt`: Python dependencies.
*   `.gitignore`: Specifies files ignored by Git.

## Contributing

Contributions are welcome! Please refer to the project's contribution guidelines (to be added).
