import torch
import pandas as pd
import numpy as np
import os
import random
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam # Assuming Adam is the default or configured optimizer

# --- Import TransPhaser Components ---
# Configuration
from .config import HLAPhasingConfig # Use relative import

# Data Preprocessing
from .data_preprocessing import GenotypeDataParser, AlleleTokenizer, CovariateEncoder, HLADataset

# Model Components
from .model import HLAPhasingModel

# Training
from .trainer import HLAPhasingTrainer
from .loss import ELBOLoss, KLAnnealingScheduler

# Evaluation
from .evaluation import HLAPhasingMetrics

# Reporting
from .performance_reporter import PerformanceReporter

# Samplers (Import specific samplers if needed for prediction logic)
# from .samplers import ...

# --- Logging Setup ---
# Configure logging within the class or assume it's configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HLAPhasingRunner:
    """
    Orchestrates the HLA phasing workflow, including data loading, preprocessing,
    model training, prediction, and evaluation, based on a configuration object.
    """
    def __init__(self, config: HLAPhasingConfig):
        """
        Initializes the HLAPhasingRunner.

        Args:
            config (HLAPhasingConfig): Configuration object containing all settings.
        """
        self.config = config
        self.device = torch.device(config.device if hasattr(config, 'device') and torch.cuda.is_available() else "cpu")
        self.output_dir = config.output_dir if hasattr(config, 'output_dir') else "output" # Default output dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"HLAPhasingRunner initialized. Output directory: {self.output_dir}")
        logging.info(f"Using device: {self.device}")

        # Initialize reporter
        self.reporter = PerformanceReporter(output_dir=self.output_dir)


    def _set_seeds(self):
        """Sets random seeds for reproducibility."""
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
        logging.info(f"Set random seed to {seed}")

    def _load_data(self):
        """Loads unphased and phased data."""
        logging.info("Loading data...")
        try:
            # Use paths from config
            unphased_path = self.config.data.unphased_data_path
            phased_path = self.config.data.phased_data_path # Optional, for evaluation
            df_unphased = pd.read_csv(unphased_path)
            df_phased_truth = pd.read_csv(phased_path) if phased_path and os.path.exists(phased_path) else None
            logging.info(f"Loaded unphased data from {unphased_path}")
            if df_phased_truth is not None:
                logging.info(f"Loaded phased ground truth data from {phased_path}")
            return df_unphased, df_phased_truth
        except FileNotFoundError as e:
            logging.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def _preprocess_data(self, df_unphased, df_phased_truth):
        """Performs data preprocessing steps."""
        logging.info("Preprocessing data...")
        loci = self.config.data.locus_columns
        covariate_cols = self.config.data.covariate_columns # Assuming these are all categorical for now based on example
        categorical_covariate_cols = covariate_cols
        numerical_covariate_cols = [] # TODO: Get from config if needed

        # --- Split Data ---
        train_df, val_df = train_test_split(df_unphased, test_size=0.2, random_state=self.config.seed) # TODO: Make test_size configurable

        # --- Initialize Tools ---
        parser = GenotypeDataParser(locus_columns=loci, covariate_columns=covariate_cols)
        tokenizer = AlleleTokenizer()
        cov_encoder = CovariateEncoder(
            categorical_covariates=categorical_covariate_cols,
            numerical_covariates=numerical_covariate_cols
        )

        # --- Build Vocabulary ---
        logging.info("Building allele vocabularies...")
        alleles_by_locus = {}
        for locus in loci:
            locus_alleles = set()
            # Handle potential missing values and different genotype formats ('/' or ',')
            for genotype_str in df_unphased[locus].dropna():
                 # Simple split assuming '/' or ',' - needs robust handling from parser ideally
                 alleles = genotype_str.replace(',', '/').split('/')
                 locus_alleles.update(a for a in alleles if a and a not in ["UNK", "<UNK>"]) # Filter empty/UNK
            alleles_by_locus[locus] = list(locus_alleles)
            tokenizer.build_vocabulary(locus, alleles_by_locus[locus])
            logging.info(f"  {locus}: {tokenizer.get_vocab_size(locus)} tokens")
        self.vocab_sizes = {locus: tokenizer.get_vocab_size(locus) for locus in loci}

        # --- Parse Genotypes (using the parser) ---
        # Note: The parser expects specific input format, might need adjustment
        # For simplicity matching the test mocks, we might bypass the parser here
        # Or assume the parser handles the format in df_unphased correctly
        # Let's assume parser works for now
        logging.info("Parsing genotypes...")
        # This parse method might need adjustment based on actual parser implementation vs DataFrame format
        # train_genotypes_parsed, train_covariates_df = parser.parse(train_df)
        # val_genotypes_parsed, val_covariates_df = parser.parse(val_df)
        # --- Manual parsing like in example script (TEMPORARY until parser is robust) ---
        def parse_df_genotypes_temp(df, loci_list):
            parsed = []
            for _, row in df.iterrows():
                sample_genotype = []
                for locus in loci_list:
                    alleles = row[locus].replace(',', '/').split('/')
                    sample_genotype.append(sorted([a for a in alleles if a])) # Sort and filter empty
                parsed.append(sample_genotype)
            return parsed
        train_genotypes_parsed = parse_df_genotypes_temp(train_df, loci)
        val_genotypes_parsed = parse_df_genotypes_temp(val_df, loci)
        train_covariates_df = train_df[covariate_cols]
        val_covariates_df = val_df[covariate_cols]
        # --- End Temporary Parsing ---


        # --- Encode Covariates ---
        logging.info("Encoding covariates...")
        # Use fit_transform for simplicity, assumes encoder handles train/val split internally if needed
        # Or fit on train, transform train/val separately
        train_covariates_encoded_np = cov_encoder.fit_transform(train_covariates_df).to_numpy(dtype=np.float32)
        val_covariates_encoded_np = cov_encoder.transform(val_covariates_df).to_numpy(dtype=np.float32)
        self.covariate_dim = train_covariates_encoded_np.shape[1]
        logging.info(f"Encoded covariate shapes: Train={train_covariates_encoded_np.shape}, Val={val_covariates_encoded_np.shape}")


        # --- Extract Phased Haplotypes (if available) ---
        train_phased_haplotypes = None
        val_phased_haplotypes = None
        if df_phased_truth is not None:
            df_phased_truth_indexed = df_phased_truth.set_index('IndividualID') # Assumes 'IndividualID' column
            train_ids = train_df['IndividualID']
            val_ids = val_df['IndividualID']
            # Ensure IDs exist in phased data before trying to loc
            train_ids_present = train_ids[train_ids.isin(df_phased_truth_indexed.index)]
            val_ids_present = val_ids[val_ids.isin(df_phased_truth_indexed.index)]
            train_phased_haplotypes = df_phased_truth_indexed.loc[train_ids_present]['Haplotype1'].tolist() # Assumes 'Haplotype1' column
            val_phased_haplotypes = df_phased_truth_indexed.loc[val_ids_present]['Haplotype1'].tolist()
            # TODO: Handle cases where not all train/val IDs have phased data

        # --- Create Datasets ---
        logging.info("Creating datasets...")
        train_dataset = HLADataset(
            genotypes=train_genotypes_parsed,
            covariates=train_covariates_encoded_np,
            phased_haplotypes=train_phased_haplotypes, # Pass None if not available
            tokenizer=tokenizer,
            loci_order=loci
        )
        val_dataset = HLADataset(
            genotypes=val_genotypes_parsed,
            covariates=val_covariates_encoded_np,
            phased_haplotypes=val_phased_haplotypes, # Pass None if not available
            tokenizer=tokenizer,
            loci_order=loci
        )

        # --- Create DataLoaders ---
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, shuffle=False)
        logging.info(f"Created Train DataLoader ({len(train_dataset)} samples) and Validation DataLoader ({len(val_dataset)} samples)")

        # Store necessary objects for later stages
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_df = val_df # Keep val_df for prediction IDs
        self.df_phased_truth = df_phased_truth # Keep for evaluation

    def _build_model(self):
        """Initializes the model, loss, optimizer, and schedulers."""
        logging.info("Initializing model components...")
        num_loci = len(self.config.data.locus_columns)

        # Prepare model configs (adjust based on actual HLAPhasingModel needs)
        encoder_cfg = self.config.model.dict() # Pass model config directly?
        decoder_cfg = self.config.model.dict()
        # Add/override specific keys if needed
        encoder_cfg.update({"vocab_sizes": self.vocab_sizes, "num_loci": num_loci, "covariate_dim": self.covariate_dim, "loci_order": self.config.data.locus_columns})
        decoder_cfg.update({"vocab_sizes": self.vocab_sizes, "num_loci": num_loci, "covariate_dim": self.covariate_dim, "loci_order": self.config.data.locus_columns})


        self.model = HLAPhasingModel(
            num_loci=num_loci,
            allele_vocabularies=self.tokenizer.locus_vocabularies, # Pass vocab dict
            covariate_dim=self.covariate_dim,
            tokenizer=self.tokenizer,
            encoder_config=encoder_cfg,
            decoder_config=decoder_cfg
        ).to(self.device)
        logging.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

        # Loss
        self.loss_fn = ELBOLoss(kl_weight=0.0, reconstruction_weight=1.0) # Start KL at 0 for annealing

        # Optimizer (assuming Adam, make configurable)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.training.learning_rate)

        # Schedulers
        steps_per_epoch = len(self.train_loader)
        self.kl_scheduler = KLAnnealingScheduler(
            anneal_type=self.config.training.kl_annealing_type,
            max_weight=1.0, # TODO: Make configurable
            total_steps=steps_per_epoch * self.config.training.epochs # Anneal over total training? Or just first epoch? Example used 1 epoch.
            # total_steps=steps_per_epoch # Anneal over 1 epoch like example
        )
        # TODO: Add LR scheduler if configured

    def _train_model(self):
        """Initializes and runs the training loop."""
        logging.info("Initializing trainer...")
        self.trainer = HLAPhasingTrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            kl_scheduler=self.kl_scheduler,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.config.training.epochs
            # TODO: Add checkpoint manager, early stopping from config
        )
        logging.info("Starting training...")
        try:
            train_losses, val_losses = self.trainer.train()
            logging.info("Training finished.")
            # Log training history
            self.reporter.log_metric("training_history", {"train_loss": train_losses, "val_loss": val_losses})
        except Exception as e:
            logging.error(f"An error occurred during training: {e}", exc_info=True)
            # Optionally save model state even if training failed
            # self._save_model("failed_training_model.pt")
            raise # Re-raise the exception

    def _predict(self):
        """Performs haplotype prediction on the validation set."""
        logging.info("Predicting haplotypes for the validation set...")
        self.model.eval()
        all_predicted_haplotypes = []
        all_individual_ids = []
        predict_loader = self.val_loader # Use the validation loader

        with torch.no_grad():
            for batch in predict_loader:
                pred_batch = {
                    'genotype_tokens': batch['genotype_tokens'].to(self.device),
                    'covariates': batch['covariates'].to(self.device)
                }
                sample_indices = batch['sample_index'] # Get original indices

                # Assuming model.predict_haplotypes predicts H1 tokens
                predicted_tokens_h1 = self.model.predict_haplotypes(pred_batch) # Shape: (batch, num_loci)

                # Derive H2 (logic copied from example script - needs verification/refinement)
                genotype_tokens_batch = batch['genotype_tokens'].to(self.device)
                num_loci = len(self.config.data.locus_columns)
                predicted_tokens_h2 = torch.zeros_like(predicted_tokens_h1)
                for i in range(predicted_tokens_h1.size(0)):
                    for j in range(num_loci):
                        locus_genotype_token1 = genotype_tokens_batch[i, j * 2]
                        locus_genotype_token2 = genotype_tokens_batch[i, j * 2 + 1]
                        pred_h1_token = predicted_tokens_h1[i, j]
                        if locus_genotype_token1 == pred_h1_token:
                            predicted_tokens_h2[i, j] = locus_genotype_token2
                        elif locus_genotype_token2 == pred_h1_token:
                            predicted_tokens_h2[i, j] = locus_genotype_token1
                        else:
                            logging.warning(f"Sample {i}, Locus {j}: Predicted H1 token {pred_h1_token.item()} not in genotype ({locus_genotype_token1.item()}, {locus_genotype_token2.item()}). Using fallback for H2.")
                            predicted_tokens_h2[i, j] = locus_genotype_token1 # Fallback

                # Detokenize and format
                batch_size = predicted_tokens_h1.shape[0]
                for i in range(batch_size):
                    hap1_alleles = [self.tokenizer.detokenize(self.config.data.locus_columns[j], predicted_tokens_h1[i, j].item()) for j in range(num_loci)]
                    hap2_alleles = [self.tokenizer.detokenize(self.config.data.locus_columns[j], predicted_tokens_h2[i, j].item()) for j in range(num_loci)]
                    hap1_str = "_".join(hap1_alleles)
                    hap2_str = "_".join(hap2_alleles)
                    all_predicted_haplotypes.append(tuple(sorted((hap1_str, hap2_str))))

                # Get corresponding IndividualIDs using original indices
                batch_individual_ids = self.val_df.iloc[sample_indices.cpu().numpy()]['IndividualID'].tolist()
                all_individual_ids.extend(batch_individual_ids)

        # Create DataFrame
        self.predictions_df = pd.DataFrame({
            'IndividualID': all_individual_ids,
            'Predicted_Haplotype1': [haps[0] for haps in all_predicted_haplotypes],
            'Predicted_Haplotype2': [haps[1] for haps in all_predicted_haplotypes]
        })
        # Save predictions (optional, reporter might handle summary)
        pred_file = os.path.join(self.output_dir, "predictions.csv")
        self.predictions_df.to_csv(pred_file, index=False)
        logging.info(f"Predictions saved to {pred_file}")


    def _evaluate(self):
        """Evaluates predictions against ground truth."""
        if self.df_phased_truth is None:
            logging.warning("No ground truth phased data provided. Skipping evaluation.")
            return

        if not hasattr(self, 'predictions_df') or self.predictions_df.empty:
            logging.warning("No predictions available to evaluate.")
            return

        logging.info("Evaluating predictions...")
        eval_df = pd.merge(self.predictions_df, self.df_phased_truth, on='IndividualID', how='inner')

        if eval_df.empty:
            logging.warning("Could not merge predictions with ground truth (check IndividualIDs). Skipping evaluation.")
            return

        evaluator = HLAPhasingMetrics(tokenizer=self.tokenizer)
        try:
            predicted_pairs = [tuple(sorted(pair)) for pair in zip(eval_df['Predicted_Haplotype1'], eval_df['Predicted_Haplotype2'])]
            true_pairs = [tuple(sorted(pair)) for pair in zip(eval_df['Haplotype1'], eval_df['Haplotype2'])] # Assumes truth columns Haplotype1/2

            metrics = evaluator.calculate_metrics(predicted_haplotypes=predicted_pairs, true_haplotypes=true_pairs)
            logging.info(f"Evaluation Metrics: {metrics}")
            self.reporter.log_metric("evaluation_summary", metrics)

        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}", exc_info=True)


    def _finalize(self):
        """Generates final reports and plots."""
        logging.info("Finalizing run and generating reports...")
        # Generate reports
        self.reporter.generate_report(report_format="json", report_filename="final_report.json")
        self.reporter.generate_report(report_format="txt", report_filename="final_report.txt")
        # Plot curves
        self.reporter.plot_training_curves(plot_filename="training_loss_curves.png")
        logging.info("Workflow finished.")


    def run(self):
        """Executes the full HLA phasing workflow."""
        self._set_seeds()
        df_unphased, df_phased_truth = self._load_data()
        self._preprocess_data(df_unphased, df_phased_truth)
        self._build_model()
        self._train_model()
        self._predict()
        self._evaluate()
        self._finalize()

# Example usage (if run directly, though typically instantiated and run)
# if __name__ == '__main__':
#     # Create a dummy config or load from file
#     config = HLAPhasingConfig() # Use defaults or load
#     # Setup necessary config paths if defaults aren't suitable
#     # config.data.unphased_data_path = 'path/to/your/unphased.csv'
#     # config.data.phased_data_path = 'path/to/your/phased.csv' # Optional
#     # config.output_dir = 'results'
#
#     runner = HLAPhasingRunner(config)
#     runner.run()
