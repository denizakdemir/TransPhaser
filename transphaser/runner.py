import torch
import pandas as pd
import numpy as np
import os
import random
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import Adam # Assuming Adam is the default or configured optimizer
# Import LR Schedulers
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR # Example schedulers

# --- Import TransPhaser Components ---
# Configuration
from transphaser.config import HLAPhasingConfig # Use relative import

# Data Preprocessing
from transphaser.data_preprocessing import GenotypeDataParser, AlleleTokenizer, CovariateEncoder, HLADataset

# Model Components
from transphaser.model import HLAPhasingModel

# Training
from transphaser.trainer import HLAPhasingTrainer
from transphaser.loss import ELBOLoss, KLAnnealingScheduler

# Evaluation
from transphaser.evaluation import HLAPhasingMetrics

# Reporting
from transphaser.performance_reporter import PerformanceReporter

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
        covariate_cols = self.config.data.covariate_columns if hasattr(self.config.data, 'covariate_columns') else []

        # Determine categorical and numerical covariate columns from config or defaults
        # Use getattr for safe access and default to None, then use 'or []' to ensure list type
        categorical_covariate_cols = getattr(self.config.data, 'categorical_covariate_columns', None) or []
        numerical_covariate_cols = getattr(self.config.data, 'numerical_covariate_columns', None) or []

        # Check if specific lists were provided or if we should default to using covariate_columns
        if categorical_covariate_cols or numerical_covariate_cols:
             # Ensure all specified columns are in the main covariate_cols list (or loaded data)
             all_specified_covs = set(categorical_covariate_cols) | set(numerical_covariate_cols)
             if not all_specified_covs.issubset(set(covariate_cols)):
                 logging.warning(f"Specified categorical/numerical covariates ({all_specified_covs}) contain columns not listed in 'covariate_columns' ({covariate_cols}). Check config.")
                 # Adjust covariate_cols to include all specified ones if necessary? Or error?
                 # For now, proceed assuming the lists in config are correct subsets.
        elif covariate_cols:
             # Default: If specific lists are empty/None but covariate_columns is not, assume all are categorical
             logging.warning("Config 'data' section missing specific 'categorical_covariate_columns' or 'numerical_covariate_columns'. Assuming all 'covariate_columns' are categorical.")
             categorical_covariate_cols = covariate_cols
             numerical_covariate_cols = []
        # else: # Both specific lists and covariate_cols are empty/None
             # categorical_covariate_cols and numerical_covariate_cols are already []


        # --- Split Data ---
        val_split_ratio = self.config.data.validation_split_ratio if hasattr(self.config.data, 'validation_split_ratio') else 0.2 # Default to 0.2 if not set
        train_df, val_df = train_test_split(df_unphased, test_size=val_split_ratio, random_state=self.config.seed)
        logging.info(f"Split data into Train ({len(train_df)} samples) and Validation ({len(val_df)} samples) using ratio {val_split_ratio}.")


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
        train_covariates_df = train_df[covariate_cols] if covariate_cols else pd.DataFrame(index=train_df.index) # Handle case with no covariates
        val_covariates_df = val_df[covariate_cols] if covariate_cols else pd.DataFrame(index=val_df.index)
        # --- End Temporary Parsing ---


        # --- Encode Covariates ---
        logging.info("Encoding covariates...")
        if not train_covariates_df.empty:
            train_covariates_encoded_np = cov_encoder.fit_transform(train_covariates_df).to_numpy(dtype=np.float32)
            val_covariates_encoded_np = cov_encoder.transform(val_covariates_df).to_numpy(dtype=np.float32)
            self.covariate_dim = train_covariates_encoded_np.shape[1]
            logging.info(f"Encoded covariate shapes: Train={train_covariates_encoded_np.shape}, Val={val_covariates_encoded_np.shape}")
        else:
            # Handle case with no covariates
            num_train_samples = len(train_df)
            num_val_samples = len(val_df)
            # Create dummy zero arrays with shape (num_samples, 0) or (num_samples, 1) with zeros?
            # Let's use shape (num_samples, 0) to indicate zero dimension
            train_covariates_encoded_np = np.zeros((num_train_samples, 0), dtype=np.float32)
            val_covariates_encoded_np = np.zeros((num_val_samples, 0), dtype=np.float32)
            self.covariate_dim = 0
            logging.info("No covariates specified or found. Covariate dimension set to 0.")


        # --- Extract Phased Haplotypes (if available) ---
        train_phased_haplotypes = None
        val_phased_haplotypes = None
        if df_phased_truth is not None:
            # Ensure 'IndividualID' exists in both dataframes
            if 'IndividualID' not in df_unphased.columns or 'IndividualID' not in df_phased_truth.columns:
                 logging.error("Missing 'IndividualID' column in unphased or phased data. Cannot match ground truth.")
                 # Decide how to proceed: raise error, continue without truth, etc.
                 # For now, log error and proceed without phased data for datasets
                 df_phased_truth = None # Effectively disable using truth data
            else:
                df_phased_truth_indexed = df_phased_truth.set_index('IndividualID')
                train_ids = train_df['IndividualID']
                val_ids = val_df['IndividualID']
                # Ensure IDs exist in phased data before trying to loc
                train_ids_present = train_ids[train_ids.isin(df_phased_truth_indexed.index)]
                val_ids_present = val_ids[val_ids.isin(df_phased_truth_indexed.index)]

                if len(train_ids_present) < len(train_ids):
                    logging.warning(f"Missing phased ground truth for {len(train_ids) - len(train_ids_present)} training samples.")
                if len(val_ids_present) < len(val_ids):
                    logging.warning(f"Missing phased ground truth for {len(val_ids) - len(val_ids_present)} validation samples.")

                # Extract haplotypes only for present IDs
                # Assuming truth columns are 'Haplotype1', 'Haplotype2'
                if 'Haplotype1' in df_phased_truth_indexed.columns and 'Haplotype2' in df_phased_truth_indexed.columns:
                    # We might only need one for the dataset target, but store both for evaluation later
                    train_phased_haplotypes_h1 = df_phased_truth_indexed.loc[train_ids_present]['Haplotype1'].tolist()
                    val_phased_haplotypes_h1 = df_phased_truth_indexed.loc[val_ids_present]['Haplotype1'].tolist()
                    # For the dataset, we typically need a target sequence. Let's use Haplotype1.
                    # Need to map these back to the original train/val_df indices if the dataset expects aligned data.
                    # This mapping can be complex if IDs are missing.
                    # Simpler approach: Filter the datasets later or handle missing targets in the dataset class.
                    # Let's pass the extracted lists directly for now, assuming dataset handles potential misalignment or filtering.
                    train_phased_haplotypes = train_phased_haplotypes_h1 # Use H1 as target for dataset
                    val_phased_haplotypes = val_phased_haplotypes_h1   # Use H1 as target for dataset
                    # Store the full truth pairs for evaluation, aligned with the val_df IDs that have truth
                    self.val_phased_truth_pairs = [tuple(sorted(pair)) for pair in zip(
                        df_phased_truth_indexed.loc[val_ids_present]['Haplotype1'],
                        df_phased_truth_indexed.loc[val_ids_present]['Haplotype2']
                    )]
                    self.val_ids_with_truth = val_ids_present.tolist() # Keep track of which val IDs have truth
                else:
                    logging.warning("Ground truth file missing 'Haplotype1' or 'Haplotype2' columns. Cannot extract phased data.")
                    df_phased_truth = None # Disable truth usage


        # --- Create Datasets ---
        logging.info("Creating datasets...")
        # Note: The dataset needs to handle cases where phased_haplotypes might be shorter
        # than genotypes/covariates if ground truth is missing for some samples.
        train_dataset = HLADataset(
            genotypes=train_genotypes_parsed,
            covariates=train_covariates_encoded_np,
            phased_haplotypes=train_phased_haplotypes, # Pass potentially shorter list or None
            tokenizer=tokenizer,
            loci_order=loci,
            # Add sample IDs if dataset needs to handle misalignment
            sample_ids=train_df['IndividualID'].tolist() if 'IndividualID' in train_df.columns else None
        )
        val_dataset = HLADataset(
            genotypes=val_genotypes_parsed,
            covariates=val_covariates_encoded_np,
            phased_haplotypes=val_phased_haplotypes, # Pass potentially shorter list or None
            tokenizer=tokenizer,
            loci_order=loci,
            sample_ids=val_df['IndividualID'].tolist() if 'IndividualID' in val_df.columns else None
        )

        # --- Create DataLoaders ---
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, shuffle=False)
        logging.info(f"Created Train DataLoader ({len(train_dataset)} samples) and Validation DataLoader ({len(val_dataset)} samples)")

        # Store necessary objects for later stages
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_df = val_df # Keep val_df for prediction IDs and matching truth
        self.df_phased_truth = df_phased_truth # Keep original loaded truth for reference

    def _build_model(self):
        """Initializes the model, loss, optimizer, and schedulers."""
        logging.info("Initializing model components...")
        num_loci = len(self.config.data.locus_columns)

        # Prepare model configs (adjust based on actual HLAPhasingModel needs)
        # Pass the whole config sub-dictionaries if the model expects them
        encoder_cfg = self.config.model.encoder.dict() if hasattr(self.config.model, 'encoder') else self.config.model.dict()
        decoder_cfg = self.config.model.decoder.dict() if hasattr(self.config.model, 'decoder') else self.config.model.dict()

        # Add/override specific keys derived during preprocessing
        # Ensure these keys match what the model components expect
        common_updates = {
            "vocab_sizes": self.vocab_sizes,
            "num_loci": num_loci,
            "covariate_dim": self.covariate_dim,
            "loci_order": self.config.data.locus_columns,
            "padding_idx": self.tokenizer.pad_token_id # Assuming tokenizer has pad_token_id
        }
        encoder_cfg.update(common_updates)
        decoder_cfg.update(common_updates)
        # Add latent_dim if it's defined at the top model level
        if hasattr(self.config.model, 'latent_dim'):
             encoder_cfg['latent_dim'] = self.config.model.latent_dim
             decoder_cfg['latent_dim'] = self.config.model.latent_dim


        self.model = HLAPhasingModel(
            num_loci=num_loci,
            allele_vocabularies=self.tokenizer.locus_vocabularies, # Pass vocab dict
            covariate_dim=self.covariate_dim,
            tokenizer=self.tokenizer,
            encoder_config=encoder_cfg,
            decoder_config=decoder_cfg,
            latent_dim=self.config.model.latent_dim # Pass latent_dim here too
        ).to(self.device)
        logging.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

        # Loss
        # Get reconstruction weight from config, default to 1.0
        recon_weight = self.config.training.reconstruction_weight if hasattr(self.config.training, 'reconstruction_weight') else 1.0
        self.loss_fn = ELBOLoss(kl_weight=0.0, reconstruction_weight=recon_weight) # Start KL at 0 for annealing

        # Optimizer (assuming Adam, make configurable later if needed)
        self.optimizer = Adam(self.model.parameters(), lr=self.config.training.learning_rate)

        # Schedulers
        steps_per_epoch = len(self.train_loader)
        kl_max_weight = self.config.training.kl_annealing_max_weight if hasattr(self.config.training, 'kl_annealing_max_weight') else 1.0
        # Determine total steps for annealing based on config (e.g., over N epochs or total steps)
        anneal_epochs = self.config.training.kl_annealing_epochs if hasattr(self.config.training, 'kl_annealing_epochs') else 1 # Default to 1 epoch
        total_anneal_steps = steps_per_epoch * anneal_epochs

        self.kl_scheduler = KLAnnealingScheduler(
            anneal_type=self.config.training.kl_annealing_type,
            max_weight=kl_max_weight,
            total_steps=total_anneal_steps
        )
        logging.info(f"KL Annealing configured: type={self.config.training.kl_annealing_type}, max_weight={kl_max_weight}, steps={total_anneal_steps}")

        # LR Scheduler (Optional)
        self.lr_scheduler = None
        # Use getattr for safer access to lr_scheduler attribute
        lr_scheduler_config = getattr(self.config.training, 'lr_scheduler', None)
        if lr_scheduler_config:
            # Use dictionary access for nested config
            scheduler_type = lr_scheduler_config.get('type')
            # Get args, default to empty dict if 'args' key is missing or None
            scheduler_args = lr_scheduler_config.get('args', {}) or {}

            if not scheduler_type:
                 logging.warning("LR scheduler configuration found but 'type' is missing. No LR scheduler will be used.")
            else:
                logging.info(f"Configuring LR Scheduler: type={scheduler_type}, args={scheduler_args}")
                try:
                    if scheduler_type == 'StepLR':
                        self.lr_scheduler = StepLR(self.optimizer, **scheduler_args)
                        logging.info("StepLR scheduler configured.")
                    elif scheduler_type == 'CosineAnnealingLR':
                         # Ensure T_max is provided or calculated if needed
                         if 'T_max' not in scheduler_args:
                             # Default T_max to total training steps if not provided
                             scheduler_args['T_max'] = self.config.training.epochs * steps_per_epoch
                             logging.info(f"CosineAnnealingLR 'T_max' not specified, defaulting to total steps: {scheduler_args['T_max']}")
                         self.lr_scheduler = CosineAnnealingLR(self.optimizer, **scheduler_args)
                         logging.info("CosineAnnealingLR scheduler configured.")
                    else:
                        logging.warning(f"Unsupported LR scheduler type: {scheduler_type}. No LR scheduler will be used.")
                except TypeError as e:
                     logging.error(f"Error initializing LR scheduler '{scheduler_type}' with args {scheduler_args}: {e}. Check config arguments.")
                     self.lr_scheduler = None # Ensure scheduler is None if init fails
                except Exception as e:
                     logging.error(f"Unexpected error initializing LR scheduler '{scheduler_type}': {e}", exc_info=True)
                     self.lr_scheduler = None

            # Remove the old placeholder warning
            # logging.warning(f"LR Scheduler '{scheduler_type}' configured but not yet implemented in runner. Add specific instantiation logic.")


    def _train_model(self):
        """Initializes and runs the training loop."""
        logging.info("Initializing trainer...")

        # Get trainer parameters from config
        checkpoint_dir = os.path.join(self.output_dir, self.config.training.checkpoint_dir) if hasattr(self.config.training, 'checkpoint_dir') else os.path.join(self.output_dir, "checkpoints")
        checkpoint_freq = self.config.training.checkpoint_frequency if hasattr(self.config.training, 'checkpoint_frequency') else 1
        early_stopping_patience = self.config.training.early_stopping_patience if hasattr(self.config.training, 'early_stopping_patience') else None
        final_model_filename = self.config.training.final_model_filename if hasattr(self.config.training, 'final_model_filename') else "final_model.pt"
        grad_accum = self.config.training.gradient_accumulation_steps if hasattr(self.config.training, 'gradient_accumulation_steps') else 1
        log_interval = self.config.training.log_interval if hasattr(self.config.training, 'log_interval') else 50


        self.trainer = HLAPhasingTrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            kl_scheduler=self.kl_scheduler,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler, # Pass the (potentially None) LR scheduler
            device=self.device,
            epochs=self.config.training.epochs,
            grad_accumulation_steps=grad_accum,
            log_interval=log_interval,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=checkpoint_freq,
            early_stopping_patience=early_stopping_patience,
            final_model_filename=final_model_filename
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
                # Ensure all required inputs for prediction are present and on device
                pred_batch = {}
                required_keys = ['genotype_tokens', 'covariates'] # Add others if model.predict needs them
                for key in required_keys:
                    if key not in batch:
                        logging.error(f"Missing key '{key}' in prediction batch. Skipping batch.")
                        continue # Or handle differently
                    # Handle covariates potentially being empty (shape [batch_size, 0])
                    if isinstance(batch[key], torch.Tensor):
                         pred_batch[key] = batch[key].to(self.device)
                    else: # Should not happen if dataset prepares tensors correctly
                         logging.warning(f"Unexpected type for key '{key}' in prediction batch: {type(batch[key])}")
                         pred_batch[key] = batch[key] # Pass as is? Risky.

                # Check if we have IndividualIDs to track predictions
                if 'sample_id' not in batch or not batch['sample_id']:
                     logging.error("Missing 'sample_id' in prediction batch. Cannot track predictions. Skipping batch.")
                     continue

                sample_ids_batch = batch['sample_id'] # Get original IDs/indices

                try:
                    # Assuming model.predict_haplotypes predicts H1 tokens
                    # Pass necessary inputs from pred_batch
                    predicted_tokens_h1 = self.model.predict_haplotypes(
                        genotype_tokens=pred_batch['genotype_tokens'],
                        covariates=pred_batch['covariates']
                        # Add other args if needed by predict_haplotypes
                    ) # Shape: (batch, num_loci)

                    # Derive H2 (logic copied from example script - needs verification/refinement)
                    genotype_tokens_batch = pred_batch['genotype_tokens'] # Already on device
                    num_loci = len(self.config.data.locus_columns)
                    predicted_tokens_h2 = torch.zeros_like(predicted_tokens_h1)
                    for i in range(predicted_tokens_h1.size(0)):
                        for j in range(num_loci):
                            # Genotype tokens are flattened pairs [locus1_a1, locus1_a2, locus2_a1, ...]
                            locus_genotype_token1 = genotype_tokens_batch[i, j * 2]
                            locus_genotype_token2 = genotype_tokens_batch[i, j * 2 + 1]
                            pred_h1_token = predicted_tokens_h1[i, j]

                            # Handle padding tokens if necessary
                            pad_token_id = self.tokenizer.pad_token_id
                            if pred_h1_token == pad_token_id:
                                predicted_tokens_h2[i, j] = pad_token_id
                                continue

                            # Basic H2 derivation logic
                            if locus_genotype_token1 == pred_h1_token and locus_genotype_token2 != pad_token_id:
                                predicted_tokens_h2[i, j] = locus_genotype_token2
                            elif locus_genotype_token2 == pred_h1_token and locus_genotype_token1 != pad_token_id:
                                predicted_tokens_h2[i, j] = locus_genotype_token1
                            elif locus_genotype_token1 == locus_genotype_token2: # Homozygous case
                                predicted_tokens_h2[i, j] = locus_genotype_token1 # H2 must be the same
                            else:
                                # This case implies pred_h1 is not one of the genotype alleles (or one is pad)
                                # This shouldn't happen if model predicts valid alleles from genotype
                                logging.warning(f"Sample {sample_ids_batch[i]}, Locus {self.config.data.locus_columns[j]}: Predicted H1 token {pred_h1_token.item()} not compatible with genotype ({locus_genotype_token1.item()}, {locus_genotype_token2.item()}). Using fallback for H2.")
                                # Fallback: use the first non-pad allele from genotype? Or UNK?
                                fallback_token = locus_genotype_token1 if locus_genotype_token1 != pad_token_id else locus_genotype_token2
                                predicted_tokens_h2[i, j] = fallback_token if fallback_token != pad_token_id else self.tokenizer.unk_token_id # Use UNK if both are pad?

                    # Detokenize and format
                    batch_size = predicted_tokens_h1.shape[0]
                    for i in range(batch_size):
                        hap1_alleles = [self.tokenizer.detokenize(self.config.data.locus_columns[j], predicted_tokens_h1[i, j].item()) for j in range(num_loci)]
                        hap2_alleles = [self.tokenizer.detokenize(self.config.data.locus_columns[j], predicted_tokens_h2[i, j].item()) for j in range(num_loci)]
                        # Join alleles, handling potential None from detokenize (e.g., for PAD)
                        hap1_str = "_".join(filter(None, hap1_alleles))
                        hap2_str = "_".join(filter(None, hap2_alleles))
                        # Store sorted pair
                        all_predicted_haplotypes.append(tuple(sorted((hap1_str, hap2_str))))
                        all_individual_ids.append(sample_ids_batch[i]) # Append the actual ID

                except Exception as e:
                    logging.error(f"Error during prediction for batch: {e}", exc_info=True)
                    # Continue to next batch or stop? Continue for now.
                    continue


        # Create DataFrame only if predictions were generated
        if all_individual_ids:
            self.predictions_df = pd.DataFrame({
                'IndividualID': all_individual_ids,
                'Predicted_Haplotype1': [haps[0] for haps in all_predicted_haplotypes],
                'Predicted_Haplotype2': [haps[1] for haps in all_predicted_haplotypes]
            })
            # Save predictions
            pred_file = os.path.join(self.output_dir, "predictions.csv")
            try:
                self.predictions_df.to_csv(pred_file, index=False)
                logging.info(f"Predictions saved to {pred_file}")
            except Exception as e:
                logging.error(f"Failed to save predictions to {pred_file}: {e}")
        else:
            logging.warning("No predictions were generated.")
            self.predictions_df = pd.DataFrame() # Create empty dataframe


    def _evaluate(self):
        """Evaluates predictions against ground truth."""
        if self.df_phased_truth is None:
            logging.warning("No ground truth phased data provided. Skipping evaluation.")
            return

        if not hasattr(self, 'predictions_df') or self.predictions_df.empty:
            logging.warning("No predictions available to evaluate.")
            return

        # Ensure we have the ground truth pairs and corresponding IDs from preprocessing
        if not hasattr(self, 'val_phased_truth_pairs') or not hasattr(self, 'val_ids_with_truth'):
             logging.warning("Preprocessed ground truth pairs/IDs not found. Cannot perform evaluation.")
             return

        logging.info("Evaluating predictions...")

        # Create a DataFrame from the preprocessed truth pairs and IDs
        truth_df = pd.DataFrame({
            'IndividualID': self.val_ids_with_truth,
            'True_Haplotype1': [pair[0] for pair in self.val_phased_truth_pairs],
            'True_Haplotype2': [pair[1] for pair in self.val_phased_truth_pairs]
        })

        # Merge predictions with the filtered ground truth
        eval_df = pd.merge(self.predictions_df, truth_df, on='IndividualID', how='inner')

        if eval_df.empty:
            logging.warning("Could not merge predictions with ground truth (check IndividualIDs or preprocessing steps). Skipping evaluation.")
            return

        logging.info(f"Evaluating on {len(eval_df)} samples with ground truth.")
        evaluator = HLAPhasingMetrics(tokenizer=self.tokenizer) # Pass tokenizer if needed by metrics
        try:
            # Prepare predicted and true pairs for the evaluator
            predicted_pairs = [tuple(sorted(pair)) for pair in zip(eval_df['Predicted_Haplotype1'], eval_df['Predicted_Haplotype2'])]
            true_pairs = [tuple(sorted(pair)) for pair in zip(eval_df['True_Haplotype1'], eval_df['True_Haplotype2'])]

            metrics = evaluator.calculate_metrics(predicted_haplotypes=predicted_pairs, true_haplotypes=true_pairs)
            logging.info(f"Evaluation Metrics: {metrics}")
            self.reporter.log_metric("evaluation_summary", metrics)

        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}", exc_info=True)


    def _finalize(self):
        """Generates final reports and plots."""
        logging.info("Finalizing run and generating reports...")
        try:
            # Generate reports
            report_formats = self.config.reporting.formats if hasattr(self.config, 'reporting') and hasattr(self.config.reporting, 'formats') else ["json", "txt"]
            base_filename = self.config.reporting.base_filename if hasattr(self.config, 'reporting') and hasattr(self.config.reporting, 'base_filename') else "final_report"

            for fmt in report_formats:
                report_filename = f"{base_filename}.{fmt}"
                self.reporter.generate_report(report_format=fmt, report_filename=report_filename)

            # Plot curves
            plot_filename = self.config.reporting.plot_filename if hasattr(self.config, 'reporting') and hasattr(self.config.reporting, 'plot_filename') else "training_loss_curves.png"
            self.reporter.plot_training_curves(plot_filename=plot_filename)

        except Exception as e:
            logging.error(f"Error during final reporting/plotting: {e}", exc_info=True)

        logging.info("Workflow finished.")


    def run(self):
        """Executes the full HLA phasing workflow."""
        try:
            self._set_seeds()
            df_unphased, df_phased_truth = self._load_data()
            self._preprocess_data(df_unphased, df_phased_truth)
            self._build_model()
            self._train_model()
            self._predict()
            self._evaluate()
        except Exception as e:
             logging.error(f"Workflow execution failed: {e}", exc_info=True)
             # Optionally add more specific error handling or cleanup
        finally:
            # Finalize should run even if errors occurred during the main workflow
            self._finalize()

# Example usage (if run directly, though typically instantiated and run)
# if __name__ == '__main__':
#     # Create a dummy config or load from file
#     # Example: Load from YAML
#     # config = HLAPhasingConfig.from_yaml('path/to/config.yaml')
#
#     # Or create programmatically (ensure all required fields are set)
#     config_dict = {
#         "seed": 42,
#         "device": "cpu",
#         "output_dir": "run_output",
#         "data": {
#             "unphased_data_path": "examples/data/synthetic_genotypes_unphased.csv",
#             "phased_data_path": "examples/data/synthetic_haplotypes_phased.csv",
#             "locus_columns": ["LocusA", "LocusB", "LocusC"],
#             "covariate_columns": ["Covariate1"],
#             "categorical_covariate_columns": ["Covariate1"],
#             "validation_split_ratio": 0.2
#         },
#         "model": {
#             "latent_dim": 32,
#             "encoder": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1},
#             "decoder": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1}
#             # Add other model params as needed by HLAPhasingModel structure
#         },
#         "training": {
#             "batch_size": 16,
#             "epochs": 5, # Small number for example
#             "learning_rate": 1e-4,
#             "kl_annealing_type": "linear",
#             "kl_annealing_epochs": 1,
#             "kl_annealing_max_weight": 1.0,
#             "reconstruction_weight": 1.0,
#             "checkpoint_dir": "checkpoints",
#             "checkpoint_frequency": 1,
#             "final_model_filename": "final_model.pt",
#             # "early_stopping_patience": 3 # Optional
#         },
#         "reporting": {
#             "formats": ["json", "txt"],
#             "base_filename": "run_report",
#             "plot_filename": "loss_curves.png"
#         }
#     }
#     config = HLAPhasingConfig(**config_dict)
#
#     runner = HLAPhasingRunner(config)
#     runner.run()
