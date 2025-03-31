import torch
import pandas as pd
import numpy as np
import os
import random
import logging # Import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# --- Add project root to sys.path ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

# --- Import TransPhaser Components ---
# Configuration
from src.config import HLAPhasingConfig, DataConfig, ModelConfig, TrainingConfig

# Data Preprocessing
from src.data_preprocessing import GenotypeDataParser, AlleleTokenizer, CovariateEncoder, HLADataset

# Model Components
from src.model import HLAPhasingModel

# Training
from src.trainer import HLAPhasingTrainer
from src.loss import ELBOLoss, KLAnnealingScheduler # Import loss function and scheduler
from torch.optim import Adam # Import optimizer

# Evaluation
from src.evaluation import HLAPhasingMetrics, PhasingUncertaintyEstimator # Import Uncertainty Estimator

# Samplers (for prediction)
# Note: GreedySampler was not found in src/samplers.py.
# Prediction logic below uses placeholders. Import specific samplers if available.
# from src.samplers import ...

# --- Configuration ---
DATA_DIR = "examples/data"
UNPHASED_DATA_FILE = os.path.join(DATA_DIR, "synthetic_genotypes_unphased.csv")
# UNPHASED_MISSING_DATA_FILE = os.path.join(DATA_DIR, "synthetic_genotypes_unphased_missing.csv") # Optional: Use this for testing missing data handling
PHASED_DATA_FILE = os.path.join(DATA_DIR, "synthetic_haplotypes_phased.csv") # Ground truth for evaluation

OUTPUT_DIR = "examples/output"
MODEL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "phaser_model_example.pt") # Changed from final_model.pt
PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "predictions.csv")
EVALUATION_FILE = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f"Using device: {DEVICE}")

# --- Helper Functions ---
# Note: setup_config is not used currently as config is built manually below
# def setup_config(loci, all_alleles, covariate_names, covariate_vocabs): ...

# --- Main Workflow ---
if __name__ == "__main__":
    logging.info("Starting TransPhaser Example Workflow...")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    logging.info("Loading data...")
    df_unphased = pd.read_csv(UNPHASED_DATA_FILE)
    df_phased_truth = pd.read_csv(PHASED_DATA_FILE) # For evaluation later

    # Infer loci and covariates from data columns
    loci = ['HLA-A', 'HLA-B', 'HLA-DRB1'] # Known from generation script
    covariate_cols = ['Population', 'AgeGroup'] # Known from generation script
    logging.info(f"Identified Loci: {loci}")
    logging.info(f"Identified Covariates: {covariate_cols}")

    # 2. Setup Configuration & Preprocessing Tools
    logging.info("Setting up configuration and preprocessing tools...")
    parser = GenotypeDataParser(locus_columns=loci, covariate_columns=covariate_cols)

    # Manually extract unique alleles
    all_alleles_set = set()
    for locus in loci:
        for genotype_str in df_unphased[locus].dropna():
            alleles = genotype_str.split('/')
            all_alleles_set.update(alleles)
    all_alleles = list(all_alleles_set)
    if "<UNK>" in all_alleles: all_alleles.remove("<UNK>")
    if "UNK" in all_alleles: all_alleles.remove("UNK")

    # Group alleles by locus for tokenizer
    alleles_by_locus = {}
    for locus in loci:
        locus_alleles = set()
        for genotype_str in df_unphased[locus].dropna():
            alleles = genotype_str.split('/')
            locus_alleles.update(alleles)
        if "<UNK>" in locus_alleles: locus_alleles.remove("<UNK>")
        if "UNK" in locus_alleles: locus_alleles.remove("UNK")
        alleles_by_locus[locus] = list(locus_alleles)

    # Initialize Tokenizer
    allele_tokenizer = AlleleTokenizer()
    logging.info("Building allele vocabularies...")
    for locus, locus_alleles_list in alleles_by_locus.items():
        allele_tokenizer.build_vocabulary(locus, locus_alleles_list)
        logging.info(f"  {locus}: {allele_tokenizer.get_vocab_size(locus)} tokens")

    # Initialize CovariateEncoder
    categorical_covariate_cols = covariate_cols
    numerical_covariate_cols = []
    covariate_encoder = CovariateEncoder(
        categorical_covariates=categorical_covariate_cols,
        numerical_covariates=numerical_covariate_cols
    )

    # --- Create Config Object ---
    vocab_sizes = {locus: allele_tokenizer.get_vocab_size(locus) for locus in loci}
    data_cfg = DataConfig(locus_columns=loci, covariate_columns=covariate_cols)
    model_cfg = ModelConfig(embedding_dim=64, latent_dim=32, num_layers=2, num_heads=4, ff_dim=128, dropout=0.1)
    # Reduce learning rate
    training_cfg = TrainingConfig(batch_size=32, learning_rate=1e-4, epochs=100) # Lower LR

    config = HLAPhasingConfig(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        device=str(DEVICE),
        seed=SEED
    )
    # TODO: Add vocab_sizes etc. fields to HLAPhasingConfig if needed by model/trainer directly

    logging.info(f"Allele Vocab Sizes determined: {vocab_sizes}")

    # 3. Preprocess Data & Create Datasets/DataLoaders
    logging.info("Preprocessing data and creating datasets...")
    train_df, val_df = train_test_split(df_unphased, test_size=0.2, random_state=SEED)

    # a) Parse Genotypes
    def parse_df_genotypes(df, loci_list):
        parsed = []
        for _, row in df.iterrows():
            sample_genotype = []
            for locus in loci_list:
                alleles = row[locus].split('/')
                sample_genotype.append(sorted(alleles))
            parsed.append(sample_genotype)
        return parsed

    train_genotypes_parsed = parse_df_genotypes(train_df, loci)
    val_genotypes_parsed = parse_df_genotypes(val_df, loci)

    # b) Encode Covariates
    logging.info("Fitting covariate encoder...")
    covariate_encoder.fit(train_df[categorical_covariate_cols + numerical_covariate_cols])
    logging.info("Transforming covariates...")
    train_covariates_encoded = covariate_encoder.transform(train_df[categorical_covariate_cols + numerical_covariate_cols]).to_numpy(dtype=np.float32)
    val_covariates_encoded = covariate_encoder.transform(val_df[categorical_covariate_cols + numerical_covariate_cols]).to_numpy(dtype=np.float32)
    logging.info(f"Encoded covariate shapes: Train={train_covariates_encoded.shape}, Val={val_covariates_encoded.shape}")

    # c) Extract Phased Haplotypes (align with train/val splits)
    df_phased_truth_indexed = df_phased_truth.set_index('IndividualID')
    train_phased_haplotypes = df_phased_truth_indexed.loc[train_df['IndividualID']]['Haplotype1'].tolist()
    val_phased_haplotypes = df_phased_truth_indexed.loc[val_df['IndividualID']]['Haplotype1'].tolist()

    # Create datasets
    train_dataset = HLADataset(
        genotypes=train_genotypes_parsed,
        covariates=train_covariates_encoded,
        phased_haplotypes=train_phased_haplotypes,
        tokenizer=allele_tokenizer,
        loci_order=loci
    )
    val_dataset = HLADataset(
        genotypes=val_genotypes_parsed,
        covariates=val_covariates_encoded,
        phased_haplotypes=val_phased_haplotypes,
        tokenizer=allele_tokenizer,
        loci_order=loci
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    logging.info(f"Created Train DataLoader ({len(train_dataset)} samples) and Validation DataLoader ({len(val_dataset)} samples)")

    # 4. Initialize Model
    logging.info("Initializing HLAPhasingModel...")
    num_loci_val = len(loci)
    allele_vocabs = allele_tokenizer.locus_vocabularies
    cov_dim = train_covariates_encoded.shape[1]

    encoder_cfg = {
        "vocab_sizes": vocab_sizes,
        "num_loci": num_loci_val,
        "embedding_dim": config.model.embedding_dim,
        "num_heads": config.model.num_heads,
        "num_layers": config.model.num_layers,
        "ff_dim": config.model.ff_dim,
        "dropout": config.model.dropout,
        "covariate_dim": cov_dim,
        "latent_dim": config.model.latent_dim,
        "loci_order": loci
    }
    decoder_cfg = {
        "vocab_sizes": vocab_sizes,
        "num_loci": num_loci_val,
        "embedding_dim": config.model.embedding_dim,
        "num_heads": config.model.num_heads,
        "num_layers": config.model.num_layers,
        "ff_dim": config.model.ff_dim,
        "dropout": config.model.dropout,
        "covariate_dim": cov_dim,
        "latent_dim": config.model.latent_dim,
        "loci_order": loci
    }

    model = HLAPhasingModel(
        num_loci=num_loci_val,
        allele_vocabularies=allele_vocabs,
        covariate_dim=cov_dim,
        tokenizer=allele_tokenizer, # Pass tokenizer instance
        encoder_config=encoder_cfg,
        decoder_config=decoder_cfg
    ).to(DEVICE)
    logging.info(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 5. Initialize Loss, Optimizer, and Trainer
    logging.info("Initializing Loss, Optimizer, and Schedulers...")
    # Initialize loss with initial KL weight 0 for annealing
    loss_fn = ELBOLoss(kl_weight=0.0, reconstruction_weight=1.0)
    optimizer = Adam(model.parameters(), lr=config.training.learning_rate) # Uses lower LR now

    # Initialize KL Annealing Scheduler
    # Ramp up linearly over the first epoch (approx steps = num_samples / batch_size)
    steps_per_epoch = len(train_loader)
    kl_scheduler = KLAnnealingScheduler(
        anneal_type='linear',
        max_weight=1.0, # Target KL weight
        total_steps=steps_per_epoch # Ramp up over 1 epoch
    )

    logging.info("Initializing HLAPhasingTrainer...")
    trainer = HLAPhasingTrainer(
        model=model,
        loss_fn=loss_fn,
        kl_scheduler=kl_scheduler, # Pass the KL scheduler
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
        epochs=config.training.epochs
    )

    logging.info("Starting training...")
    train_losses = []
    val_losses = []
    try:
        # Run the actual training loop implemented in the trainer
        train_losses, val_losses = trainer.train() # Capture loss histories
        logging.info("Training finished.")
        # Note: trainer.train() now saves final_model.pt

    except NotImplementedError:
         logging.warning("Trainer.train() or sub-methods (train_epoch/evaluate) not fully implemented. Skipping training loop.")
         torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
         logging.info(f"Saved initial model state to {MODEL_CHECKPOINT_PATH} for prediction demo.")
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        # Save initial model state if training failed
        torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
        logging.info(f"Saved initial model state to {MODEL_CHECKPOINT_PATH} due to training error.")

    # --- Plot Loss Curves ---
    if train_losses and val_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        # Filter out potential NaN values from validation loss for plotting
        val_epochs = [i + 1 for i, loss in enumerate(val_losses) if not np.isnan(loss)]
        valid_val_losses = [loss for loss in val_losses if not np.isnan(loss)]
        if valid_val_losses: # Only plot if there are valid validation losses
             plt.plot(val_epochs, valid_val_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (ELBO)')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(OUTPUT_DIR, "loss_curves.png")
        try:
            plt.savefig(loss_plot_path)
            logging.info(f"Loss curves saved to {loss_plot_path}")
        except Exception as e:
            logging.error(f"Error saving loss plot: {e}")
        plt.close() # Close the plot figure
    else:
        logging.warning("Loss histories not available, skipping plotting.")
    # --- End Plotting ---


    # 6. Predict Haplotypes (using the validation set for demonstration)
    logging.info("Predicting haplotypes for the validation set...")
    model.eval()

    # --- Debug Print Tokenizer ---
    # print("\n--- DEBUG Tokenizer Reverse Vocab (HLA-A) ---")
    # print(allele_tokenizer.locus_reverse_vocabularies.get('HLA-A', 'HLA-A not found'))
    # print("---------------------------------------------\n")
    # --- End Debug Print ---


    # Define predict_loader outside the try block
    predict_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # Use the implemented prediction method
    try:
        all_predicted_haplotypes = []
        all_individual_ids = []
        with torch.no_grad():
            # Enumerate the loader to get batch_idx
            for batch_idx, batch in enumerate(predict_loader):
                # Prepare batch for prediction (only need genotype and covariates)
                pred_batch = {
                    'genotype_tokens': batch['genotype_tokens'].to(DEVICE),
                    'covariates': batch['covariates'].to(DEVICE)
                }
                sample_indices = batch['sample_index']

                # Use model's predict method
                # predicted_tokens_h1 shape: (batch_size, num_loci)
                predicted_tokens_h1 = model.predict_haplotypes(pred_batch)

                # --- Derive Haplotype 2 from Predicted H1 and Input Genotype ---
                # The model predicts one haplotype (h1). We derive the second (h2)
                # by finding the allele in the input genotype that complements h1 at each locus.
                genotype_tokens_batch = batch['genotype_tokens'].to(DEVICE) # Shape (batch, num_loci * 2)
                predicted_tokens_h2 = torch.zeros_like(predicted_tokens_h1) # Initialize h2 tensor

                for i in range(predicted_tokens_h1.size(0)): # Iterate through samples in batch
                    for j in range(num_loci_val): # Iterate through loci
                        # Get the two input genotype tokens for this locus
                        # Ensure comparison happens on the correct device
                        locus_genotype_token1 = genotype_tokens_batch[i, j * 2]
                        locus_genotype_token2 = genotype_tokens_batch[i, j * 2 + 1]
                        pred_h1_token = predicted_tokens_h1[i, j]

                        # Find the complementary token from the input genotype
                        # Handle potential homozygosity (both input tokens are the same)
                        if locus_genotype_token1 == pred_h1_token:
                            # If token1 matches h1, h2 must be token2
                            predicted_tokens_h2[i, j] = locus_genotype_token2
                        elif locus_genotype_token2 == pred_h1_token:
                             # If token2 matches h1, h2 must be token1
                             predicted_tokens_h2[i, j] = locus_genotype_token1
                        else:
                             # This case should ideally not happen if the model predicts an allele
                             # present in the input genotype. If it does, it indicates a potential
                             # issue with the model's prediction or the input data consistency.
                             # As a fallback, arbitrarily pick one of the genotype tokens.
                             # Log a warning if this happens frequently.
                             logging.warning(f"Sample {i}, Locus {loci[j]}: Predicted H1 token {pred_h1_token.item()} not found in genotype tokens ({locus_genotype_token1.item()}, {locus_genotype_token2.item()}). Using fallback for H2.")
                             predicted_tokens_h2[i, j] = locus_genotype_token1 # Fallback
                # --- End Haplotype 2 Derivation ---

                # Convert predicted tokens to allele strings
                batch_size = predicted_tokens_h1.shape[0]
                predicted_haplotypes_batch = []
                for i in range(batch_size):
                    hap1_alleles = []
                    hap2_alleles = []
                    for j, locus_name in enumerate(loci): # Use loci order
                        token_idx1 = predicted_tokens_h1[i, j].item()
                        token_idx2 = predicted_tokens_h2[i, j].item() # Using derived H2 tokens
                        allele1 = allele_tokenizer.detokenize(locus_name, token_idx1)
                        allele2 = allele_tokenizer.detokenize(locus_name, token_idx2)
                        hap1_alleles.append(allele1)
                        hap2_alleles.append(allele2)
                    hap1_str = "_".join(hap1_alleles)
                    # --- Debug Print H2 String ---
                    # if i == 0 and batch_idx == 0:
                    #     logging.debug(f"  Sample 0 Haplotype Alleles: H1={hap1_alleles}, H2={hap2_alleles}")
                    hap2_str = "_".join(hap2_alleles)
                    # if i == 0 and batch_idx == 0:
                    #     logging.debug(f"  Sample 0 Haplotype Strings: H1='{hap1_str}', H2='{hap2_str}'")
                    # --- End Debug Print ---
                    # Ensure consistent pair order for evaluation (e.g., lexicographically)
                    predicted_haplotypes_batch.append(tuple(sorted((hap1_str, hap2_str))))

                all_predicted_haplotypes.extend(predicted_haplotypes_batch)
                batch_individual_ids = val_df.iloc[sample_indices.cpu().numpy()]['IndividualID'].tolist()
                all_individual_ids.extend(batch_individual_ids)

        predictions_df = pd.DataFrame({
            'IndividualID': all_individual_ids,
            'Predicted_Haplotype1': [haps[0] for haps in all_predicted_haplotypes],
            'Predicted_Haplotype2': [haps[1] for haps in all_predicted_haplotypes]
        })
        predictions_df.to_csv(PREDICTIONS_FILE, index=False)
        logging.info(f"Predictions saved to {PREDICTIONS_FILE}")

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True) # Add exc_info
        predictions_df = pd.DataFrame()

    # 6.5 Estimate Uncertainty (using the validation set)
    logging.info("Estimating prediction uncertainty for the validation set...")
    all_uncertainties = []
    try:
        uncertainty_estimator = PhasingUncertaintyEstimator(model=model)
        with torch.no_grad():
            for batch in predict_loader: # Use the same loader as prediction
                # Prepare batch for uncertainty estimation (needs genotype, covariates)
                uncertainty_batch = {
                    'genotype_tokens': batch['genotype_tokens'].to(DEVICE),
                    'covariates': batch['covariates'].to(DEVICE)
                }
                uncertainty_results = uncertainty_estimator.estimate_uncertainty(uncertainty_batch)
                # Assuming 'mean_prediction_entropy' is returned
                if 'mean_prediction_entropy' in uncertainty_results:
                    batch_uncertainty = uncertainty_results['mean_prediction_entropy'].cpu().numpy()
                    all_uncertainties.extend(batch_uncertainty)
                else:
                    logging.warning("Uncertainty estimator did not return 'mean_prediction_entropy'.")
                    # Add NaNs to maintain length if needed, or break
                    all_uncertainties.extend([np.nan] * uncertainty_batch['genotype_tokens'].size(0))


        if all_uncertainties:
            # Filter out NaNs before calculating mean
            valid_uncertainties = [u for u in all_uncertainties if not np.isnan(u)]
            if valid_uncertainties:
                 mean_uncertainty = np.mean(valid_uncertainties)
                 logging.info(f"Mean Prediction Entropy (Uncertainty) on Validation Set: {mean_uncertainty:.4f}")
                 # Optionally, save uncertainties to the predictions_df
                 if len(all_uncertainties) == len(predictions_df):
                     predictions_df['MeanPredictionEntropy'] = all_uncertainties
                     predictions_df.to_csv(PREDICTIONS_FILE, index=False) # Re-save with uncertainty
                     logging.info(f"Updated predictions with uncertainty saved to {PREDICTIONS_FILE}")
                 else:
                     logging.warning("Length mismatch between uncertainties and predictions. Not adding to CSV.")
            else:
                 logging.warning("No valid uncertainty values calculated.")
        else:
            logging.warning("Uncertainty estimation did not produce results.")

    except Exception as e:
        logging.error(f"An error occurred during uncertainty estimation: {e}", exc_info=True)


    # 7. Evaluate Predictions
    logging.info("Evaluating predictions...")
    if not predictions_df.empty:
        val_ids = val_df['IndividualID'].tolist()
        df_phased_truth_val = df_phased_truth[df_phased_truth['IndividualID'].isin(val_ids)]
        eval_df = pd.merge(predictions_df, df_phased_truth_val, on='IndividualID', how='inner')

        if not eval_df.empty:
            evaluator = HLAPhasingMetrics(tokenizer=allele_tokenizer)
            try:
                # Prepare predicted and true pairs for evaluation (ensure consistent ordering)
                predicted_pairs = [tuple(sorted(pair)) for pair in zip(eval_df['Predicted_Haplotype1'], eval_df['Predicted_Haplotype2'])]
                true_pairs = [tuple(sorted(pair)) for pair in zip(eval_df['Haplotype1'], eval_df['Haplotype2'])]

                # The calculate_metrics method itself is now implemented for accuracy
                metrics = evaluator.calculate_metrics(predicted_haplotypes=predicted_pairs, true_haplotypes=true_pairs)

                logging.info("Evaluation Metrics:")
                metrics_str = "\n".join([f"- {key}: {value:.4f}" for key, value in metrics.items()])
                logging.info(f"\n{metrics_str}") # Log metrics to console
                with open(EVALUATION_FILE, 'w') as f:
                    f.write("Evaluation Metrics:\n")
                    f.write(metrics_str) # Write the same string to file
                logging.info(f"Evaluation metrics saved to {EVALUATION_FILE}")

            except NotImplementedError as e:
                 logging.warning(f"Evaluation logic not fully implemented: {e}. Skipping evaluation.")
            except Exception as e:
                logging.error(f"An error occurred during evaluation: {e}")
        else:
            logging.warning("Could not merge predictions with ground truth. Skipping evaluation.")
    else:
        logging.warning("No predictions generated. Skipping evaluation.")

    logging.info("TransPhaser Example Workflow Finished.")
