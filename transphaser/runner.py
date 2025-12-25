"""
TransPhaser Runner: High-level API for training, evaluation, and inference.

This module provides the TransPhaserRunner class, which serves as the main entry point
for interacting with the TransPhaser model. It handles data preparation, model training,
evaluation, persistence, and analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple, Any, Union
import os
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

from transphaser.model import TransPhaser, TransPhaserLoss
from transphaser.data_preprocessing import AlleleTokenizer, CovariateEncoder, HLADataset, GenotypeDataParser
from transphaser.evaluation import HLAPhasingMetrics
from transphaser.config import TransPhaserConfig, HLAPhasingConfig


class TransPhaserRunner:
    """
    Runner for TransPhaser model training, evaluation, and analysis.
    Serves as the high-level API for the library.
    """
    
    def __init__(self, config: TransPhaserConfig):
        """
        Initialize the TransPhaser runner.
        
        Args:
            config: TransPhaserConfig instance
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        self._set_seeds(config.seed)
        
        # Placeholders
        self.model = None
        self.tokenizer = None
        self.covariate_encoder = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        
        logging.info(f"TransPhaserRunner initialized on device: {self.device}")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _prepare_data(self, df_unphased: pd.DataFrame, df_phased: Optional[pd.DataFrame] = None):
        """
        Prepare data loaders from DataFrames.
        
        Args:
            df_unphased: DataFrame with unphased genotypes
            df_phased: DataFrame with phased haplotypes (for evaluation/supervised training)
        """
        locus_columns = self.config.data.locus_columns
        covariate_columns = self.config.data.covariate_columns
        categorical_covariates = self.config.data.categorical_covariate_columns
        
        # Build tokenizer
        self.tokenizer = AlleleTokenizer()
        self.tokenizer.build_vocabulary_from_dataframe(df_unphased, locus_columns)
        
        # Build covariate encoder
        self.covariate_encoder = CovariateEncoder(
            categorical_covariates=categorical_covariates,
            numerical_covariates=[c for c in covariate_columns if c not in categorical_covariates]
        )
        self.covariate_encoder.fit(df_unphased)
        
        # Split data
        split_idx = int(len(df_unphased) * (1 - self.config.data.validation_split_ratio))
        df_train = df_unphased.iloc[:split_idx].reset_index(drop=True)
        df_val = df_unphased.iloc[split_idx:].reset_index(drop=True)
        df_phased_train = df_phased.iloc[:split_idx].reset_index(drop=True) if df_phased is not None else None
        df_phased_val = df_phased.iloc[split_idx:].reset_index(drop=True) if df_phased is not None else None
        
        # Parse genotypes
        parser = GenotypeDataParser(locus_columns, covariate_columns)
        train_genotypes, _ = parser.parse(df_train)
        val_genotypes, _ = parser.parse(df_val)
        
        # Encode covariates
        train_covariates = self.covariate_encoder.transform(df_train).values
        val_covariates = self.covariate_encoder.transform(df_val).values
        
        # Parse phased haplotypes for supervised training and evaluation
        train_haplotypes = None
        val_haplotypes = None
        
        if df_phased_train is not None:
            train_haplotypes = []
            for _, row in df_phased_train.iterrows():
                h1 = row['Haplotype1']
                h2 = row['Haplotype2']
                train_haplotypes.append((h1, h2))
        
        if df_phased_val is not None:
            val_haplotypes = []
            for _, row in df_phased_val.iterrows():
                h1 = row['Haplotype1']
                h2 = row['Haplotype2']
                val_haplotypes.append((h1, h2))
        
        # Create datasets - use 'eval' mode for both to include ground truth haplotypes
        # This enables supervised learning from phased data
        train_dataset = HLADataset(
            genotypes=train_genotypes,
            covariates=train_covariates,
            tokenizer=self.tokenizer,
            loci_order=locus_columns,
            phased_haplotypes=train_haplotypes,  # Include for supervised loss
            mode='eval' if train_haplotypes is not None else 'train',
        )
        
        val_dataset = HLADataset(
            genotypes=val_genotypes,
            covariates=val_covariates,
            tokenizer=self.tokenizer,
            loci_order=locus_columns,
            phased_haplotypes=val_haplotypes,  # For evaluation
            mode='eval',
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
        )
        
        # Store validation ground truth for evaluation
        self.val_phased_df = df_phased_val
        
        logging.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    def _build_model(self):
        """Build TransPhaser model."""
        
        # Compute covariate dimension from training data
        covariate_dim = 0
        if self.train_loader is not None and len(self.train_loader) > 0:
            # Get first batch to determine covariate dimension
            for batch in self.train_loader:
                if 'covariates' in batch:
                    covariate_dim = batch['covariates'].shape[-1]
                break
        
        # Build config for TransPhaser
        model_config = {
            "num_loci": len(self.config.data.locus_columns),
            "loci_order": self.config.data.locus_columns,
            "vocab_sizes": {
                locus: self.tokenizer.get_vocab_size(locus) 
                for locus in self.tokenizer.locus_vocabularies.keys()
            },
            "embedding_dim": self.config.model.embedding_dim,
            "latent_dim": self.config.model.latent_dim,
            "num_heads": getattr(self.config.model.encoder, "num_heads", 4),
            "num_layers": getattr(self.config.model.encoder, "num_layers", 2),
            "ff_dim": getattr(self.config.model.encoder, "ff_dim", 128),
            "dropout": getattr(self.config.model.encoder, "dropout", 0.2), # Increased to 0.2 to fight overfitting
            "covariate_dim": covariate_dim,
            "top_k": 64,  # Use more candidates for better coverage
            "padding_idx": self.tokenizer.pad_token_id,
            "tokenizer": self.tokenizer,
        }
        
        self.model = TransPhaser(model_config)
        self.model.to(self.device)
        
        # Loss function with supervised learning
        self.loss_fn = TransPhaserLoss(
            proposal_weight=0.1,   # Lower distillation (supervised provides stronger signal)
            entropy_weight=0.001,  # Very low entropy for confident predictions
            supervised_weight=1.0, # Strong supervised signal
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.005,  # Higher LR for faster convergence
            weight_decay=1e-4,
        )
        
        # Simple cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.epochs,
            eta_min=1e-5,
        )
        
        logging.info(f"TransPhaser model built with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.loss_fn(output, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set. Returns metrics dictionary."""
        self.model.eval()
        
        all_predictions = []
        all_ground_truth = []
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass
            output = self.model(batch)
            loss = self.loss_fn(output, batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            h1_tokens = output["h1_tokens"].cpu()  # (batch, num_loci)
            h2_tokens = output["h2_tokens"].cpu()  # (batch, num_loci)
            
            # Convert to haplotype strings for evaluation
            batch_size = h1_tokens.shape[0]
            for i in range(batch_size):
                h1_alleles = []
                h2_alleles = []
                for j, locus in enumerate(self.config.data.locus_columns):
                    h1_token = h1_tokens[i, j].item()
                    h2_token = h2_tokens[i, j].item()
                    h1_alleles.append(self.tokenizer.detokenize(locus, h1_token))
                    h2_alleles.append(self.tokenizer.detokenize(locus, h2_token))
                
                h1_str = "_".join(h1_alleles)
                h2_str = "_".join(h2_alleles)
                all_predictions.append(tuple(sorted((h1_str, h2_str))))
            
            # Get ground truth if available
            if "target_h1_tokens" in batch and "target_h2_tokens" in batch:
                gt_h1 = batch["target_h1_tokens"].cpu()  # (batch, num_loci + 2) includes BOS, EOS
                gt_h2 = batch["target_h2_tokens"].cpu()
                
                for i in range(batch_size):
                    h1_alleles = []
                    h2_alleles = []
                    for j, locus in enumerate(self.config.data.locus_columns):
                        # Skip BOS at index 0
                        h1_token = gt_h1[i, j + 1].item()
                        h2_token = gt_h2[i, j + 1].item()
                        h1_alleles.append(self.tokenizer.detokenize(locus, h1_token))
                        h2_alleles.append(self.tokenizer.detokenize(locus, h2_token))
                    
                    h1_str = "_".join(h1_alleles)
                    h2_str = "_".join(h2_alleles)
                    all_ground_truth.append(tuple(sorted((h1_str, h2_str))))
        
        # Compute metrics
        metrics = {"val_loss": total_loss / max(num_batches, 1)}
        
        if len(all_ground_truth) == len(all_predictions) and len(all_ground_truth) > 0:
            metrics_calc = HLAPhasingMetrics(tokenizer=self.tokenizer)
            eval_metrics = metrics_calc.calculate_metrics(all_predictions, all_ground_truth)
            metrics.update(eval_metrics)
        else:
            # If no ground truth available, we can't calculate accuracy
            metrics["phasing_accuracy"] = 0.0
        
        return metrics
    
    def run(self, df_unphased: pd.DataFrame, df_phased: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Args:
            df_unphased: DataFrame with unphased genotypes
            df_phased: DataFrame with phased haplotypes (optional, for validation)
        
        Returns:
            Dictionary with training stats and metrics.
        """
        # Prepare data
        self._prepare_data(df_unphased, df_phased)
        
        # Build model if not already built (e.g. from load)
        if self.model is None:
            self._build_model()
        
        # Training loop
        best_val_accuracy = 0.0
        epochs_without_improvement = 0
        best_model_state = None
        
        logging.info(f"Starting training for {self.config.training.epochs} epochs.")
        
        for epoch in range(self.config.training.epochs):
            # Train
            train_loss = self._train_epoch()
            
            # Validate
            val_metrics = self._evaluate()
            val_loss = val_metrics.get("val_loss", 0.0)
            val_accuracy = val_metrics.get("phasing_accuracy", 0.0)
            
            # Update learning rate
            self.scheduler.step()
            
            # Track history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_accuracy)
            
            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(
                    f"Epoch {epoch + 1}/{self.config.training.epochs}: "
                    f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                    f"Val Accuracy={val_accuracy:.2%}"
                )
            
            # Early stopping check (based on accuracy if available, else loss)
            improvement = False
            if "phasing_accuracy" in val_metrics and val_metrics["phasing_accuracy"] > 0:
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    improvement = True
            else:
                # Use loss (minimize)
                # Just placeholder logic as accuracy is main metric for now
                pass
            
            if improvement:
                epochs_without_improvement = 0
                best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            else:
                epochs_without_improvement += 1
            
            # Always save best state if accuracy improved, or just use last if no improve?
            # If no ground truth, we can only rely on loss, but unsupervised loss can be misleading.
            
            if self.config.training.early_stopping_patience is not None:
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model if found
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logging.info(f"Restored best model with Validation Accuracy: {best_val_accuracy:.2%}")
        
        # Final evaluation
        final_metrics = self._evaluate()
        logging.info(f"Final TransPhaser metrics: {final_metrics}")
        
        return {
            "training_history": self.training_history,
            "final_metrics": final_metrics,
            "evaluation_summary": final_metrics,
        }
    
    # --- Persistence ---
    
    def save(self, path: str):
        """
        Save the entire runner state (model, optimizer, history) to a file.
        
        Args:
            path: File path (e.g., 'checkpoints/model.pt')
        """
        if self.model is None:
            logging.warning("Model not built. Nothing to save.")
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            # We also need to save tokenizer vocab and encoder mapping to be fully portable
            "tokenizer_vocab": self.tokenizer.locus_vocabularies,
        }, path)
        logging.info(f"TransPhaser model saved to {path}")
    
    def load(self, path: str):
        """
        Load runner state from a file.
        
        Args:
            path: File path
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Ensure tokenizer is built/loaded (critical for model structure)
        if "tokenizer_vocab" in checkpoint:
            self.tokenizer = AlleleTokenizer()
            self.tokenizer.locus_vocabularies = checkpoint["tokenizer_vocab"]
            # Build inverse mapping
            self.tokenizer.id_to_token = {
                locus: {idx: token for token, idx in vocab.items()}
                for locus, vocab in self.tokenizer.locus_vocabularies.items()
            }
        elif self.tokenizer is None:
            raise ValueError("Tokenizer not initialized and not found in checkpoint. Cannot load.")
            
        # Build model if needed (requires tokenizer)
        if self.model is None:
            self._build_model()
            
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        if "training_history" in checkpoint:
            self.training_history = checkpoint["training_history"]
            
        logging.info(f"TransPhaser model loaded from {path}")

    # Aliases for compatibility
    save_model = save
    load_model = load

    # --- Helper Methods ---
    
    def extract_loss_history(self) -> Dict[str, List[float]]:
        """Return the training history dictionary."""
        return self.training_history
    
    @torch.no_grad()
    def get_most_likely_haplotypes(self, df_samples: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the most likely haplotype pair for each sample in the dataframe.
        
        Args:
            df_samples: DataFrame with genotype columns.
            
        Returns:
            DataFrame with 'Haplotype1', 'Haplotype2', and 'Probability' columns.
        """
        self.model.eval()
        
        # Prepare input
        locus_columns = self.config.data.locus_columns
        covariate_columns = self.config.data.covariate_columns
        
        parser = GenotypeDataParser(locus_columns, covariate_columns)
        genotypes, _ = parser.parse(df_samples)
        
        # Covariates - need to handle robustly if encoder not fit or check compatibility
        # For simplicity, if encoder exists, try transform. Else zeros.
        if self.covariate_encoder:
            try:
                covariates = self.covariate_encoder.transform(df_samples).values
            except Exception:
                # Fallback or zeros
                covariates = np.zeros((len(df_samples), 0))
        
        # Create dataset
        dataset = HLADataset(genotypes, covariates, self.tokenizer, locus_columns, mode='predict')
        loader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=False)
        
        results = []
        
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            output = self.model(batch)
            
            # Responsibilities are probabilities over K candidates
            # We want the prob of the BEST candidate
            probs = output["responsibilities"] # (batch, k)
            max_probs, best_idx = torch.max(probs, dim=-1) # (batch,)
            
            h1_tokens = output["h1_tokens"].cpu() # (batch, num_loci)
            h2_tokens = output["h2_tokens"].cpu()
            max_probs = max_probs.cpu()
            
            batch_size = h1_tokens.shape[0]
            for i in range(batch_size):
                h1_str = self._detokenize_haplotype(h1_tokens[i])
                h2_str = self._detokenize_haplotype(h2_tokens[i])
                prob = max_probs[i].item()
                
                results.append({
                    "Haplotype1": h1_str,
                    "Haplotype2": h2_str,
                    "Probability": prob
                })
        
        return pd.DataFrame(results)

    def _detokenize_haplotype(self, tokens: torch.Tensor) -> str:
        """Helper to convert tensor tokens to string."""
        alleles = []
        for j, locus in enumerate(self.config.data.locus_columns):
            token = tokens[j].item()
            alleles.append(self.tokenizer.detokenize(locus, token))
        return "_".join(alleles)

    @torch.no_grad()
    def get_haplotype_frequencies(self, df_samples: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate population haplotype frequencies based on predictions.
        
        Returns:
            DataFrame with 'Haplotype', 'Frequency', 'Count'.
        """
        df_pred = self.get_most_likely_haplotypes(df_samples)
        
        # Count all haplotypes (Flatten)
        all_haps = list(df_pred['Haplotype1']) + list(df_pred['Haplotype2'])
        
        counts = pd.Series(all_haps).value_counts()
        total = len(all_haps)
        
        freqs = pd.DataFrame({
            "Haplotype": counts.index,
            "Count": counts.values,
            "Frequency": counts.values / total
        })
        return freqs
    
    def get_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Extract allele embeddings for each locus.
        
        Returns:
            Dictionary: {locus_name: embedding_matrix (vocab_size, dim)}
        """
        if self.model is None:
            return {}
            
        embeddings = {}
        for locus, embed_layer in self.model.prior.haplotype_embedding.allele_embeddings.items():
            embeddings[locus] = embed_layer.weight.detach().cpu().numpy()
            
        return embeddings
    
    def plot_embeddings(self, locus: str, method='pca', output_path: str = None):
        """
        Plot allele embeddings for a given locus.
        
        Args:
            locus: Locus name (e.g. 'HLA-A')
            method: 'pca' or 'tsne'
            output_path: If provided, save plot to this path.
        """
        embeds_dict = self.get_embeddings()
        if locus not in embeds_dict:
            logging.warning(f"Locus {locus} not found in model.")
            return
        
        matrix = embeds_dict[locus]
        # remove padding if desired, or keep (idx 0)
        
        if method == 'tsne' and TSNE_AVAILABLE:
            reducer = TSNE(n_components=2, perplexity=min(30, len(matrix)-1), random_state=42)
        else:
            reducer = PCA(n_components=2)
            
        reduced = reducer.fit_transform(matrix)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        
        # Annotate with allele names
        vocab = self.tokenizer.locus_vocabularies.get(locus, {})
        # Inverse vocab
        id_to_token = {v: k for k, v in vocab.items()}
        
        for i in range(len(matrix)):
            label = id_to_token.get(i, str(i))
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.8)
            
        plt.title(f"{locus} Allele Embeddings ({method.upper()})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(output_path)
            logging.info(f"Embedding plot saved to {output_path}")
        else:
            plt.show()
        plt.close()
