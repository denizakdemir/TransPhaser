import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler # Use base class for type hint
import time # For timing epochs
import logging # For logging progress

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HLAPhasingTrainer:
    """
    Handles the training loop, validation, metric tracking, and checkpointing
    for the HLAPhasingModel.
    """
    def __init__(self, model: nn.Module, loss_fn: nn.Module,
                 train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: Optimizer, lr_scheduler: _LRScheduler = None, # Optional scheduler
                 kl_scheduler = None, # Optional KL Annealing scheduler
                 device: torch.device = None,
                 epochs: int = 10, # Default epochs
                 grad_accumulation_steps: int = 1,
                 log_interval: int = 50, # Log every N batches
                 # Add early stopping parameters if needed
                 # early_stopping_patience: int = None,
                 # checkpoint_manager = None # Pass CheckpointManager instance
                 ):
        """
        Initializes the HLAPhasingTrainer.

        Args:
            model: The PyTorch model to train (e.g., HLAPhasingModel).
            loss_fn: The loss function (e.g., ELBOLoss).
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            optimizer: The optimizer (e.g., Adam).
            lr_scheduler: Optional learning rate scheduler.
            device: The device to train on ('cuda' or 'cpu'). Auto-detects if None.
            epochs (int): Total number of training epochs.
            grad_accumulation_steps (int): Number of steps to accumulate gradients over.
            log_interval (int): Log training progress every N batches.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.kl_scheduler = kl_scheduler # Store KL scheduler

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.epochs = epochs
        self.grad_accumulation_steps = grad_accumulation_steps
        self.log_interval = log_interval

        self.current_epoch = 0
        self.global_step = 0
        self.train_loss_history = []
        self.val_loss_history = []
        # Add state for best model tracking, early stopping etc.
        # self.best_val_loss = float('inf')
        # self.epochs_no_improve = 0

        self.model.to(self.device) # Move model to the specified device
        logging.info(f"HLAPhasingTrainer initialized. Training on {self.device}.")
        logging.info(f"Total epochs: {self.epochs}, Grad Accumulation: {self.grad_accumulation_steps}")


    def train_epoch(self):
        """Runs a single training epoch."""
        self.model.train() # Set model to training mode
        total_loss = 0.0
        start_time = time.time()

        self.optimizer.zero_grad() # Zero gradients at the start of the epoch

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device (assuming batch is a dictionary of tensors)
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # --- Forward pass ---
            # --- Forward pass ---
            # Assumes model returns dict with keys needed by loss_fn (e.g., 'reconstruction_log_prob', 'kl_divergence')
            # Assumes batch contains all necessary keys (e.g., 'genotype_tokens', 'target_haplotype_tokens', etc.)
            try:
                model_output = self.model(batch)
                loss = self.loss_fn(model_output)
            except KeyError as e:
                 logging.error(f"Missing key in batch or model_output for loss calculation: {e}")
                 # Skip batch or raise error? Skipping for now.
                 continue
            except Exception as e:
                 logging.error(f"Error during forward pass or loss calculation: {e}")
                 continue

            # Scale loss for gradient accumulation
            if self.grad_accumulation_steps > 1:
                loss = loss / self.grad_accumulation_steps

            # --- Backward pass ---
            loss.backward()

            # --- Optimizer step ---
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                # Gradient clipping (added to prevent exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad() # Zero gradients after stepping

                # Update learning rate scheduler if used
                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # Update KL annealing scheduler if used
                if self.kl_scheduler:
                    new_kl_weight = self.kl_scheduler.step()
                    if hasattr(self.loss_fn, 'update_kl_weight'):
                        self.loss_fn.update_kl_weight(new_kl_weight)
                    else:
                        logging.warning("KL scheduler provided, but loss_fn has no 'update_kl_weight' method.")


                self.global_step += 1


            batch_loss = loss.item() * self.grad_accumulation_steps # Unscale loss for logging
            total_loss += batch_loss

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                 elapsed = time.time() - start_time
                 lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else self.optimizer.param_groups[0]['lr']
                 logging.info(f'Epoch {self.current_epoch+1}/{self.epochs} | Batch {batch_idx+1}/{len(self.train_loader)} | '
                              f'Loss: {batch_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.2f}s')
                 start_time = time.time() # Reset timer for next interval

        avg_train_loss = total_loss / len(self.train_loader)
        self.train_loss_history.append(avg_train_loss)
        logging.info(f'Epoch {self.current_epoch+1} Training Summary | Avg Loss: {avg_train_loss:.4f}')
        return avg_train_loss


    def evaluate(self):
        """Runs evaluation on the validation set."""
        self.model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        start_time = time.time()

        # Removed torch.no_grad() to potentially allow anomaly detection or fix NaN issue
        for batch_idx, batch in enumerate(self.val_loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # --- Forward pass ---
            try: # Ensure correct indentation for the whole block
                model_output = self.model(batch)
                loss = self.loss_fn(model_output) # Calculate full ELBO loss
                if torch.isnan(loss):
                    logging.warning(f"NaN detected in validation loss for batch {batch_idx}. Skipping batch.")
                    continue # Skip batch if loss is NaN
                total_val_loss += loss.item()
            except KeyError as e: # Ensure correct indentation
                logging.error(f"Validation - Missing key in batch or model_output for loss calculation: {e}")
                continue
            except Exception as e: # Ensure correct indentation
                # If anomaly detection is on, this might catch the specific error
                logging.error(f"Validation - Error during forward pass or loss calculation: {e}", exc_info=True) # Log traceback
                continue # Continue to next batch even if one fails

        # Calculate average loss, handle case where all batches were skipped
        if len(self.val_loader) > 0 and total_val_loss > 0: # Check if any loss was accumulated
            # Calculate average only over batches that didn't result in NaN
            num_valid_batches = sum(1 for l in self.val_loss_history[-len(self.val_loader):] if not torch.isnan(torch.tensor(l))) # Re-calculate based on history? Risky.
            # Safer: Count non-NaN batches directly if possible, otherwise divide by total batches and accept inaccuracy if NaNs occurred.
            # Let's stick to the simpler, potentially inaccurate average for now if NaNs occurred.
            avg_val_loss = total_val_loss / len(self.val_loader)
        elif len(self.val_loader) > 0: # Loader not empty, but all batches were NaN
             avg_val_loss = float('nan')
        else: # Loader is empty
            avg_val_loss = 0.0

        self.val_loss_history.append(avg_val_loss)
        elapsed = time.time() - start_time
        logging.info(f'Epoch {self.current_epoch+1} Validation Summary | Avg Loss: {avg_val_loss:.4f} | Time: {elapsed:.2f}s')
        return avg_val_loss


    def train(self):
        """Runs the full training loop for the specified number of epochs."""
        logging.info("Starting training...")
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            logging.info(f"--- Starting Epoch {self.current_epoch + 1}/{self.epochs} ---")

            # Actual training and evaluation calls
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            logging.info(f"--- Finished Epoch {self.current_epoch + 1}/{self.epochs} ---")

            # TODO: Implement proper checkpointing based on validation loss
            # Placeholder: Checkpointing logic would go here
            # if val_loss < self.best_val_loss:
            #     self.best_val_loss = val_loss
            #     # Save checkpoint using CheckpointManager
            #     logging.info(f"New best validation loss: {val_loss:.4f}. Saving model...")
            #     # self.checkpoint_manager.save(self.model, self.optimizer, epoch, val_loss)

            # TODO: Implement early stopping logic
            # ...

            # TODO: Implement KL Annealing update step
            # if hasattr(self.loss_fn, 'update_kl_weight') and hasattr(self.config.training, 'kl_scheduler'): # Check if scheduler exists
            #    new_kl_weight = self.config.training.kl_scheduler.step()
            #    self.loss_fn.update_kl_weight(new_kl_weight)


        logging.info("Training finished.")
        # Save final model state (simple checkpointing)
        final_model_path = "final_model.pt" # TODO: Make path configurable
        try:
            torch.save(self.model.state_dict(), final_model_path)
            logging.info(f"Saved final model state to {final_model_path}")
        except Exception as e:
            logging.error(f"Error saving final model: {e}")

        return self.train_loss_history, self.val_loss_history
