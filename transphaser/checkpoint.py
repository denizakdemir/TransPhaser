import os
import torch
import glob # For finding checkpoint files
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CheckpointManager:
    """
    Manages saving and loading model checkpoints during training.
    Keeps track of the best performing models and limits the total number
    of saved checkpoints.
    """
    def __init__(self, save_dir, max_checkpoints=5):
        """
        Initializes the CheckpointManager.

        Args:
            save_dir (str): The directory where checkpoints will be saved.
            max_checkpoints (int): The maximum number of checkpoints to keep.
                                   Older checkpoints (excluding the best one) will be removed.
                                   Defaults to 5.
        """
        if not isinstance(save_dir, str) or not save_dir:
            raise ValueError("save_dir must be a non-empty string.")
        if not isinstance(max_checkpoints, int) or max_checkpoints <= 0:
            raise ValueError("max_checkpoints must be a positive integer.")

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('inf') # Assuming lower is better (e.g., validation loss)

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        logging.info(f"CheckpointManager initialized. Saving checkpoints to: {self.save_dir}")

    def save(self, model, optimizer, epoch, current_metric, is_best=False, **kwargs):
        """
        Saves a checkpoint.

        Args:
            model: The model instance (nn.Module).
            optimizer: The optimizer instance.
            epoch (int): The current epoch number.
            current_metric (float): The metric value for this checkpoint (e.g., validation loss).
            is_best (bool): Whether this checkpoint corresponds to the best metric so far.
            **kwargs: Additional metadata to save (e.g., lr_scheduler state, training args).
        """
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': current_metric,
            **kwargs # Include any extra metadata
        }

        # Define filename: e.g., checkpoint_epoch_005_loss_0.1234.pt
        filename = f"checkpoint_epoch_{epoch:03d}_metric_{current_metric:.4f}.pt"
        filepath = os.path.join(self.save_dir, filename)

        logging.info(f"Saving checkpoint: {filepath}")
        torch.save(state, filepath)

        # Save a separate 'best_model.pt' if this is the best metric so far
        if is_best:
            best_filepath = os.path.join(self.save_dir, "best_model.pt")
            logging.info(f"Saving best model checkpoint to: {best_filepath} (Metric: {current_metric:.4f})")
            torch.save(state, best_filepath)
            self.best_metric = current_metric # Update best metric

        # --- Pruning old checkpoints ---
        self._prune_checkpoints()


    def load(self, model, optimizer, filepath):
        """
        Loads a checkpoint from a file.

        Args:
            model: The model instance to load state into.
            optimizer: The optimizer instance to load state into.
            filepath (str): The path to the checkpoint file.

        Returns:
            dict: The loaded state dictionary (including epoch, metric, and any extra kwargs).
                  Returns None if the file doesn't exist.
        """
        if not os.path.exists(filepath):
            logging.warning(f"Checkpoint file not found: {filepath}")
            return None

        logging.info(f"Loading checkpoint: {filepath}")
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage) # Load to CPU first

        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info(f"Checkpoint loaded successfully (Epoch {checkpoint.get('epoch', '?')}, Metric: {checkpoint.get('metric', '?'):.4f})")
            return checkpoint
        except Exception as e:
            logging.error(f"Error loading checkpoint from {filepath}: {e}")
            return None


    def load_latest(self, model, optimizer):
        """Loads the most recent checkpoint based on epoch number."""
        checkpoints = self._get_sorted_checkpoints()
        if not checkpoints:
            logging.info("No checkpoints found to load latest from.")
            return None
        latest_filepath = checkpoints[-1][1] # Get path of the last checkpoint
        return self.load(model, optimizer, latest_filepath)


    def load_best(self, model, optimizer):
        """Loads the checkpoint marked as 'best_model.pt'."""
        best_filepath = os.path.join(self.save_dir, "best_model.pt")
        if not os.path.exists(best_filepath):
            logging.warning(f"Best model checkpoint not found: {best_filepath}")
            return None
        return self.load(model, optimizer, best_filepath)


    def _get_sorted_checkpoints(self):
        """Returns a list of (epoch, filepath) tuples, sorted by epoch."""
        checkpoint_files = glob.glob(os.path.join(self.save_dir, "checkpoint_epoch_*.pt"))
        checkpoints = []
        for f in checkpoint_files:
            try:
                # Extract epoch number from filename
                epoch_str = os.path.basename(f).split('_')[2]
                epoch = int(epoch_str)
                checkpoints.append((epoch, f))
            except (IndexError, ValueError):
                logging.warning(f"Could not parse epoch from checkpoint filename: {f}")
        return sorted(checkpoints, key=lambda x: x[0]) # Sort by epoch


    def _prune_checkpoints(self):
        """Removes older checkpoints, keeping only the latest 'max_checkpoints' (excluding best)."""
        checkpoints = self._get_sorted_checkpoints()
        num_to_delete = len(checkpoints) - self.max_checkpoints

        if num_to_delete > 0:
            checkpoints_to_delete = checkpoints[:num_to_delete]
            logging.info(f"Pruning {len(checkpoints_to_delete)} old checkpoints...")
            for epoch, filepath in checkpoints_to_delete:
                try:
                    os.remove(filepath)
                    logging.debug(f"Removed checkpoint: {filepath}")
                except OSError as e:
                    logging.error(f"Error removing checkpoint {filepath}: {e}")
