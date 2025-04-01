import os
import json
import logging
from typing import Dict, Any, Optional, List

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not found. Plotting functionality will be disabled.")

class PerformanceReporter:
    """
    Generates comprehensive performance reports including metrics logs,
    training curves, and runtime statistics.
    """
    SUPPORTED_FORMATS = ["json", "txt"]

    def __init__(self, output_dir: str):
        """
        Initializes the PerformanceReporter.

        Args:
            output_dir (str): The directory where reports and plots will be saved.
        """
        if not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string.")

        self.output_dir = output_dir
        self.metrics_log: Dict[str, Any] = {} # Initialize metrics log

        # Ensure output directory exists
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create output directory {self.output_dir}: {e}")
            raise # Re-raise error if directory creation fails

    def log_metric(self, key: str, value: Any):
        """
        Logs a metric or a set of metrics.

        Args:
            key (str): The identifier for the metric (e.g., 'epoch_1', 'summary', 'training_history').
            value (Any): The value of the metric. Can be a single value, dict, or list.
        """
        if not isinstance(key, str):
            raise TypeError("Metric key must be a string.")
        self.metrics_log[key] = value
        logging.debug(f"Logged metric '{key}': {value}")

    def generate_report(self, report_format: str = "json", report_filename: str = "performance_report"):
        """
        Generates a performance report file based on the logged metrics.

        Args:
            report_format (str): The format for the report ('json' or 'txt'). Defaults to 'json'.
            report_filename (str): The base name for the report file (extension will be added).
                                   Defaults to 'performance_report'.
        """
        if report_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported report format: '{report_format}'. Supported formats: {self.SUPPORTED_FORMATS}")

        # Construct filename correctly, removing existing extension if present
        base_filename, _ = os.path.splitext(report_filename)
        filename = f"{base_filename}.{report_format}"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, 'w') as f:
                if report_format == "json":
                    json.dump(self.metrics_log, f, indent=4)
                elif report_format == "txt":
                    for key, value in self.metrics_log.items():
                        f.write(f"--- {key} ---\n")
                        # Pretty print dictionaries or lists
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                f.write(f"  {sub_key}: {sub_value}\n")
                        elif isinstance(value, list):
                            for i, item in enumerate(value):
                                f.write(f"  Item {i}: {item}\n")
                        else:
                            f.write(f"  {value}\n")
                        f.write("\n")
            logging.info(f"Performance report saved to: {filepath}")
        except IOError as e:
            logging.error(f"Error writing report file {filepath}: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred during report generation: {e}")

    def plot_training_curves(self, plot_filename: str = "training_curves.png"):
        """
        Plots training and validation loss curves if data is available in metrics_log.

        Args:
            plot_filename (str): The name for the output plot file. Defaults to 'training_curves.png'.
        """
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Cannot plot training curves: matplotlib is not installed.")
            return

        history = self.metrics_log.get("training_history")
        if not isinstance(history, dict):
            logging.warning("No 'training_history' dictionary found in metrics log. Cannot plot curves.")
            return

        train_loss = history.get("train_loss")
        val_loss = history.get("val_loss")

        if not isinstance(train_loss, list) or not isinstance(val_loss, list):
            logging.warning("Training history does not contain 'train_loss' and 'val_loss' lists. Cannot plot curves.")
            return

        if not train_loss and not val_loss:
             logging.warning("Training and validation loss lists are empty. Cannot plot curves.")
             return

        filepath = os.path.join(self.output_dir, plot_filename)

        try:
            plt.figure(figsize=(10, 6)) # Mocked in test

            epochs = range(1, len(train_loss) + 1)
            if train_loss:
                plt.plot(epochs, train_loss, label='Training Loss') # Mocked in test

            # Handle potential NaNs or length mismatch in validation loss if needed
            val_epochs = range(1, len(val_loss) + 1) # Assuming same length for simplicity now
            if val_loss:
                plt.plot(val_epochs, val_loss, label='Validation Loss', marker='o') # Mocked in test

            plt.xlabel("Epoch") # Mocked in test
            plt.ylabel("Loss") # Mocked in test
            plt.title("Training and Validation Loss Curves") # Mocked in test
            if train_loss or val_loss: # Only add legend if something was plotted
                 plt.legend() # Mocked in test
            plt.grid(True) # Mocked in test
            plt.savefig(filepath) # Mocked in test
            plt.close() # Mocked in test
            logging.info(f"Training curves plot saved to: {filepath}")

        except Exception as e:
            logging.error(f"Error generating training curves plot {filepath}: {e}")
            # Ensure plot is closed even if error occurs during saving
            if MATPLOTLIB_AVAILABLE:
                plt.close()

    # Potential future methods:
    # def compare_runs(self, run_dirs: List[str]): pass
    # def plot_metric(self, metric_name: str): pass
