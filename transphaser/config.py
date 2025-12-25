
import json
import logging
import pandas as pd
from typing import Optional, Dict, List, Any
import os
import math

try:
    from pydantic import BaseModel, Field, field_validator, ValidationError
    import torch  # Needed for device check validator
except ImportError:
    logging.error("Pydantic library not found. Please install it: pip install pydantic")
    raise ImportError("Pydantic is required but not installed. Please run: pip install pydantic")

# Define potential sub-configurations (examples)
class DataConfig(BaseModel):
    locus_columns: List[str] = ["HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"] # Example k=6
    covariate_columns: List[str] = []
    categorical_covariate_columns: Optional[List[str]] = None # Explicit list for categorical
    numerical_covariate_columns: Optional[List[str]] = None # Explicit list for numerical
    validation_split_ratio: float = 0.2 # Ratio for train/validation split
    unphased_data_path: Optional[str] = None # Path to unphased genotype data
    phased_data_path: Optional[str] = None # Path to phased haplotype data (optional, for evaluation)

class ModelEncoderConfig(BaseModel): # Example nested config
    ff_dim: int = 128
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1

class ModelDecoderConfig(BaseModel): # Example nested config
    ff_dim: int = 128
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1

class ModelConfig(BaseModel):
    # Common params
    embedding_dim: int = 128
    latent_dim: int = 64 # Example latent dimension for VAE
    # Nested encoder/decoder configs
    encoder: ModelEncoderConfig = Field(default_factory=ModelEncoderConfig)
    decoder: ModelDecoderConfig = Field(default_factory=ModelDecoderConfig)

class TrainingLRSchedulerConfig(BaseModel): # Example nested config for LR scheduler
    type: str
    args: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = "Adam" # Could use Literal["Adam", "AdamW"]
    lr_scheduler: Optional[TrainingLRSchedulerConfig] = None # Nested scheduler config
    gradient_accumulation_steps: int = 1
    kl_annealing_type: str = 'linear' # Options: 'linear', 'sigmoid', 'cyclical', 'none'
    kl_annealing_epochs: int = 1 # Number of epochs over which to anneal KL weight
    kl_annealing_max_weight: float = 1.0 # Maximum KL weight
    reconstruction_weight: float = 1.0 # Weight for reconstruction loss term
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 1 # Save every N epochs
    final_model_filename: str = "final_model.pt"
    # Early Stopping
    early_stopping_patience: Optional[int] = None # Patience in epochs, None to disable
    # Logging
    log_interval: int = 50 # Log every N batches
    # Imputation
    genotype_mask_prob: float = 0.0 # Probability of masking genotype alleles for robust training


class ReportingConfig(BaseModel):
     formats: List[str] = ["json", "txt"] # Report formats to generate
     base_filename: str = "final_report" # Base name for report files
     plot_filename: str = "training_loss_curves.png" # Filename for loss curve plot


# Main Configuration Class
class TransPhaserConfig(BaseModel):
    """
    Main configuration class for the TransPhaser project (formerly HLAPhasingConfig).
    Loads configuration from a file or uses default values.
    """
    model_name: str = "TransPhaser-v1"
    seed: int = 42
    device: str = "cpu" # Device to use ('cuda' or 'cpu')
    output_dir: str = "output" # Directory for saving results, models, etc.
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig) # Added reporting config

    def __init__(self, config_path: Optional[str] = None, **data):
        """
        Initializes the configuration.
        """
        loaded_data = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_data = json.load(f)
                logging.info(f"Loaded configuration from: {config_path}")
            except FileNotFoundError:
                logging.warning(f"Config file not found at {config_path}. Using defaults.")
            except Exception as e:
                 logging.error(f"Error loading config file {config_path}: {e}. Using defaults.")

        # Merge loaded data with keyword arguments (kwargs override file data)
        merged_data = {**loaded_data, **data}

        # Initialize Pydantic BaseModel
        super().__init__(**merged_data)


    # Pydantic V2 field validators
    @field_validator('training')
    @classmethod
    def check_kl_annealing(cls, training_config: TrainingConfig) -> TrainingConfig:
        if training_config.kl_annealing_type not in ['linear', 'sigmoid', 'cyclical', 'none']:
             raise ValueError(f"Invalid kl_annealing_type: {training_config.kl_annealing_type}. Must be one of ['linear', 'sigmoid', 'cyclical', 'none']")
        return training_config

    @field_validator('device')
    @classmethod
    def check_device(cls, v: str) -> str:
        if v not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        if v == 'cuda' and not torch.cuda.is_available():
            logging.warning("Config specified 'cuda' but CUDA is not available. Falling back to 'cpu'.")
            return 'cpu'
        return v

    def save(self, filepath):
        """Saves the current configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists
            with open(filepath, 'w') as f:
                # Use Pydantic's model_dump_json for v2+ or json() for v1
                json_dump_method = getattr(self, 'model_dump_json', getattr(self, 'json', None))
                if json_dump_method:
                     f.write(json_dump_method(indent=4))
                else:
                     # Fallback if method not found (shouldn't happen with Pydantic)
                     json.dump(self.dict(), f, indent=4) # Use dict() for v1 fallback
            logging.info(f"Configuration saved to: {filepath}")
        except Exception as e:
            logging.error(f"Error saving configuration to {filepath}: {e}")

# Maintain alias for compatibility during migration if needed
HLAPhasingConfig = TransPhaserConfig

class LocusMetadataManager:
    """
    Manages metadata about HLA loci.
    """
    def __init__(self, locus_config: Optional[Dict[str, Any]] = None):
        self.metadata = locus_config if locus_config is not None else {}
        logging.debug("LocusMetadataManager initialized.")

    def get_locus_info(self, locus: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(locus)

    def get_allele_frequency(self, locus: str, allele: str, population: Optional[str] = None) -> Optional[float]:
        locus_info = self.get_locus_info(locus)
        if locus_info and 'frequencies' in locus_info:
            freq_data = locus_info['frequencies']
            if population and isinstance(freq_data, dict) and population in freq_data:
                return freq_data[population].get(allele)
            elif not population and isinstance(freq_data, dict):
                 if allele in freq_data:
                     return freq_data.get(allele)
                 elif 'ALL' in freq_data and isinstance(freq_data['ALL'], dict):
                      return freq_data['ALL'].get(allele)
        return None


class AlleleFrequencyPrior:
    """
    Loads and provides access to external allele frequency data.
    """
    def __init__(self, prior_source: str, prior_weight: float = 1.0):
        self.prior_source = prior_source
        self.prior_weight = prior_weight
        self.frequency_data = self._load_frequencies()
        logging.debug("AlleleFrequencyPrior initialized.")

    def _load_frequencies(self):
        logging.info(f"Loading allele frequencies from {self.prior_source}")
        if not os.path.exists(self.prior_source):
             logging.warning(f"Prior source file not found: {self.prior_source}. Prior will be empty.")
             return {}

        try:
            df = pd.read_csv(self.prior_source)
            freq_dict = {}
            has_population = 'Population' in df.columns

            for _, row in df.iterrows():
                locus = row['Locus']
                allele = row['Allele']
                freq = row['Frequency']
                pop = row.get('Population', 'ALL') if has_population else 'ALL'

                if locus not in freq_dict:
                    freq_dict[locus] = {}

                if has_population:
                    if pop not in freq_dict[locus]:
                        freq_dict[locus][pop] = {}
                    freq_dict[locus][pop][allele] = freq
                else:
                    freq_dict[locus][allele] = freq

            return freq_dict

        except Exception as e:
            logging.error(f"Error loading prior source: {e}")
            return {}


    def get_frequency(self, locus: str, allele: str, population: Optional[str] = None) -> Optional[float]:
        locus_data = self.frequency_data.get(locus)
        if not locus_data:
            return None

        target_pop = population if population else 'ALL'

        if target_pop in locus_data and isinstance(locus_data[target_pop], dict):
            return locus_data[target_pop].get(allele)
        elif 'ALL' in locus_data and isinstance(locus_data['ALL'], dict) and not population:
             return locus_data['ALL'].get(allele)
        elif isinstance(locus_data, dict) and allele in locus_data:
             return locus_data.get(allele)

        return None

    def get_prior_log_probs(self, locus: str, alleles: List[str], population: Optional[str] = None) -> Optional[Dict[str, float]]:
        log_probs = {}
        has_freq_for_locus = False

        for allele in alleles:
            freq = self.get_frequency(locus, allele, population)
            if freq is not None and freq > 0:
                log_probs[allele] = math.log(freq) * self.prior_weight
                has_freq_for_locus = True
            else:
                log_probs[allele] = -20.0 * self.prior_weight

        return log_probs if has_freq_for_locus else None


class CovariateManager:
    def __init__(self, categorical_covariates: Optional[List[str]] = None,
                 numerical_covariates: Optional[List[str]] = None,
                 encoding_strategies: Optional[Dict[str, str]] = None):
        self.categorical_covariates = categorical_covariates if categorical_covariates is not None else []
        self.numerical_covariates = numerical_covariates if numerical_covariates is not None else []
        self.encoding_strategies = encoding_strategies if encoding_strategies is not None else {}

    def get_encoding_strategy(self, covariate_name: str) -> Optional[str]:
        return self.encoding_strategies.get(covariate_name)
