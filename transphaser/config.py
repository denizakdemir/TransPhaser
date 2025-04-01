import json
import logging
import pandas as pd # Added for potential frequency file loading
from typing import Optional, Dict, List, Any
import os
import math

# Configure basic logging if not already done elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import Pydantic components
try:
    from pydantic import BaseModel, Field, validator, ValidationError
except ImportError:
    logging.error("Pydantic library not found. Please install it: pip install pydantic")
    # Re-raise the error to prevent execution with missing dependency
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
    # Add other data-related params like validation rules etc.

class ModelEncoderConfig(BaseModel): # Example nested config
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    # Add other encoder-specific params

class ModelDecoderConfig(BaseModel): # Example nested config
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    # Add other decoder-specific params

class ModelConfig(BaseModel):
    # Common params
    embedding_dim: int = 128
    num_heads: int = 8
    # Removed num_layers, ff_dim, dropout as they might be encoder/decoder specific
    latent_dim: int = 64 # Example latent dimension for VAE
    # Nested encoder/decoder configs
    encoder: ModelEncoderConfig = Field(default_factory=ModelEncoderConfig)
    decoder: ModelDecoderConfig = Field(default_factory=ModelDecoderConfig)
    # Add other model-wide params

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


class ReportingConfig(BaseModel):
     formats: List[str] = ["json", "txt"] # Report formats to generate
     base_filename: str = "final_report" # Base name for report files
     plot_filename: str = "training_loss_curves.png" # Filename for loss curve plot


# Main Configuration Class
class HLAPhasingConfig(BaseModel):
    """
    Main configuration class for the HLA Phasing project, using Pydantic.
    Loads configuration from a file or uses default values.
    """
    model_name: str = "HLA-Phaser-v1"
    seed: int = 42
    device: str = "cpu" # Device to use ('cuda' or 'cpu')
    output_dir: str = "output" # Directory for saving results, models, etc.
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig) # Added reporting config

    # Add other top-level configs like evaluation etc. if needed

    def __init__(self, config_path: Optional[str] = None, **data):
        """
        Initializes the configuration.
        Loads from config_path (JSON) if provided, otherwise uses defaults
        and allows overrides via keyword arguments.

        Args:
            config_path (Optional[str]): Path to a JSON configuration file.
            **data: Keyword arguments to override default values or values from file.
        """
        loaded_data = {}
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    loaded_data = json.load(f)
                logging.info(f"Loaded configuration from: {config_path}")
            except FileNotFoundError:
                logging.warning(f"Config file not found at {config_path}. Using defaults.")
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {config_path}. Using defaults.")
            except Exception as e:
                 logging.error(f"Error loading config file {config_path}: {e}. Using defaults.")

        # Merge loaded data with keyword arguments (kwargs override file data)
        merged_data = {**loaded_data, **data}

        # Initialize Pydantic BaseModel
        try:
            # Pydantic v2+ generally handles nested models better with direct init
            super().__init__(**merged_data)
            logging.info("Configuration initialized successfully.")
        except ValidationError as e:
            logging.error(f"Configuration validation failed:\n{e}")
            raise # Re-raise the validation error
        except TypeError as e:
             # Handle potential issues with nested model initialization if super().__init__ fails
             # This might be less necessary with Pydantic v2+ but kept for robustness
             logging.warning(f"Direct initialization failed ({e}), attempting manual nested init (may indicate Pydantic version issue or config structure mismatch).")
             try:
                 init_data = {}
                 # Use model_fields for Pydantic v2+
                 fields_dict = getattr(self, '__fields__', getattr(self, 'model_fields', {}))
                 for field_name, field_model in fields_dict.items():
                     field_type = getattr(field_model, 'annotation', getattr(field_model, 'type_', None))
                     # Check if field_type is a class and a subclass of BaseModel
                     is_nested_model = isinstance(field_type, type) and issubclass(field_type, BaseModel)

                     if field_name in merged_data:
                         if is_nested_model:
                              nested_data = merged_data[field_name]
                              if isinstance(nested_data, dict):
                                   init_data[field_name] = field_type(**nested_data)
                              elif isinstance(nested_data, BaseModel): # Allow passing already instantiated models
                                   init_data[field_name] = nested_data
                              else:
                                   raise TypeError(f"Expected dict or BaseModel for nested model '{field_name}', got {type(nested_data)}")
                         else:
                              init_data[field_name] = merged_data[field_name]
                     else:
                         # Handle defaults (simplified, Pydantic usually handles this)
                         if getattr(field_model, 'default_factory', None):
                              init_data[field_name] = field_model.default_factory()
                         elif hasattr(field_model, 'default'):
                              init_data[field_name] = field_model.default

                 super().__init__(**init_data)
                 logging.info("Manual nested configuration initialized successfully.")
             except ValidationError as e_manual:
                 logging.error(f"Manual nested configuration validation failed:\n{e_manual}")
                 raise e_manual
             except Exception as e_manual_other:
                  logging.error(f"Error during manual nested init: {e_manual_other}")
                  raise e_manual_other


    # Example validator
    @validator('training')
    def check_kl_annealing(cls, training_config):
        if training_config.kl_annealing_type not in ['linear', 'sigmoid', 'cyclical', 'none']:
             raise ValueError(f"Invalid kl_annealing_type: {training_config.kl_annealing_type}. Must be one of ['linear', 'sigmoid', 'cyclical', 'none']")
        return training_config

    @validator('device')
    def check_device(cls, v):
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


class LocusMetadataManager:
    """
    Manages metadata about HLA loci, such as allele frequencies,
    gene positions, or other relevant biological information.
    """
    def __init__(self, locus_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the LocusMetadataManager.

        Args:
            locus_config (Optional[Dict[str, Any]]): A dictionary where keys are
                locus names and values are dictionaries containing metadata
                (e.g., {"HLA-A": {"frequency_source": "db1", "resolution": 4}}).
                Defaults to None, resulting in empty metadata.
        """
        self.metadata = locus_config if locus_config is not None else {}
        logging.debug("LocusMetadataManager initialized.")
        # Potential future work: Load metadata from external files (e.g., frequency databases)

    def get_locus_info(self, locus: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific locus."""
        return self.metadata.get(locus)

    def get_allele_frequency(self, locus: str, allele: str, population: Optional[str] = None) -> Optional[float]:
        """
        Retrieves the frequency of a specific allele within a locus,
        optionally filtered by population.
        """
        locus_info = self.get_locus_info(locus)
        if locus_info and 'frequencies' in locus_info:
            freq_data = locus_info['frequencies'] # Assuming dict like {pop: {allele: freq}} or {allele: freq}
            if population and isinstance(freq_data, dict) and population in freq_data:
                # Population-specific frequency
                return freq_data[population].get(allele)
            elif not population and isinstance(freq_data, dict):
                 # Check for allele frequency directly under locus (no population specified)
                 if allele in freq_data:
                     return freq_data.get(allele)
                 # Check for a default population key like 'ALL'
                 elif 'ALL' in freq_data and isinstance(freq_data['ALL'], dict):
                      return freq_data['ALL'].get(allele)
            # Add more sophisticated logic here if frequency structure is complex
            logging.debug(f"Frequency data found for locus {locus}, but not for allele {allele} / population {population}")

        return None # Return None if locus, frequencies, allele, or pop not found


class AlleleFrequencyPrior:
    """
    Loads and provides access to external allele frequency data,
    acting as a prior for the model.
    """
    def __init__(self, prior_source: str, prior_weight: float = 1.0):
        """
        Initializes the AlleleFrequencyPrior.

        Args:
            prior_source (str): Path to the allele frequency data file (e.g., CSV).
            prior_weight (float): Weight to apply to this prior. Defaults to 1.0.
        """
        if not isinstance(prior_source, str) or not prior_source:
             raise ValueError("prior_source must be a non-empty string path.")
        if not isinstance(prior_weight, (int, float)) or prior_weight < 0:
             raise ValueError("prior_weight must be a non-negative number.")

        self.prior_source = prior_source
        self.prior_weight = prior_weight
        self.frequency_data = self._load_frequencies()
        logging.debug("AlleleFrequencyPrior initialized.")

    def _load_frequencies(self):
        """
        Loads frequency data from the specified source (CSV expected).
        Assumes CSV format: Locus,Allele,Frequency,[Population] (Population optional)
        """
        logging.info(f"Loading allele frequencies from {self.prior_source}")
        if not os.path.exists(self.prior_source):
             logging.warning(f"Prior source file not found: {self.prior_source}. Prior will be empty.")
             return {}

        try:
            df = pd.read_csv(self.prior_source)
            freq_dict = {}
            # Determine if population column exists
            has_population = 'Population' in df.columns

            for _, row in df.iterrows():
                locus = row['Locus']
                allele = row['Allele']
                freq = row['Frequency']
                pop = row.get('Population', 'ALL') if has_population else 'ALL' # Default pop if col missing

                if locus not in freq_dict:
                    freq_dict[locus] = {}

                if has_population:
                    if pop not in freq_dict[locus]:
                        freq_dict[locus][pop] = {}
                    freq_dict[locus][pop][allele] = freq
                else:
                    # If no population column, store directly under locus
                    freq_dict[locus][allele] = freq

            logging.info(f"Successfully loaded frequencies for {len(freq_dict)} loci.")
            return freq_dict

        except pd.errors.EmptyDataError:
            logging.warning(f"Prior source file is empty: {self.prior_source}")
            return {}
        except KeyError as e:
             logging.error(f"Missing expected column in prior source file {self.prior_source}: {e}. Required: Locus, Allele, Frequency.")
             return {}
        except Exception as e:
            logging.error(f"Error loading or parsing prior source file {self.prior_source}: {e}")
            return {}


    def get_frequency(self, locus: str, allele: str, population: Optional[str] = None) -> Optional[float]:
        """Retrieves the frequency for a specific allele, considering population."""
        locus_data = self.frequency_data.get(locus)
        if not locus_data:
            return None

        target_pop = population if population else 'ALL'

        if target_pop in locus_data and isinstance(locus_data[target_pop], dict):
            # Found data for the specific or default population
            return locus_data[target_pop].get(allele)
        elif 'ALL' in locus_data and isinstance(locus_data['ALL'], dict) and not population:
             # Fallback to 'ALL' if no specific population was requested and 'ALL' exists
             return locus_data['ALL'].get(allele)
        elif isinstance(locus_data, dict) and allele in locus_data and not any(isinstance(v, dict) for v in locus_data.values()):
             # Check if frequencies are directly under locus (no population structure)
             return locus_data.get(allele)


        return None # Allele not found for the specified locus/population

    def get_prior_log_probs(self, locus: str, alleles: List[str], population: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Returns log probabilities based on the prior frequencies for a list of alleles.
        """
        log_probs = {}
        has_freq_for_locus = False

        for allele in alleles:
            freq = self.get_frequency(locus, allele, population)
            if freq is not None and freq > 0:
                # Use math.log for standard log
                log_probs[allele] = math.log(freq) * self.prior_weight
                has_freq_for_locus = True
            else:
                # Handle alleles not in prior (assign low prob? ignore?)
                # Assigning a very small log probability instead of -inf might be more stable
                # This value should ideally be configurable or based on a smoothing factor
                log_probs[allele] = -20.0 * self.prior_weight # Example: Assign low log prob (-20 corresponds to ~1e-9 freq)

        # Return None if no frequency information was found for any allele in this locus/pop
        return log_probs if has_freq_for_locus else None


class CovariateManager:
    """
    Manages covariate information, including definitions, encoding strategies.
    (This might overlap significantly with CovariateEncoder - consider merging or refining roles).
    """
    def __init__(self, categorical_covariates: Optional[List[str]] = None,
                 numerical_covariates: Optional[List[str]] = None,
                 encoding_strategies: Optional[Dict[str, str]] = None):
        """
        Initializes the CovariateManager.

        Args:
            categorical_covariates (Optional[List[str]]): List of categorical covariate names.
            numerical_covariates (Optional[List[str]]): List of numerical covariate names.
            encoding_strategies (Optional[Dict[str, str]]): Dictionary mapping covariate names
                                                            to specific encoding strategies
                                                            (e.g., {'Race': 'one-hot', 'Age': 'standardize'}).
                                                            Defaults to None.
        """
        self.categorical_covariates = categorical_covariates if categorical_covariates is not None else []
        self.numerical_covariates = numerical_covariates if numerical_covariates is not None else []
        self.encoding_strategies = encoding_strategies if encoding_strategies is not None else {}
        logging.debug("CovariateManager initialized.")

    def get_encoding_strategy(self, covariate_name: str) -> Optional[str]:
        """Returns the specified encoding strategy for a covariate."""
        # Could add default strategy logic here if needed
        return self.encoding_strategies.get(covariate_name)


# Example usage (optional)
if __name__ == '__main__':
    # Configure logging for example run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create dummy prior file
    dummy_prior_path = "dummy_freqs.csv"
    if not os.path.exists(dummy_prior_path): # Avoid error if run multiple times
        with open(dummy_prior_path, "w") as f:
            f.write("Locus,Allele,Frequency,Population\n")
            f.write("HLA-A,A*01:01,0.25,EUR\n")
            f.write("HLA-A,A*02:01,0.15,EUR\n")
            f.write("HLA-A,A*01:01,0.10,ASN\n") # Different freq for different pop
            f.write("HLA-B,B*07:02,0.10,EUR\n")
            f.write("HLA-C,C*01:02,0.05,ALL\n") # Using 'ALL' population

    try:
        # Test default init
        print("--- Default Config ---")
        default_config = HLAPhasingConfig()
        print(default_config.model_dump_json(indent=2)) # Use model_dump_json for Pydantic v2+

        # Test init with dict override
        print("\n--- Config with Dict Override ---")
        override_dict = {"training": {"learning_rate": 5e-5, "batch_size": 64}}
        dict_config = HLAPhasingConfig(**override_dict)
        print(dict_config.model_dump_json(indent=2))

        # Test init with file (create dummy file)
        print("\n--- Config with File Load ---")
        dummy_path = "temp_config.json"
        dummy_data = {"model_name": "LoadedFromFile", "data": {"covariate_columns": ["Age"]}}
        with open(dummy_path, 'w') as f:
            json.dump(dummy_data, f)
        file_config = HLAPhasingConfig(config_path=dummy_path)
        print(file_config.model_dump_json(indent=2))
        if os.path.exists(dummy_path): os.remove(dummy_path) # Clean up dummy file

        # Test init with file and dict override
        print("\n--- Config with File + Override ---")
        with open(dummy_path, 'w') as f:
             json.dump(dummy_data, f)
        override_dict_2 = {"training": {"epochs": 100}}
        file_override_config = HLAPhasingConfig(config_path=dummy_path, **override_dict_2)
        print(file_override_config.model_dump_json(indent=2))
        if os.path.exists(dummy_path): os.remove(dummy_path)

        # Test validation error
        print("\n--- Testing Validation Error ---")
        invalid_data = {"training": {"learning_rate": "invalid"}}
        try:
             HLAPhasingConfig(**invalid_data)
        except ValidationError as e:
             print(f"Caught expected validation error:\n{e}")

        # Test LocusMetadataManager
        print("\n--- LocusMetadataManager ---")
        # Example with frequency data structure
        meta_data = {
            "HLA-A": {"res": 4, "frequencies": {"EUR": {"A*01:01": 0.25, "A*02:01": 0.15}, "ASN": {"A*01:01": 0.10}}},
            "HLA-B": {"res": 2, "frequencies": {"ALL": {"B*07:02": 0.10}}},
            "HLA-C": {"res": 2} # No frequency info
        }
        meta_manager = LocusMetadataManager(meta_data)
        print(f"Metadata for HLA-A: {meta_manager.get_locus_info('HLA-A')}")
        print(f"Freq A*01:01 (EUR): {meta_manager.get_allele_frequency('HLA-A', 'A*01:01', 'EUR')}")
        print(f"Freq A*01:01 (ASN): {meta_manager.get_allele_frequency('HLA-A', 'A*01:01', 'ASN')}")
        print(f"Freq A*01:01 (AFR): {meta_manager.get_allele_frequency('HLA-A', 'A*01:01', 'AFR')}") # Should be None
        print(f"Freq A*01:01 (None): {meta_manager.get_allele_frequency('HLA-A', 'A*01:01')}") # Should be None (no 'ALL')
        print(f"Freq B*07:02 (EUR): {meta_manager.get_allele_frequency('HLA-B', 'B*07:02', 'EUR')}") # Should be None
        print(f"Freq B*07:02 (None): {meta_manager.get_allele_frequency('HLA-B', 'B*07:02')}") # Should get from 'ALL'
        print(f"Freq C*01:01 (None): {meta_manager.get_allele_frequency('HLA-C', 'C*01:01')}") # Should be None

        # Test AlleleFrequencyPrior
        print("\n--- AlleleFrequencyPrior ---")
        try:
            prior = AlleleFrequencyPrior(prior_source=dummy_prior_path)
            print(f"Loaded prior data structure: {type(prior.frequency_data)}") # Show type, not full data
            print(f"Freq A*01:01 (EUR): {prior.get_frequency('HLA-A', 'A*01:01', 'EUR')}")
            print(f"Freq A*01:01 (ASN): {prior.get_frequency('HLA-A', 'A*01:01', 'ASN')}")
            print(f"Freq B*07:02 (EUR): {prior.get_frequency('HLA-B', 'B*07:02', 'EUR')}")
            print(f"Freq C*01:02 (ALL): {prior.get_frequency('HLA-C', 'C*01:02', 'ALL')}")
            print(f"Freq C*01:02 (None): {prior.get_frequency('HLA-C', 'C*01:02')}") # Should get from ALL
            print(f"Freq D*01:01 (None): {prior.get_frequency('HLA-D', 'D*01:01')}") # Should be None
            # Test log probs
            alleles_a = ["A*01:01", "A*02:01", "A*03:01"]
            log_probs_a_eur = prior.get_prior_log_probs('HLA-A', alleles_a, 'EUR')
            print(f"Log Probs HLA-A (EUR): {log_probs_a_eur}")
            log_probs_a_asn = prior.get_prior_log_probs('HLA-A', alleles_a, 'ASN')
            print(f"Log Probs HLA-A (ASN): {log_probs_a_asn}")
            log_probs_c = prior.get_prior_log_probs('HLA-C', ["C*01:02", "C*02:02"], 'ALL')
            print(f"Log Probs HLA-C (ALL): {log_probs_c}")

        except Exception as e_prior:
            print(f"Error testing AlleleFrequencyPrior: {e_prior}")

        # Test CovariateManager
        print("\n--- CovariateManager ---")
        cov_manager = CovariateManager(categorical_covariates=['Race'], numerical_covariates=['Age'])
        print(f"Covariates managed: Cat={cov_manager.categorical_covariates}, Num={cov_manager.numerical_covariates}")


    except ImportError:
        print("\nPydantic not installed, skipping example usage.")
    except Exception as e:
        print(f"\nAn error occurred during example usage: {e}")
    finally:
        # Clean up dummy prior file
        if os.path.exists(dummy_prior_path):
            os.remove(dummy_prior_path)
            logging.info(f"Removed dummy prior file: {dummy_prior_path}")
        # Clean up dummy config file if it still exists
        if os.path.exists(dummy_path):
             os.remove(dummy_path)
             logging.info(f"Removed dummy config file: {dummy_path}")
