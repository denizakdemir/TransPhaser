import json
import logging
from typing import Optional, Dict, List, Any
import os # Added for path checks
import math # Added for AlleleFrequencyPrior

# Import Pydantic components
try:
    from pydantic import BaseModel, Field, validator, ValidationError
except ImportError:
    logging.error("Pydantic library not found. Please install it: pip install pydantic")
    # Define dummy classes if Pydantic is not available to avoid crashing on import
    class BaseModel: pass
    class Field: pass
    class validator: pass
    class ValidationError(Exception): pass

# Define potential sub-configurations (examples)
class DataConfig(BaseModel):
    locus_columns: List[str] = ["HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1"] # Example k=6
    covariate_columns: List[str] = []
    unphased_data_path: Optional[str] = None # Path to unphased genotype data
    phased_data_path: Optional[str] = None # Path to phased haplotype data (optional, for evaluation)
    # Add other data-related params like validation rules etc.

class ModelConfig(BaseModel):
    # Common params
    embedding_dim: int = 128
    num_heads: int = 8
    num_layers: int = 6
    ff_dim: int = 512
    dropout: float = 0.1
    # Specific params (could be nested further)
    covariate_dim: int = 32 # Example dimension after encoding
    latent_dim: int = 64 # Example latent dimension for VAE
    # Add specific encoder/decoder configs if needed

class TrainingConfig(BaseModel):
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    optimizer: str = "Adam" # Could use Literal["Adam", "AdamW"]
    lr_scheduler: Optional[str] = None # e.g., "CosineAnnealingLR"
    grad_accumulation_steps: int = 1
    kl_annealing_type: str = 'linear'
    kl_annealing_steps: int = 10000
    # Add early stopping, checkpointing config etc.

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

    # Add other top-level configs like evaluation, logging etc.

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
            # Pydantic v1 uses __init__ differently, need to handle nested manually if needed
            # For simplicity, assume direct init works or adjust based on Pydantic version
            super().__init__(**merged_data)
            logging.info("Configuration initialized successfully.")
        except ValidationError as e:
            logging.error(f"Configuration validation failed:\n{e}")
            raise # Re-raise the validation error
        except TypeError as e:
             # Handle potential issues with nested model initialization if super().__init__ fails
             logging.warning(f"Direct initialization failed ({e}), attempting manual nested init.")
             # Manual nested init (example for Pydantic v1 style)
             try:
                 init_data = {}
                 for field_name, field_model in self.__fields__.items():
                     if field_name in merged_data:
                         # Check if field type is a subclass of BaseModel before attempting nested init
                         # Also check if the field_model.type_ is actually a type
                         field_type = field_model.type_
                         if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                              # Ensure the value is a dict before passing to nested model
                              nested_data = merged_data[field_name]
                              if isinstance(nested_data, dict):
                                   init_data[field_name] = field_type(**nested_data)
                              else:
                                   # Handle cases where nested data isn't a dict (e.g., already an object?)
                                   # Or raise a more specific error
                                   raise TypeError(f"Expected dict for nested model '{field_name}', got {type(nested_data)}")
                         else:
                              init_data[field_name] = merged_data[field_name]
                     elif field_model.default_factory:
                          init_data[field_name] = field_model.default_factory()
                     # Handle fields without defaults if necessary, Pydantic might raise error anyway
                     # elif field_model.required:
                     #     raise ValueError(f"Missing required field: {field_name}")
                     else:
                          # Use get_default() if available (Pydantic v2) or .default
                          default_value = getattr(field_model, 'get_default', lambda: field_model.default)()
                          init_data[field_name] = default_value


                 super().__init__(**init_data)
                 logging.info("Manual nested configuration initialized successfully.")
             except ValidationError as e_manual:
                 logging.error(f"Manual nested configuration validation failed:\n{e_manual}")
                 raise e_manual # Re-raise the validation error
             except Exception as e_manual_other:
                  logging.error(f"Error during manual nested init: {e_manual_other}")
                  raise e_manual_other


    # Example validator
    @validator('training')
    def check_kl_annealing(cls, training_config):
        if training_config.kl_annealing_type not in ['linear', 'sigmoid', 'cyclical', 'none']:
             raise ValueError("Invalid kl_annealing_type")
        return training_config

    def save(self, filepath):
        """Saves the current configuration to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                # Use Pydantic's json() method for proper serialization
                f.write(self.json(indent=4))
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
        print("Placeholder: LocusMetadataManager initialized.")
        # Potential future work: Load metadata from external files (e.g., frequency databases)

    def get_locus_info(self, locus: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific locus."""
        return self.metadata.get(locus)

    def get_allele_frequency(self, locus: str, allele: str, population: Optional[str] = None) -> Optional[float]:
        """
        Retrieves the frequency of a specific allele within a locus,
        optionally filtered by population. (Placeholder)
        """
        # Placeholder implementation - requires loading frequency data
        print(f"Placeholder: Getting frequency for {locus} allele {allele} (Pop: {population})")
        locus_info = self.get_locus_info(locus)
        if locus_info and 'frequencies' in locus_info:
            # Add logic to query frequency data based on allele and population
            pass
        return None # Placeholder return


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
        print("Placeholder: AlleleFrequencyPrior initialized.")

    def _load_frequencies(self):
        """Loads frequency data from the specified source."""
        # Placeholder implementation
        print(f"Placeholder: Loading allele frequencies from {self.prior_source}")
        # Actual implementation would involve:
        # 1. Checking if file exists (os.path.exists(self.prior_source)).
        # 2. Parsing the file based on its format (e.g., pd.read_csv).
        # 3. Structuring the data for efficient lookup (e.g., nested dict: {locus: {allele: freq}}).
        if not os.path.exists(self.prior_source):
             logging.warning(f"Prior source file not found: {self.prior_source}. Prior will be empty.")
             return {}
        # Dummy data structure
        return {"HLA-A": {"A*01:01": 0.2, "A*02:01": 0.15}} # Example

    def get_frequency(self, locus: str, allele: str) -> Optional[float]:
        """Retrieves the frequency for a specific allele."""
        return self.frequency_data.get(locus, {}).get(allele)

    def get_prior_log_probs(self, locus: str, alleles: List[str]) -> Optional[Dict[str, float]]:
        """
        Returns log probabilities based on the prior frequencies for a list of alleles.
        (May be needed for incorporating prior into loss).
        """
        # Placeholder implementation
        log_probs = {}
        locus_freqs = self.frequency_data.get(locus, {})
        if not locus_freqs:
            return None # No prior info for this locus

        for allele in alleles:
            freq = locus_freqs.get(allele)
            if freq is not None and freq > 0:
                # Use math.log for standard log
                log_probs[allele] = math.log(freq) * self.prior_weight
            else:
                # Handle alleles not in prior (assign low prob? ignore?)
                log_probs[allele] = -float('inf') * self.prior_weight # Example: Assign -inf log prob

        return log_probs


class CovariateManager:
    """
    Manages covariate information, including definitions, encoding strategies,
    and potentially importance assessment.
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
        print("Placeholder: CovariateManager initialized.")

    def get_encoding_strategy(self, covariate_name: str) -> Optional[str]:
        """Returns the specified encoding strategy for a covariate."""
        return self.encoding_strategies.get(covariate_name)

    # Potential future methods:
    # def assess_importance(self, model_results): pass
    # def get_covariate_definitions(self): pass


# Example usage (optional)
if __name__ == '__main__':
    # Create dummy prior file
    dummy_prior_path = "dummy_freqs.csv"
    if not os.path.exists(dummy_prior_path): # Avoid error if run multiple times
        with open(dummy_prior_path, "w") as f:
            f.write("Locus,Allele,Frequency\n")
            f.write("HLA-A,A*01:01,0.25\n")
            f.write("HLA-A,A*02:01,0.15\n")
            f.write("HLA-B,B*07:02,0.10\n")

    try:
        # Test default init
        print("--- Default Config ---")
        default_config = HLAPhasingConfig()
        print(default_config.json(indent=2))

        # Test init with dict override
        print("\n--- Config with Dict Override ---")
        override_dict = {"training": {"learning_rate": 5e-5, "batch_size": 64}}
        dict_config = HLAPhasingConfig(**override_dict)
        print(dict_config.json(indent=2))

        # Test init with file (create dummy file)
        print("\n--- Config with File Load ---")
        dummy_path = "temp_config.json"
        dummy_data = {"model_name": "LoadedFromFile", "data": {"covariate_columns": ["Age"]}}
        with open(dummy_path, 'w') as f:
            json.dump(dummy_data, f)
        file_config = HLAPhasingConfig(config_path=dummy_path)
        print(file_config.json(indent=2))
        if os.path.exists(dummy_path): os.remove(dummy_path) # Clean up dummy file

        # Test init with file and dict override
        print("\n--- Config with File + Override ---")
        with open(dummy_path, 'w') as f:
             json.dump(dummy_data, f)
        override_dict_2 = {"training": {"epochs": 100}}
        file_override_config = HLAPhasingConfig(config_path=dummy_path, **override_dict_2)
        print(file_override_config.json(indent=2))
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
        meta_manager = LocusMetadataManager({"HLA-A": {"res": 4}, "HLA-B": {"res": 2}})
        print(f"Metadata for HLA-A: {meta_manager.get_locus_info('HLA-A')}")
        print(f"Metadata for HLA-C: {meta_manager.get_locus_info('HLA-C')}")
        meta_manager.get_allele_frequency("HLA-A", "A*01:01")

        # Test AlleleFrequencyPrior
        print("\n--- AlleleFrequencyPrior ---")
        # This part will only work if the dummy file exists and parsing logic is added
        try:
            prior = AlleleFrequencyPrior(prior_source=dummy_prior_path)
            print(f"Loaded prior data: {prior.frequency_data}")
            print(f"Freq A*01:01: {prior.get_frequency('HLA-A', 'A*01:01')}")
            print(f"Freq B*07:02: {prior.get_frequency('HLA-B', 'B*07:02')}")
            print(f"Freq C*01:01: {prior.get_frequency('HLA-C', 'C*01:01')}")
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
