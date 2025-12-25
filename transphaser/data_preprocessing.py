import pandas as pd
import random
# Import necessary libraries for encoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import torch # Add torch import
from torch.utils.data import Dataset, DataLoader # Added DataLoader import
import logging # Added for HLADataset warning

# Configure basic logging



class GenotypeDataParser:
    """
    Parses genotype data from a pandas DataFrame, handling different formats
    and validating input.

    Supported Genotype Formats:
        - Slash-separated (recommended): "A*01/A*02", "B*07/B*08"
        - Comma-separated: "A*01:01,A*02:01", "B*07:02,B*08:01"
        - Homozygous (single allele): "A*01" (automatically expanded to "A*01/A*01")
        - List format: ['A*01', 'A*02'] or ['B*07:02', 'B*08:01']

    The parser automatically detects slash vs comma separators and handles
    both heterozygous and homozygous genotypes.
    """
    def __init__(self, locus_columns, covariate_columns):
        """
        Initializes the parser with configuration for locus and covariate columns.

        Args:
            locus_columns (list): List of column names containing HLA genotype data.
                                 Each column should contain genotypes in one of the
                                 supported formats (slash or comma-separated).
            covariate_columns (list): List of column names containing covariate data
                                     (e.g., Population, AgeGroup, etc.).
        """
        if not isinstance(locus_columns, list):
            raise TypeError("locus_columns must be a list")
        if not isinstance(covariate_columns, list):
            raise TypeError("covariate_columns must be a list")

        self.locus_columns = locus_columns
        self.covariate_columns = covariate_columns

    def parse(self, dataframe):
        """
        Parses the input DataFrame to extract and standardize genotype and covariate data.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing genotype and covariate data.

        Returns:
            tuple: A tuple containing processed genotype data (list of lists of lists)
                   and covariate data (pd.DataFrame).
        """
        # Basic validation
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_locus_cols = [col for col in self.locus_columns if col not in dataframe.columns]
        if missing_locus_cols:
            raise ValueError(f"Missing required locus columns: {missing_locus_cols}")

        missing_covariate_cols = [col for col in self.covariate_columns if col not in dataframe.columns]
        if missing_covariate_cols:
            raise ValueError(f"Missing required covariate columns: {missing_covariate_cols}")

        parsed_genotypes = []
        # Iterate over each row (sample) in the DataFrame
        for index, row in dataframe.iterrows():
            sample_genotypes = []
            # Iterate over each specified locus column
            for locus_col in self.locus_columns:
                genotype_entry = row[locus_col]
                alleles = []
                
                if isinstance(genotype_entry, list):
                     # Handle list format
                     alleles = [str(allele).strip() for allele in genotype_entry] # Ensure strings
                elif isinstance(genotype_entry, str):
                    if genotype_entry == "" or genotype_entry.strip() == "":
                         alleles = []
                    elif '/' in genotype_entry:
                        alleles = [allele.strip() for allele in genotype_entry.split('/')]
                    else:
                        alleles = [allele.strip() for allele in genotype_entry.split(',')]
                elif pd.isna(genotype_entry):
                    alleles = []
                else:
                    # Handle other types or missing data if necessary
                    raise TypeError(f"Unsupported genotype format in column {locus_col}, row {index}: {type(genotype_entry)}")
                
                # Filter out empty strings
                alleles = [a for a in alleles if a]

                # Standardize
                if len(alleles) == 0:
                     # Missing data for this locus
                     alleles = ["MISSING", "MISSING"]
                elif len(alleles) == 1:
                    # Homozygous or single missing?
                    if alleles[0].upper() in ["MISSING", "?", "NA", "NAN"]:
                        alleles = ["MISSING", "MISSING"]
                    else:
                        alleles.append(alleles[0]) # Homozygous
                elif len(alleles) == 2:
                    # Check for explicit missing markers
                    alleles = ["MISSING" if a.upper() in ["MISSING", "?", "NA", "NAN"] else a for a in alleles]
                else:
                    raise ValueError(f"Invalid number of alleles ({len(alleles)}) found in column {locus_col}, row {index}. Expected 1 or 2.")

                sample_genotypes.append(sorted(alleles)) # Store sorted alleles for consistency

            parsed_genotypes.append(sample_genotypes)

        # Extract relevant covariate columns
        parsed_covariates = dataframe[self.covariate_columns].copy() # Return a copy

        return parsed_genotypes, parsed_covariates


class AlleleTokenizer:
    """
    Manages vocabularies for HLA alleles across different loci and provides
    tokenization functionality.
    """
    def __init__(self):
        """
        Initializes the tokenizer with separate vocabularies per locus
        and predefined special tokens.
        """
        self.locus_vocabularies = {}  # Dict mapping locus -> {allele: token_id, ...}
        # Reverse mapping: locus -> {token_id: allele, ...}
        self.locus_reverse_vocabularies = {}
        self.special_tokens = {
            "PAD": 0,  # Padding token
            "UNK": 1,  # Unknown token
            "BOS": 2,  # Beginning of sequence token
            "EOS": 3,  # End of sequence token
            "MISSING": 4 # Explicit missing data token
        }
        # Ensure special tokens have unique indices starting from 0
        if len(self.special_tokens) != len(set(self.special_tokens.values())):
            raise ValueError("Special token indices must be unique.")
        if min(self.special_tokens.values()) != 0:
             raise ValueError("Special token indices must start from 0.")
         # Next available ID for non-special tokens - this should be locus-specific
         # self.next_token_id = len(self.special_tokens) # Removed global counter

        # Add properties for convenient access to special token IDs
        self.pad_token_id = self.special_tokens["PAD"]
        self.unk_token_id = self.special_tokens["UNK"]
        self.bos_token_id = self.special_tokens["BOS"]
        self.eos_token_id = self.special_tokens["EOS"]
        self.missing_token_id = self.special_tokens["MISSING"]

    def build_vocabulary(self, locus, alleles):
        """
        Builds or updates the vocabulary for a specific locus.
        Assigns unique integer IDs to alleles, starting after special tokens.

        Args:
            locus (str): The name of the locus (e.g., 'HLA-A').
            alleles (list): A list of allele strings for this locus.
        """
        if locus not in self.locus_vocabularies:
            # Initialize vocabulary for this locus with special tokens
            self.locus_vocabularies[locus] = self.special_tokens.copy()
            self.locus_reverse_vocabularies[locus] = {
                v: k for k, v in self.special_tokens.items()
            }
            # Start allele IDs after special tokens
            next_id = len(self.special_tokens)
        else:
            # Find the next available ID if vocabulary already exists
            next_id = max(self.locus_vocabularies[locus].values()) + 1

        for allele in alleles:
            if allele not in self.locus_vocabularies[locus]:
                self.locus_vocabularies[locus][allele] = next_id
                self.locus_reverse_vocabularies[locus][next_id] = allele
                next_id += 1 # Corrected indentation

    def build_vocabulary_from_dataframe(self, df, locus_columns):
        """
        Builds vocabularies for multiple loci directly from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing genotype data.
            locus_columns (list): A list of column names for the loci.
        """
        for locus in locus_columns:
            locus_alleles = set()
            for genotype_str in df[locus].dropna():
                if isinstance(genotype_str, str): # Handle mixed types
                    alleles = genotype_str.replace(',', '/').split('/')
                    locus_alleles.update(a for a in alleles if a and a not in ["UNK", "<UNK>", "MISSING"])
            self.build_vocabulary(locus, list(locus_alleles))

    def tokenize(self, locus, allele):
        """
        Converts an allele string to its token ID for a specific locus.

        Args:
            locus (str): The locus name.
            allele (str): The allele string to tokenize.

        Returns:
            int: The token ID for the allele. Returns the UNK token ID if the
                 allele is not in the vocabulary for the locus or if the locus
                 itself is unknown.
        """
        if locus not in self.locus_vocabularies:
            # Option 1: Raise error for unknown locus
            raise KeyError(f"No vocabulary built for locus: {locus}")
            # Option 2: Return UNK for unknown locus (might hide errors)
            # return self.special_tokens['UNK']

        if allele == "MISSING":
            return self.missing_token_id

        # Return the token ID if allele is known, otherwise return UNK token ID
        return self.locus_vocabularies[locus].get(allele, self.special_tokens['UNK'])

    def detokenize(self, locus, token_id):
        """
        Converts a token ID back to its allele string for a specific locus.

        Args:
            locus (str): The locus name.
            token_id (int): The token ID to detokenize.

        Returns:
            str: The allele string corresponding to the token ID. Returns the
                 string 'UNK' if the token ID is not found in the vocabulary
                 for the locus or if the locus itself is unknown.
        """
        if locus not in self.locus_reverse_vocabularies:
            # Option 1: Raise error for unknown locus
            raise KeyError(f"No vocabulary built for locus: {locus}")
            # Option 2: Return 'UNK' string (consistent with test)
            # return 'UNK'

        # Return the allele string if token ID is known, otherwise return 'UNK' string
        return self.locus_reverse_vocabularies[locus].get(token_id, 'UNK')


    def get_vocab_size(self, locus):
        """
        Returns the vocabulary size for a specific locus (including special tokens).

        Args:
            locus (str): The locus name.

        Returns:
            int: The total number of tokens (alleles + special tokens) for the locus.

        Raises:
            KeyError: If the locus is not found in the vocabularies.
        """
        if locus not in self.locus_vocabularies:
            raise KeyError(f"No vocabulary built for locus: {locus}")
        return len(self.locus_vocabularies[locus])


class CovariateEncoder:
    """
    Encodes categorical and numerical covariates for use in the model.
    Handles fitting encoders and transforming data.
    """
    def __init__(self, categorical_covariates=None, numerical_covariates=None):
        """
        Initializes the encoder with lists of categorical and numerical covariate names.

        Args:
            categorical_covariates (list, optional): List of column names for categorical features. Defaults to None.
            numerical_covariates (list, optional): List of column names for numerical features. Defaults to None.
        """
        self.categorical_covariates = categorical_covariates if categorical_covariates else []
        self.numerical_covariates = numerical_covariates if numerical_covariates else []
        self.encoders = {}  # Store fitted encoders (e.g., OneHotEncoder, StandardScaler)
        self._fitted = False # Track if encoders have been fitted

    def fit(self, dataframe):
        """
        Fits encoders to the provided dataframe based on the specified covariate columns.

        Args:
            dataframe (pd.DataFrame): The dataframe containing covariate data to fit encoders on.
        """
        # Basic validation
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_cat_cols = [col for col in self.categorical_covariates if col not in dataframe.columns]
        if missing_cat_cols:
            raise ValueError(f"Missing required categorical covariate columns for fitting: {missing_cat_cols}")

        missing_num_cols = [col for col in self.numerical_covariates if col not in dataframe.columns]
        if missing_num_cols:
            raise ValueError(f"Missing required numerical covariate columns for fitting: {missing_num_cols}")

        self.encoders = {} # Reset encoders before fitting

        # --- Fit Categorical Encoders ---
        for col in self.categorical_covariates:
            # Using sparse_output=False for easier handling in pandas DataFrame later
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            # Fit requires a 2D array, hence dataframe[[col]]
            encoder.fit(dataframe[[col]])
            self.encoders[col] = encoder

        # --- Fit Numerical Encoders ---
        for col in self.numerical_covariates:
            encoder = StandardScaler()
            # Fit requires a 2D array
            encoder.fit(dataframe[[col]])
            self.encoders[col] = encoder

        self._fitted = True # Mark as fitted

    def transform(self, dataframe):
        """
        Transforms the covariate data in the dataframe using the fitted encoders.

        Args:
            dataframe (pd.DataFrame): The dataframe containing covariate data to transform.

        Returns:
            pd.DataFrame: A dataframe with encoded covariates.
        """
        if not self._fitted:
            raise RuntimeError("Encoders must be fitted before transforming data. Call fit() first.")

        # Basic validation
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_cat_cols = [col for col in self.categorical_covariates if col not in dataframe.columns]
        if missing_cat_cols:
            raise ValueError(f"Missing required categorical covariate columns for transforming: {missing_cat_cols}")

        missing_num_cols = [col for col in self.numerical_covariates if col not in dataframe.columns]
        if missing_num_cols:
            raise ValueError(f"Missing required numerical covariate columns for transforming: {missing_num_cols}")

        encoded_parts = []

        # --- Transform Categorical ---
        for col in self.categorical_covariates:
            if col in self.encoders:
                encoder = self.encoders[col]
                encoded_data = encoder.transform(dataframe[[col]])
                try:
                    new_cols = encoder.get_feature_names_out([col]) # Use get_feature_names_out
                except AttributeError: # Fallback for older sklearn
                    new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]

                encoded_df = pd.DataFrame(encoded_data, columns=new_cols, index=dataframe.index)
                encoded_parts.append(encoded_df)

        # --- Transform Numerical ---
        for col in self.numerical_covariates:
             if col in self.encoders:
                encoder = self.encoders[col]
                encoded_data = encoder.transform(dataframe[[col]])
                encoded_df = pd.DataFrame(encoded_data, columns=[col], index=dataframe.index)
                encoded_parts.append(encoded_df)

        # Combine encoded parts
        if not encoded_parts:
            return pd.DataFrame(index=dataframe.index)
        else:
            final_encoded_df = pd.concat(encoded_parts, axis=1)
            return final_encoded_df


    def fit_transform(self, dataframe):
        """
        Fits the encoders and then transforms the data in one step.

        Args:
            dataframe (pd.DataFrame): The dataframe containing covariate data.

        Returns:
            pd.DataFrame: A dataframe with encoded covariates.
        """
        self.fit(dataframe)
        return self.transform(dataframe)


class HLADataset(Dataset):
    """
    PyTorch Dataset for HLA phasing data.
    
    During training, phased haplotypes are NOT included in batches.
    During evaluation, phased haplotypes are included for metrics.
    """
    def __init__(self, genotypes, covariates, tokenizer, loci_order, 
                 phased_haplotypes=None, sample_ids=None, mask_prob=0.0, mode='train'):
        """
        Initializes the dataset.

        Args:
            genotypes (list): List of parsed genotypes. Each element is a list
                              representing a sample, containing sub-lists for each locus,
                              which in turn contain two allele strings (sorted).
                              Example: [[['A*01:01', 'A*02:01'], ['B*07:02', 'B*07:02']], ...]
            covariates (np.ndarray): Numpy array of encoded covariates, shape (num_samples, num_features).
            tokenizer (AlleleTokenizer): An initialized and fitted AlleleTokenizer instance.
            loci_order (list): List of locus names corresponding to the order in parsed_genotypes.
            phased_haplotypes (list, optional): List of ground truth haplotype PAIRS.
                                                Each element is a tuple/list of two haplotype strings: (H1, H2).
                                                Each haplotype string contains alleles separated by '_', corresponding
                                                to loci_order. Example: [('A*01_B*07', 'A*02_B*08'), ...].
                                                ONLY used in 'eval' mode for metrics. Defaults to None.
            sample_ids (list, optional): List of original sample identifiers. Defaults to None.
            mask_prob (float): Probability of randomly masking a genotype allele with the MISSING token.
                               Used for training robust imputation. Defaults to 0.0 (no masking).
            mode (str): 'train', 'eval' or 'predict'.
        """
        if len(genotypes) != len(covariates):
            raise ValueError(f"Number of samples in genotypes ({len(genotypes)}) and covariates ({len(covariates)}) must match.")
        if phased_haplotypes is not None and len(genotypes) != len(phased_haplotypes):
            raise ValueError("Number of samples in genotypes and phased_haplotypes must match.")
        if sample_ids is not None and len(genotypes) != len(sample_ids):
            raise ValueError("Number of samples in genotypes and sample_ids must match.")
        
        valid_modes = ['train', 'eval', 'predict']
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")


        self.genotypes = genotypes
        self.covariates = covariates
        self.tokenizer = tokenizer
        self.loci_order = loci_order
        self.phased_haplotypes = phased_haplotypes  # Store phased data (only used in eval mode)
        self.sample_ids = sample_ids
        self.mask_prob = mask_prob
        self.mode = mode

        # Log mode for clarity
        if mode == 'train':
            logging.info(f"HLADataset in TRAINING mode - phased haplotypes will NOT be included in batches")
        elif mode == 'predict':
            logging.info(f"HLADataset in PREDICT mode - phased haplotypes disregarded")
        else:
            logging.info(f"HLADataset in EVALUATION mode - phased haplotypes will be included for metrics")

        # Get special token IDs once
        self.bos_token_id = self.tokenizer.special_tokens.get("BOS", 2)
        self.eos_token_id = self.tokenizer.special_tokens.get("EOS", 3)
        self.pad_token_id = self.tokenizer.special_tokens.get("PAD", 0)
        self.missing_token_id = self.tokenizer.special_tokens.get("MISSING", 4)


    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.genotypes)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                  'genotype_tokens': torch.Tensor of tokenized genotype alleles (long). Shape (num_loci * 2,).
                  'covariates': torch.Tensor of encoded covariates (float). Shape (num_features,).
                  'sample_id': The original identifier of the sample (e.g., string from input data).
                  'target_h1_tokens' (optional): torch.Tensor of tokenized H1 haplotype
                                                 alleles (long) including BOS/EOS. Shape (num_loci + 2,).
                  'target_h2_tokens' (optional): torch.Tensor of tokenized H2 haplotype
                                                 alleles (long) including BOS/EOS. Shape (num_loci + 2,).
                                                 Present only if phased_haplotypes were provided.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()


        sample_genotype = self.genotypes[idx]
        sample_covariates = self.covariates[idx]

        # Tokenize genotype based on the specified loci_order
        # Validate genotype length matches expected number of loci
        if len(sample_genotype) != len(self.loci_order):
            raise ValueError(
                f"Sample {idx}: Genotype has {len(sample_genotype)} loci but "
                f"expected {len(self.loci_order)} (loci_order). "
                f"This indicates a data preprocessing error. Check that all samples "
                f"have genotypes for all required loci in the correct order."
            )

        genotype_tokens = []
        for i, locus in enumerate(self.loci_order):
            locus_alleles = sample_genotype[i]
            token1 = self.tokenizer.tokenize(locus, locus_alleles[0])
            token2 = self.tokenizer.tokenize(locus, locus_alleles[1])
            genotype_tokens.extend([token1, token2])

        # Apply random masking for imputation training
        if self.mask_prob > 0:
            for i in range(len(genotype_tokens)):
                if genotype_tokens[i] != self.pad_token_id:
                    if random.random() < self.mask_prob:
                        genotype_tokens[i] = self.missing_token_id


        # Convert to tensors
        genotype_tokens_tensor = torch.tensor(genotype_tokens, dtype=torch.long)
        covariates_tensor = torch.tensor(sample_covariates, dtype=torch.float32)

        # Prepare output dictionary (ALWAYS include genotype and covariates)
        output_dict = {
            'genotype_tokens': genotype_tokens_tensor,
            'covariates': covariates_tensor,
            # Return sample_id if available, otherwise return index
            'sample_id': self.sample_ids[idx] if self.sample_ids is not None else idx
        }

        # Add target haplotype tokens ONLY in evaluation mode (for computing metrics)
        # In training mode, NO phased targets are included!
        if self.mode == 'eval' and self.phased_haplotypes is not None:
            haplotype_pair = self.phased_haplotypes[idx]  # Should be (H1_string, H2_string)
            
            # Handle both tuple/list and backward compatibility with single string
            if isinstance(haplotype_pair, (tuple, list)) and len(haplotype_pair) == 2:
                h1_str, h2_str = haplotype_pair
            else:
                # Backward compatibility: if single string, treat as H1 and derive H2 from genotype
                logging.warning(f"Sample {idx}: phased_haplotypes is not a pair. Expected (H1, H2) tuple.")
                h1_str = haplotype_pair
                h2_str = None  # Will skip H2 tokenization
            
            # Tokenize H1
            h1_alleles = h1_str.split('_')
            if len(h1_alleles) != len(self.loci_order):
                raise ValueError(
                    f"Sample {idx} H1 haplotype length ({len(h1_alleles)}) "
                    f"mismatch with loci_order length ({len(self.loci_order)})."
                )
            
            h1_tokens = [self.bos_token_id]
            for i, locus in enumerate(self.loci_order):
                allele = h1_alleles[i]
                h1_tokens.append(self.tokenizer.tokenize(locus, allele))
            h1_tokens.append(self.eos_token_id)
            
            output_dict['target_h1_tokens'] = torch.tensor(h1_tokens, dtype=torch.long)
            
            # Tokenize H2 if available
            if h2_str is not None:
                h2_alleles = h2_str.split('_')
                if len(h2_alleles) != len(self.loci_order):
                    raise ValueError(
                        f"Sample {idx} H2 haplotype length ({len(h2_alleles)}) "
                        f"mismatch with loci_order length ({len(self.loci_order)})."
                    )
                
                h2_tokens = [self.bos_token_id]
                for i, locus in enumerate(self.loci_order):
                    allele = h2_alleles[i]
                    h2_tokens.append(self.tokenizer.tokenize(locus, allele))
                h2_tokens.append(self.eos_token_id)
                
                output_dict['target_h2_tokens'] = torch.tensor(h2_tokens, dtype=torch.long)

        return output_dict
