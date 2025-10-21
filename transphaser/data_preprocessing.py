import pandas as pd
# Import necessary libraries for encoding
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import torch # Add torch import
from torch.utils.data import Dataset, DataLoader # Added DataLoader import
import logging # Added for HLADataset warning

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GenotypeDataParser:
    """
    Parses genotype data from a pandas DataFrame, handling different formats
    and validating input.
    """
    def __init__(self, locus_columns, covariate_columns):
        """
        Initializes the parser with configuration for locus and covariate columns.

        Args:
            locus_columns (list): List of column names containing HLA genotype data.
            covariate_columns (list): List of column names containing covariate data.
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
                if isinstance(genotype_entry, str):
                    # Split comma-separated string, strip whitespace
                    alleles = [allele.strip() for allele in genotype_entry.split(',')]
                elif isinstance(genotype_entry, list):
                     # Handle list format (to be tested later)
                     alleles = [str(allele).strip() for allele in genotype_entry] # Ensure strings
                else:
                    # Handle other types or missing data if necessary
                    # For now, assume valid string or list input based on description
                    raise TypeError(f"Unsupported genotype format in column {locus_col}, row {index}: {type(genotype_entry)}")

                # Standardize to exactly two alleles
                if len(alleles) == 1:
                    alleles.append(alleles[0]) # Homozygous
                elif len(alleles) == 2:
                    pass # Heterozygous, already correct
                elif len(alleles) == 0:
                     # Handle empty entries if needed, e.g., represent as missing
                     # For now, raise error or use placeholder? Let's raise error.
                     raise ValueError(f"Empty genotype entry in column {locus_col}, row {index}")
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
            "EOS": 3   # End of sequence token
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
    Handles tokenization of genotypes and optionally target haplotypes.
    """
    def __init__(self, genotypes, covariates, tokenizer, loci_order, phased_haplotypes=None, sample_ids=None): # Added sample_ids
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
            phased_haplotypes (list, optional): List of ground truth haplotype strings (e.g., Haplotype1).
                                                Each string contains alleles separated by '_', corresponding
                                                to loci_order. Example: ['A*01:01_B*07:02', ...].
                                                Required for training if model needs target sequences. Defaults to None.
            sample_ids (list, optional): List of original sample identifiers. Defaults to None.
        """
        if len(genotypes) != len(covariates):
            raise ValueError("Number of samples in genotypes and covariates must match.")
        if phased_haplotypes is not None and len(genotypes) != len(phased_haplotypes):
            raise ValueError("Number of samples in genotypes and phased_haplotypes must match.")
        if sample_ids is not None and len(genotypes) != len(sample_ids): # Added check for sample_ids
            raise ValueError("Number of samples in genotypes and sample_ids must match.")


        self.genotypes = genotypes
        self.covariates = covariates
        self.tokenizer = tokenizer
        self.loci_order = loci_order
        self.phased_haplotypes = phased_haplotypes # Store phased data if provided
        self.sample_ids = sample_ids # Store sample IDs if provided

        # Get special token IDs once
        self.bos_token_id = self.tokenizer.special_tokens.get("BOS", 2)
        self.eos_token_id = self.tokenizer.special_tokens.get("EOS", 3)
        self.pad_token_id = self.tokenizer.special_tokens.get("PAD", 0)

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
                  'target_haplotype_tokens' (optional): torch.Tensor of tokenized target haplotype
                                                        alleles (long) including BOS/EOS. Shape (num_loci + 2,).
                                                        Present only if phased_haplotypes were provided.
                  'target_locus_indices' (optional): torch.Tensor of locus indices corresponding to
                                                     target_haplotype_tokens. Shape (num_loci + 2,).
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


        # Convert to tensors
        genotype_tokens_tensor = torch.tensor(genotype_tokens, dtype=torch.long)
        covariates_tensor = torch.tensor(sample_covariates, dtype=torch.float32)

        # Prepare output dictionary
        output_dict = {
            'genotype_tokens': genotype_tokens_tensor,
            'covariates': covariates_tensor,
            # Return sample_id if available, otherwise return index
            'sample_id': self.sample_ids[idx] if self.sample_ids is not None else idx
        }

        # Add target haplotype tokens if phased data is available
        if self.phased_haplotypes is not None:
            haplotype_str = self.phased_haplotypes[idx]
            haplotype_alleles = haplotype_str.split('_') # Assuming '_' separator

            if len(haplotype_alleles) != len(self.loci_order):
                 logging.warning(f"Sample {idx} haplotype length ({len(haplotype_alleles)}) mismatch with loci_order length ({len(self.loci_order)}).")
                 # Handle error or pad? For now, create dummy target
                 target_tokens = [self.bos_token_id] + [self.pad_token_id] * len(self.loci_order) + [self.eos_token_id]
                 target_locus_indices = list(range(len(self.loci_order))) # Placeholder indices
            else:
                target_tokens = [self.bos_token_id]
                target_locus_indices = [-1] # Index for BOS token (or use special value)
                for i, locus in enumerate(self.loci_order):
                    allele = haplotype_alleles[i]
                    target_tokens.append(self.tokenizer.tokenize(locus, allele))
                    target_locus_indices.append(i) # Store the locus index (0 to num_loci-1)
                target_tokens.append(self.eos_token_id)
                target_locus_indices.append(-2) # Index for EOS token (or use special value)


            output_dict['target_haplotype_tokens'] = torch.tensor(target_tokens, dtype=torch.long)
            # Note: target_locus_indices might not be directly usable as embedding indices if using -1/-2
            # Adjust if LocusPositionalEmbedding expects only 0 to num_loci-1
            output_dict['target_locus_indices'] = torch.tensor(target_locus_indices, dtype=torch.long)


        return output_dict


# Example usage (optional, for demonstration)
if __name__ == '__main__':
    print("--- Running Data Preprocessing Example ---")
    # --- GenotypeDataParser Example ---
    data = {
        'SampleID': ['S1', 'S2', 'S3'],
        'HLA-A': ["A*01:01,A*02:01", "A*02:01", ['A*03:01', 'A*11:01']],
        'HLA-B': ["B*07:02,B*08:01", ['B*15:01'], "B*44:02,B*44:03"],
        'Age': [30, 45, 60],
        'Race': ['White', 'Black', 'Asian'],
        'BMI': [22.5, 28.1, 24.0]
    }
    df = pd.DataFrame(data)
    print("\nInput DataFrame:")
    print(df)

    locus_cols = ['HLA-A', 'HLA-B'] # Define the order for parsing
    covariate_cols = ['Age', 'Race', 'BMI'] # Added BMI

    parsed_genotypes = None
    parsed_covariates_df = None
    tokenizer = None
    encoded_covariates_np = None
    hla_dataset = None

    try:
        parser = GenotypeDataParser(locus_columns=locus_cols, covariate_columns=covariate_cols)
        print("\nParser initialized successfully.")
        parsed_genotypes, parsed_covariates_df = parser.parse(df)
        print("Parsing successful.")
        print("Parsed Genotypes (first sample):", parsed_genotypes[0] if parsed_genotypes else "N/A")
        print("Parsed Covariates DF (head):\n", parsed_covariates_df.head() if parsed_covariates_df is not None else "N/A")
    except Exception as e:
        print(f"Error in Parser: {e}")

    # --- AlleleTokenizer Example ---
    if parsed_genotypes:
        try:
            tokenizer = AlleleTokenizer()
            print("\nTokenizer initialized successfully.")
            print(f"Special tokens: {tokenizer.special_tokens}")
            # Build vocab from parsed genotypes
            all_alleles = {}
            for i, locus in enumerate(locus_cols):
                 # Flatten alleles for the current locus across all samples
                 alleles_for_locus = [allele for sample in parsed_genotypes for allele in sample[i]]
                 all_alleles[locus] = set(alleles_for_locus)

            for locus, alleles in all_alleles.items():
                 tokenizer.build_vocabulary(locus, list(alleles))
                 print(f"{locus} Vocab built. Size: {tokenizer.get_vocab_size(locus)}")
            print("Token for A*01:01:", tokenizer.tokenize('HLA-A', 'A*01:01'))
            print("Token for B*15:01:", tokenizer.tokenize('HLA-B', 'B*15:01'))

        except Exception as e:
            print(f"Error initializing or building tokenizer: {e}")
            tokenizer = None # Ensure tokenizer is None if error occurs

    # --- CovariateEncoder Example ---
    if parsed_covariates_df is not None:
        try:
            cat_cols = ['Race']
            num_cols = ['Age', 'BMI']
            cov_encoder = CovariateEncoder(categorical_covariates=cat_cols, numerical_covariates=num_cols)
            print("\nCovariateEncoder initialized successfully.")
            print("Fitting and transforming covariates...")
            encoded_covariates_df = cov_encoder.fit_transform(parsed_covariates_df)
            print("Covariate encoding successful.")
            print("Encoded Covariates DF (head):\n", encoded_covariates_df.head())
            encoded_covariates_np = encoded_covariates_df.to_numpy(dtype=np.float32) # Convert to numpy float32
        except Exception as e:
            print(f"Error in CovariateEncoder: {e}")
            encoded_covariates_np = None
    else:
         print("\nSkipping CovariateEncoder example as parsed_covariates_df not available.")
         encoded_covariates_np = None

    # --- HLADataset Example ---
    if parsed_genotypes is not None and encoded_covariates_np is not None and tokenizer is not None:
        try:
            # Use the same locus order as defined for the parser
            # Pass sample IDs from the original dataframe
            sample_ids_list = df['SampleID'].tolist() if 'SampleID' in df.columns else None

            hla_dataset = HLADataset(
                genotypes=parsed_genotypes,
                covariates=encoded_covariates_np,
                tokenizer=tokenizer,
                loci_order=locus_cols, # Use the order from parser
                sample_ids=sample_ids_list # Pass the list of IDs
            )
            print(f"\nHLADataset created successfully with {len(hla_dataset)} samples.")

            # Example: Get the first item
            if len(hla_dataset) > 0:
                first_item = hla_dataset[0]
                print("\nFirst item from dataset:")
                for key, value in first_item.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}\n    {value}")
                    else:
                        print(f"  {key}: {value} (type: {type(value).__name__})")
            else:
                 print("\nDataset is empty, cannot get first item.")

            # Example: Use DataLoader
            if len(hla_dataset) > 0:
                dataloader = DataLoader(hla_dataset, batch_size=2, shuffle=False) # No shuffle for predictable output
                print("\nDataLoader created. Example batch:")
                for batch in dataloader:
                    print(" Batch Genotype Tokens Shape:", batch['genotype_tokens'].shape)
                    print(" Batch Covariates Shape:", batch['covariates'].shape)
                    print(" Batch Sample IDs:", batch['sample_id']) # Changed from sample_index
                    print(" Batch Genotype Tokens (first sample):\n", batch['genotype_tokens'][0])
                    print(" Batch Covariates (first sample):\n", batch['covariates'][0])
                    break # Show only first batch
            else:
                 print("\nDataset is empty, cannot create DataLoader example.")

        except Exception as e:
            print(f"Error creating or using HLADataset: {e}")
    else:
        print("\nSkipping HLADataset example due to missing prerequisites.")

    print("\n--- Data Preprocessing Example Finished ---")
