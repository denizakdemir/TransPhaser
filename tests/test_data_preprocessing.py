import unittest
import pandas as pd
import numpy as np # Add numpy import
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Add sklearn imports
from pandas.testing import assert_frame_equal # For comparing dataframes
import torch # Add torch import
from torch.utils.data import Dataset, DataLoader # Add Dataset and DataLoader import

# Placeholder for the classes we are about to create
from src.data_preprocessing import GenotypeDataParser, AlleleTokenizer, CovariateEncoder, HLADataset # Added HLADataset

class TestGenotypeDataParser(unittest.TestCase):

    def test_initialization(self):
        """Test that GenotypeDataParser initializes correctly."""
        locus_columns = ['HLA-A', 'HLA-B']
        covariate_columns = ['Age', 'Race']
        # This will fail until we create the class
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        self.assertEqual(parser.locus_columns, locus_columns)
        self.assertEqual(parser.covariate_columns, covariate_columns)

    def test_parse_validates_dataframe(self):
        """Test that parse method raises TypeError for non-DataFrame input."""
        locus_columns = ['HLA-A']
        covariate_columns = ['Age']
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        with self.assertRaisesRegex(TypeError, "Input must be a pandas DataFrame"):
            parser.parse("not a dataframe") # Pass invalid input

    def test_parse_validates_missing_locus_columns(self):
        """Test that parse method raises ValueError for missing locus columns."""
        locus_columns = ['HLA-A', 'HLA-B'] # Expects A and B
        covariate_columns = ['Age']
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        data = {'HLA-A': ['A*01:01'], 'Age': [30]} # Only provides A
        df = pd.DataFrame(data)
        with self.assertRaisesRegex(ValueError, "Missing required locus columns: \\['HLA-B'\\]"):
             parser.parse(df)

    def test_parse_validates_missing_covariate_columns(self):
        """Test that parse method raises ValueError for missing covariate columns."""
        locus_columns = ['HLA-A']
        covariate_columns = ['Age', 'Race'] # Expects Age and Race
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        data = {'HLA-A': ['A*01:01'], 'Age': [30]} # Only provides Age
        df = pd.DataFrame(data)
        with self.assertRaisesRegex(ValueError, "Missing required covariate columns: \\['Race'\\]"):
            parser.parse(df)

    def test_parse_genotype_string_format(self):
        """Test parsing genotype data provided as comma-separated strings."""
        locus_columns = ['HLA-A', 'HLA-B']
        covariate_columns = ['Age']
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        data = {
            'HLA-A': ["A*01:01,A*02:01", "A*03:01"], # Heterozygous and Homozygous
            'HLA-B': ["B*07:02", "B*08:01,B*15:01"], # Homozygous and Heterozygous
            'Age': [30, 45]
        }
        df = pd.DataFrame(data)
        parsed_genotypes, parsed_covariates = parser.parse(df)

        expected_genotypes = [
            [['A*01:01', 'A*02:01'], ['B*07:02', 'B*07:02']], # Sample 1
            [['A*03:01', 'A*03:01'], ['B*08:01', 'B*15:01']]  # Sample 2
        ]
        self.assertEqual(parsed_genotypes, expected_genotypes)
        # We are not testing covariate parsing yet
        # self.assertIsNotNone(parsed_covariates)

    def test_parse_genotype_list_format(self):
        """Test parsing genotype data provided as lists."""
        locus_columns = ['HLA-A', 'HLA-B']
        covariate_columns = ['Age']
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        data = {
            'HLA-A': [['A*01:01', 'A*02:01'], ['A*03:01']], # Heterozygous and Homozygous list
            'HLA-B': [['B*07:02'], ['B*08:01', 'B*15:01']], # Homozygous and Heterozygous list
            'Age': [30, 45]
        }
        df = pd.DataFrame(data)
        parsed_genotypes, parsed_covariates = parser.parse(df)

        expected_genotypes = [
            [['A*01:01', 'A*02:01'], ['B*07:02', 'B*07:02']], # Sample 1
            [['A*03:01', 'A*03:01'], ['B*08:01', 'B*15:01']]  # Sample 2
        ]
        # Ensure alleles within a locus are sorted for consistent comparison
        parsed_genotypes_sorted = [[sorted(locus) for locus in sample] for sample in parsed_genotypes]
        expected_genotypes_sorted = [[sorted(locus) for locus in sample] for sample in expected_genotypes]

        self.assertEqual(parsed_genotypes_sorted, expected_genotypes_sorted)

    def test_parse_invalid_allele_count(self):
        """Test that parse raises ValueError for invalid number of alleles."""
        locus_columns = ['HLA-A']
        covariate_columns = ['Age']
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        # String format with 3 alleles
        data_str = {'HLA-A': ["A*01:01,A*02:01,A*03:01"], 'Age': [30]}
        df_str = pd.DataFrame(data_str)
        with self.assertRaisesRegex(ValueError, "Invalid number of alleles \\(3\\)"):
            parser.parse(df_str)
        # List format with 3 alleles
        data_list = {'HLA-A': [['A*01:01', 'A*02:01', 'A*03:01']], 'Age': [30]}
        df_list = pd.DataFrame(data_list)
        with self.assertRaisesRegex(ValueError, "Invalid number of alleles \\(3\\)"):
            parser.parse(df_list)
        # Empty string format
        data_empty_str = {'HLA-A': [""], 'Age': [30]}
        df_empty_str = pd.DataFrame(data_empty_str)
        # The current code splits "" into [''], len 1, so it becomes homozygous ['','']
        # Let's refine the test/code later if empty strings need specific handling.
        # For now, we test the explicit > 2 case.

        # Empty list format
        data_empty_list = {'HLA-A': [[]], 'Age': [30]}
        df_empty_list = pd.DataFrame(data_empty_list)
        with self.assertRaisesRegex(ValueError, "Empty genotype entry"):
             parser.parse(df_empty_list)

    def test_parse_unsupported_genotype_type(self):
        """Test that parse raises TypeError for unsupported genotype data types."""
        locus_columns = ['HLA-A']
        covariate_columns = ['Age']
        parser = GenotypeDataParser(locus_columns=locus_columns, covariate_columns=covariate_columns)
        data = {'HLA-A': [123], 'Age': [30]} # Integer instead of string/list
        df = pd.DataFrame(data)
        with self.assertRaisesRegex(TypeError, "Unsupported genotype format"):
            parser.parse(df)
        data_float = {'HLA-A': [1.23], 'Age': [30]} # Float instead of string/list
        df_float = pd.DataFrame(data_float)
        with self.assertRaisesRegex(TypeError, "Unsupported genotype format"):
            parser.parse(df_float)


class TestAlleleTokenizer(unittest.TestCase):

    def test_initialization(self):
        """Test that AlleleTokenizer initializes correctly."""
        tokenizer = AlleleTokenizer()
        self.assertEqual(tokenizer.locus_vocabularies, {})
        expected_special_tokens = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
        self.assertEqual(tokenizer.special_tokens, expected_special_tokens)
        # Check that special tokens have unique indices starting from 0
        self.assertEqual(len(tokenizer.special_tokens), len(set(tokenizer.special_tokens.values())))
        self.assertEqual(min(tokenizer.special_tokens.values()), 0)

    def test_build_vocabulary(self):
        """Test building vocabulary for a specific locus."""
        tokenizer = AlleleTokenizer()
        locus = 'HLA-A'
        alleles1 = ['A*01:01', 'A*02:01']
        tokenizer.build_vocabulary(locus, alleles1)

        # Check if locus vocabulary is created
        self.assertIn(locus, tokenizer.locus_vocabularies)
        self.assertIn(locus, tokenizer.locus_reverse_vocabularies)

        # Check initial vocabulary content (special tokens + alleles1)
        expected_vocab_size1 = len(tokenizer.special_tokens) + len(alleles1)
        self.assertEqual(len(tokenizer.locus_vocabularies[locus]), expected_vocab_size1)
        self.assertEqual(len(tokenizer.locus_reverse_vocabularies[locus]), expected_vocab_size1)

        # Check specific token assignments
        start_id = len(tokenizer.special_tokens)
        self.assertEqual(tokenizer.locus_vocabularies[locus]['A*01:01'], start_id)
        self.assertEqual(tokenizer.locus_vocabularies[locus]['A*02:01'], start_id + 1)
        self.assertEqual(tokenizer.locus_reverse_vocabularies[locus][start_id], 'A*01:01')
        self.assertEqual(tokenizer.locus_reverse_vocabularies[locus][start_id + 1], 'A*02:01')

        # Check special tokens are present
        for token, token_id in tokenizer.special_tokens.items():
            self.assertEqual(tokenizer.locus_vocabularies[locus][token], token_id)
            self.assertEqual(tokenizer.locus_reverse_vocabularies[locus][token_id], token)

        # Test adding more alleles (including duplicates)
        alleles2 = ['A*02:01', 'A*03:01', 'A*11:01']
        tokenizer.build_vocabulary(locus, alleles2)

        expected_vocab_size2 = expected_vocab_size1 + 2 # Only A*03:01 and A*11:01 are new
        self.assertEqual(len(tokenizer.locus_vocabularies[locus]), expected_vocab_size2)
        self.assertEqual(len(tokenizer.locus_reverse_vocabularies[locus]), expected_vocab_size2)

        # Check new token assignments
        self.assertEqual(tokenizer.locus_vocabularies[locus]['A*03:01'], start_id + 2)
        self.assertEqual(tokenizer.locus_vocabularies[locus]['A*11:01'], start_id + 3)
        self.assertEqual(tokenizer.locus_reverse_vocabularies[locus][start_id + 2], 'A*03:01')
        self.assertEqual(tokenizer.locus_reverse_vocabularies[locus][start_id + 3], 'A*11:01')

        # Check that existing tokens were not reassigned
        self.assertEqual(tokenizer.locus_vocabularies[locus]['A*01:01'], start_id)
        self.assertEqual(tokenizer.locus_vocabularies[locus]['A*02:01'], start_id + 1)

    def test_tokenize(self):
        """Test tokenizing alleles for a specific locus."""
        tokenizer = AlleleTokenizer()
        locus = 'HLA-A'
        alleles = ['A*01:01', 'A*02:01']
        tokenizer.build_vocabulary(locus, alleles)

        start_id = len(tokenizer.special_tokens)
        unk_id = tokenizer.special_tokens['UNK']

        # Test known alleles
        self.assertEqual(tokenizer.tokenize(locus, 'A*01:01'), start_id)
        self.assertEqual(tokenizer.tokenize(locus, 'A*02:01'), start_id + 1)

        # Test unknown allele
        self.assertEqual(tokenizer.tokenize(locus, 'A*99:99'), unk_id)

        # Test special tokens
        self.assertEqual(tokenizer.tokenize(locus, 'PAD'), tokenizer.special_tokens['PAD'])
        self.assertEqual(tokenizer.tokenize(locus, 'UNK'), tokenizer.special_tokens['UNK'])
        self.assertEqual(tokenizer.tokenize(locus, 'BOS'), tokenizer.special_tokens['BOS'])
        self.assertEqual(tokenizer.tokenize(locus, 'EOS'), tokenizer.special_tokens['EOS'])

        # Test unknown locus
        with self.assertRaises(KeyError):
            tokenizer.tokenize('HLA-B', 'B*07:02')

    def test_detokenize(self):
        """Test detokenizing token IDs back to alleles."""
        tokenizer = AlleleTokenizer()
        locus = 'HLA-A'
        alleles = ['A*01:01', 'A*02:01']
        tokenizer.build_vocabulary(locus, alleles)

        start_id = len(tokenizer.special_tokens)
        unk_token_str = 'UNK' # String representation for unknown token ID

        # Test known allele IDs
        self.assertEqual(tokenizer.detokenize(locus, start_id), 'A*01:01')
        self.assertEqual(tokenizer.detokenize(locus, start_id + 1), 'A*02:01')

        # Test special token IDs
        self.assertEqual(tokenizer.detokenize(locus, tokenizer.special_tokens['PAD']), 'PAD')
        self.assertEqual(tokenizer.detokenize(locus, tokenizer.special_tokens['UNK']), 'UNK')
        self.assertEqual(tokenizer.detokenize(locus, tokenizer.special_tokens['BOS']), 'BOS')
        self.assertEqual(tokenizer.detokenize(locus, tokenizer.special_tokens['EOS']), 'EOS')

        # Test unknown token ID within the locus vocab
        unknown_id = 999
        self.assertEqual(tokenizer.detokenize(locus, unknown_id), unk_token_str) # Expect UNK string

        # Test unknown locus
        with self.assertRaises(KeyError):
            tokenizer.detokenize('HLA-B', start_id)

    def test_get_vocab_size(self):
        """Test getting the vocabulary size for a locus."""
        tokenizer = AlleleTokenizer()
        locus = 'HLA-A'
        alleles = ['A*01:01', 'A*02:01', 'A*03:01']
        tokenizer.build_vocabulary(locus, alleles)

        expected_size = len(tokenizer.special_tokens) + len(alleles)
        self.assertEqual(tokenizer.get_vocab_size(locus), expected_size)

        # Test unknown locus
        with self.assertRaises(KeyError):
            tokenizer.get_vocab_size('HLA-B')


class TestCovariateEncoder(unittest.TestCase):

    def test_initialization(self):
        """Test that CovariateEncoder initializes correctly."""
        categorical_covariates = ['Race', 'Ethnicity']
        numerical_covariates = ['Age', 'BMI']
        encoder = CovariateEncoder(
            categorical_covariates=categorical_covariates,
            numerical_covariates=numerical_covariates
        )
        self.assertEqual(encoder.categorical_covariates, categorical_covariates)
        self.assertEqual(encoder.numerical_covariates, numerical_covariates)
        self.assertEqual(encoder.encoders, {}) # Encoders dict should be empty initially
        self.assertFalse(encoder._fitted) # Should not be fitted initially

    def test_fit_validates_dataframe(self):
        """Test that fit method raises TypeError for non-DataFrame input."""
        encoder = CovariateEncoder(categorical_covariates=['Race'], numerical_covariates=['Age'])
        with self.assertRaisesRegex(TypeError, "Input must be a pandas DataFrame"):
            encoder.fit("not a dataframe")

    def test_fit_validates_missing_columns(self):
        """Test fit raises ValueError if required covariate columns are missing."""
        # Missing categorical
        encoder_cat = CovariateEncoder(categorical_covariates=['Race', 'Ethnicity'], numerical_covariates=['Age'])
        data_cat = {'Race': ['White'], 'Age': [30]} # Missing Ethnicity
        df_cat = pd.DataFrame(data_cat)
        with self.assertRaisesRegex(ValueError, "Missing required categorical covariate columns.*\\['Ethnicity'\\]"):
            encoder_cat.fit(df_cat)

        # Missing numerical
        encoder_num = CovariateEncoder(categorical_covariates=['Race'], numerical_covariates=['Age', 'BMI'])
        data_num = {'Race': ['White'], 'Age': [30]} # Missing BMI
        df_num = pd.DataFrame(data_num)
        with self.assertRaisesRegex(ValueError, "Missing required numerical covariate columns.*\\['BMI'\\]"):
            encoder_num.fit(df_num)

    def test_fit_sets_fitted_flag(self):
        """Test that fit method sets the _fitted flag to True."""
        encoder = CovariateEncoder(categorical_covariates=['Race'], numerical_covariates=['Age'])
        data = {'Race': ['White', 'Black'], 'Age': [30, 40]}
        df = pd.DataFrame(data)
        self.assertFalse(encoder._fitted) # Check initial state
        encoder.fit(df)
        self.assertTrue(encoder._fitted) # Check state after fitting

    def test_transform_before_fit_raises_error(self):
        """Test transform raises RuntimeError if called before fit."""
        encoder = CovariateEncoder(categorical_covariates=['Race'], numerical_covariates=['Age'])
        data = {'Race': ['White'], 'Age': [30]}
        df = pd.DataFrame(data)
        with self.assertRaisesRegex(RuntimeError, "Encoders must be fitted before transforming data"):
            encoder.transform(df)

    def test_transform_validates_dataframe(self):
        """Test that transform method raises TypeError for non-DataFrame input."""
        encoder = CovariateEncoder(categorical_covariates=['Race'], numerical_covariates=['Age'])
        # Fit the encoder first
        fit_data = {'Race': ['White'], 'Age': [30]}
        fit_df = pd.DataFrame(fit_data)
        encoder.fit(fit_df)
        # Now test transform with invalid input
        with self.assertRaisesRegex(TypeError, "Input must be a pandas DataFrame"):
            encoder.transform("not a dataframe")

    def test_transform_validates_missing_columns(self):
        """Test transform raises ValueError if required covariate columns are missing."""
        # Fit with complete data
        fit_data = {'Race': ['White', 'Black'], 'Age': [30, 40], 'BMI': [25, 28]}
        fit_df = pd.DataFrame(fit_data)

        # Test missing categorical
        encoder_cat = CovariateEncoder(categorical_covariates=['Race', 'Ethnicity'], numerical_covariates=['Age'])
        fit_data_cat = {'Race': ['White'], 'Ethnicity': ['X'], 'Age': [30]}
        encoder_cat.fit(pd.DataFrame(fit_data_cat))
        transform_data_cat = {'Race': ['White'], 'Age': [30]} # Missing Ethnicity
        transform_df_cat = pd.DataFrame(transform_data_cat)
        with self.assertRaisesRegex(ValueError, "Missing required categorical covariate columns.*\\['Ethnicity'\\]"):
            encoder_cat.transform(transform_df_cat)

        # Test missing numerical
        encoder_num = CovariateEncoder(categorical_covariates=['Race'], numerical_covariates=['Age', 'BMI'])
        fit_data_num = {'Race': ['White'], 'Age': [30], 'BMI': [25]}
        encoder_num.fit(pd.DataFrame(fit_data_num))
        transform_data_num = {'Race': ['White'], 'Age': [30]} # Missing BMI
        transform_df_num = pd.DataFrame(transform_data_num)
        with self.assertRaisesRegex(ValueError, "Missing required numerical covariate columns.*\\['BMI'\\]"):
            encoder_num.transform(transform_df_num)

    def test_fit_transform_encoding(self):
        """Test fit_transform applies OneHotEncoder and StandardScaler correctly."""
        cat_cols = ['Race']
        num_cols = ['Age']
        encoder = CovariateEncoder(categorical_covariates=cat_cols, numerical_covariates=num_cols)
        data = {'Race': ['White', 'Black', 'White'], 'Age': [30, 40, 50]}
        df = pd.DataFrame(data)

        # Expected output after OneHotEncoding 'Race' and StandardScaler 'Age'
        # OneHotEncoder: White -> [0.0, 1.0], Black -> [1.0, 0.0] (order depends on fit)
        # StandardScaler: Age -> [(30-40)/std, (40-40)/std, (50-40)/std] where std=sqrt(((30-40)^2+(40-40)^2+(50-40)^2)/3) = sqrt(200/3) = 8.165
        # Scaled Age: [-1.2247, 0.0, 1.2247]

        # We need to know the category order from OneHotEncoder to build the expected df
        # Let's fit a temporary encoder to get the order
        temp_ohe = OneHotEncoder(sparse_output=False)
        temp_ohe.fit(df[['Race']])
        cat_order = temp_ohe.categories_[0] # e.g., ['Black', 'White']

        # Build expected DataFrame based on category order
        expected_data = {}
        if cat_order[0] == 'Black': # Black is first column
             expected_data['Race_Black'] = [0.0, 1.0, 0.0]
             expected_data['Race_White'] = [1.0, 0.0, 1.0]
        else: # White is first column
             expected_data['Race_White'] = [1.0, 0.0, 1.0]
             expected_data['Race_Black'] = [0.0, 1.0, 0.0]

        # Calculate scaled Age
        age_mean = df['Age'].mean()
        age_std = df['Age'].std(ddof=0) # Use population std dev like StandardScaler
        expected_data['Age'] = (df['Age'] - age_mean) / age_std

        expected_df = pd.DataFrame(expected_data)

        # Run the actual fit_transform
        transformed_df = encoder.fit_transform(df)

        # Check that the encoders were stored
        self.assertIn('Race', encoder.encoders)
        self.assertIsInstance(encoder.encoders['Race'], OneHotEncoder)
        self.assertIn('Age', encoder.encoders)
        self.assertIsInstance(encoder.encoders['Age'], StandardScaler)

        # Compare the transformed DataFrame with the expected one
        # Need to sort columns as concat order might vary
        assert_frame_equal(
            transformed_df.sort_index(axis=1),
            expected_df.sort_index(axis=1),
            check_dtype=False, # Allow float differences
            atol=1e-4 # Tolerance for float comparisons
        )


# Mock Tokenizer for HLADataset tests
class MockAlleleTokenizer:
    def __init__(self):
        self.special_tokens = {"PAD": 0, "UNK": 1, "BOS": 2, "EOS": 3}
        self.locus_vocabularies = {
            'HLA-A': {'A*01:01': 4, 'A*02:01': 5, **self.special_tokens},
            'HLA-B': {'B*07:02': 4, 'B*08:01': 5, **self.special_tokens}
        }
        self.locus_order = ['HLA-A', 'HLA-B'] # Define an order

    def tokenize(self, locus, allele):
        return self.locus_vocabularies.get(locus, {}).get(allele, self.special_tokens['UNK'])

    def get_vocab_size(self, locus):
         return len(self.locus_vocabularies.get(locus, {}))

# Mock CovariateEncoder for HLADataset tests
class MockCovariateEncoder:
    def __init__(self):
        self._fitted = True # Assume fitted

    def fit_transform(self, df):
        # Return a simple numpy array representation
        return df.to_numpy().astype(np.float32)

    def transform(self, df):
         return df.to_numpy().astype(np.float32)


class TestHLADataset(unittest.TestCase):

     def setUp(self):
        # Create sample data that GenotypeDataParser would produce
        self.parsed_genotypes = [
            [['A*01:01', 'A*02:01'], ['B*07:02', 'B*07:02']], # Sample 1
            [['A*01:01', 'A*01:01'], ['B*08:01', 'B*08:01']]  # Sample 2
        ]
        # Create sample covariate data (already encoded)
        self.encoded_covariates = np.array([[0.1, 0.9, -1.0], [0.8, 0.2, 1.0]], dtype=np.float32)

        self.mock_tokenizer = MockAlleleTokenizer()
        # self.mock_encoder = MockCovariateEncoder() # Not needed directly if covariates are pre-encoded

     def test_initialization_and_len(self):
         """Test HLADataset initialization and __len__."""
         dataset = HLADataset(
             genotypes=self.parsed_genotypes,
             covariates=self.encoded_covariates,
             tokenizer=self.mock_tokenizer,
             loci_order=['HLA-A', 'HLA-B'] # Specify locus order
         )
         self.assertEqual(len(dataset), 2) # Should match number of samples

     def test_getitem(self):
         """Test HLADataset __getitem__ returns correctly tokenized data."""
         dataset = HLADataset(
             genotypes=self.parsed_genotypes,
             covariates=self.encoded_covariates,
             tokenizer=self.mock_tokenizer,
             loci_order=['HLA-A', 'HLA-B']
         )

         # Get item for index 0
         item0 = dataset[0]

         # Expected tokenized genotype for sample 0:
         # HLA-A: A*01:01 -> 4, A*02:01 -> 5 => [4, 5]
         # HLA-B: B*07:02 -> 4, B*07:02 -> 4 => [4, 4]
         # Combined (assuming order HLA-A, HLA-B): [4, 5, 4, 4]
         expected_genotype_tokens0 = torch.tensor([4, 5, 4, 4], dtype=torch.long)
         expected_covariates0 = torch.tensor([0.1, 0.9, -1.0], dtype=torch.float32)

         self.assertTrue(torch.equal(item0['genotype_tokens'], expected_genotype_tokens0))
         self.assertTrue(torch.equal(item0['covariates'], expected_covariates0))
         self.assertEqual(item0['sample_index'], 0) # Check sample index

         # Get item for index 1
         item1 = dataset[1]
         # Expected tokenized genotype for sample 1:
         # HLA-A: A*01:01 -> 4, A*01:01 -> 4 => [4, 4]
         # HLA-B: B*08:01 -> 5, B*08:01 -> 5 => [5, 5]
         # Combined: [4, 4, 5, 5]
         expected_genotype_tokens1 = torch.tensor([4, 4, 5, 5], dtype=torch.long)
         expected_covariates1 = torch.tensor([0.8, 0.2, 1.0], dtype=torch.float32)

         self.assertTrue(torch.equal(item1['genotype_tokens'], expected_genotype_tokens1))
         self.assertTrue(torch.equal(item1['covariates'], expected_covariates1))
         self.assertEqual(item1['sample_index'], 1)


if __name__ == '__main__':
    unittest.main()
