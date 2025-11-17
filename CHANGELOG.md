# Changelog

All notable changes to TransPhaser will be documented in this file.

## [0.2.0] - 2025-10-28

### Critical Bug Fixes

#### Parser Format Mismatch (Critical)
- **Fixed**: Genotype parser now correctly handles slash-separated format (`A*01/A*02`)
- **Impact**: Phasing accuracy improved from 0% to 45% on synthetic data
- **Details**: The parser was configured for comma-separated format but data used slash separation, causing all alleles to be tokenized as UNK tokens, which led to invalid "PAD_PAD_PAD" predictions
- **Files Modified**: `transphaser/data_preprocessing.py:66-71`

### Added

#### Testing
- **Integration Tests**: Added 5 comprehensive end-to-end tests in `tests/test_integration.py`:
  - `test_full_pipeline_with_training`: Complete workflow from training to evaluation
  - `test_model_save_and_load`: Model persistence verification
  - `test_input_format_validation`: Validates both slash and comma formats work
  - `test_tokenization_roundtrip`: Ensures tokenize/detokenize consistency
  - `test_dataset_batch_generation`: Validates data loader functionality
- **Test Coverage**: Increased from 96 to 101 total tests

#### Documentation
- **README.md**: Complete rewrite with:
  - Clear input data format specifications
  - Supported formats (slash-separated, comma-separated, homozygous)
  - Detailed usage examples
  - Troubleshooting guide
  - Performance benchmarks
  - Known issues and limitations
- **Input Format Documentation**: Added clear tables showing expected CSV formats
- **Parser Docstrings**: Updated to document supported separators

#### Repository Cleanup
- Removed empty directories: `/checkpoints/`, `/visualizer_output/`, `/output/`, `/tests/visualizer_output/`
- Removed debug files: `examples/debug_predictions.py`, `examples/test_parsing.py`
- Updated `.gitignore` to prevent future clutter

### Changed

#### Data Preprocessing
- `GenotypeDataParser.parse()`: Now tries slash separator first, then falls back to comma separator
- Maintains backward compatibility with comma-separated format
- Better error handling for unsupported formats

### Test Results

**Before Fix:**
```
Phasing Accuracy: 0.0%
Avg Hamming Distance: 6.0
All predictions: "PAD_PAD_PAD"
```

**After Fix:**
```
Phasing Accuracy: 45.4%
Avg Hamming Distance: 1.092
Valid allele predictions: ✓
```

### Breaking Changes

None - changes are backward compatible.

## [0.1.0] - 2025-03-31

### Initial Release

- Transformer-based variational autoencoder for HLA phasing
- Support for multiple HLA loci
- Covariate integration
- Missing data handling
- ELBO loss with KL annealing
- Model persistence and checkpointing
- Comprehensive evaluation metrics
- Example scripts and synthetic data generation
- 95 unit tests covering all components

---

## Format

This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

Categories:
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security fixes
