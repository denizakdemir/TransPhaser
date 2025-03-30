# Transformer-Based HLA Phasing Suite (k-Locus): Implementation Plan

I'll develop a detailed implementation plan for each component of the HLA phasing system, focusing on design patterns, architecture, and implementation details rather than the actual code.

## Component 1: Data Preprocessing

### 1.1 GenotypeDataParser Class
```python
class GenotypeDataParser:
    def __init__(self, locus_columns, covariate_columns):
        # Store configuration of which columns contain genotype data vs covariates
        # Support flexible column naming patterns for HLA loci
```

- Parse DataFrame with validation for required columns and data types
- Extract HLA genotype data from specified columns 
- Support both list format (e.g., `['A*02:01', 'A*24:02']`) and string format (e.g., `'A*02:01,A*24:02'`)
- Standardize representation across different input formats

### 1.2 AlleleTokenizer Class
```python
class AlleleTokenizer:
    def __init__(self):
        self.locus_vocabularies = {}  # Dict mapping locus -> vocabulary
        self.special_tokens = {
            "PAD": 0,
            "UNK": 1,
            "BOS": 2,
            "EOS": 3
        }
```

- Build separate vocabulary for each locus to maintain locus-specific allele semantics
- Implement bidirectional mapping between allele strings and integer tokens
- Support vocabulary serialization/deserialization for model persistence
- Include frequency-based pruning for rare alleles

### 1.3 CovariateEncoder Class
```python
class CovariateEncoder:
    def __init__(self, categorical_covariates, numerical_covariates):
        # Store which covariates are categorical vs numerical
        self.encoders = {}  # Store fitted encoders for each covariate
```

- Encode categorical variables (race, ethnicity) using one-hot or embedding approaches
- Normalize numerical covariates with configurable normalization strategies
- Handle missing covariate values with imputation or special tokens
- Produce fixed-dimension embedding for all covariates

### 1.4 BatchGenerator Implementation
```python
class HLADataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, covariate_encoder, max_length=None):
        # Store preprocessed data and configuration
```

- Implement PyTorch Dataset/DataLoader pattern for efficient batch processing
- Handle padding and masking for batched transformer inputs
- Generate appropriate attention masks for transformer processing
- Support dynamic batch sizing based on sequence length

## Component 2: Latent Variable Formulation

### 2.1 ProbabilisticModel Class
```python
class HLAPhasingModel:
    def __init__(self, num_loci, allele_vocabularies, covariate_dim):
        # Initialize encoder and decoder models
        # Store configuration about problem structure
```

- Define prior distribution p(h|c) using decoder transformer
- Define approximate posterior q(h|g,c) using encoder transformer
- Track ELBO components for optimization
- Support conditioning on covariates throughout the model

### 2.2 CompatibilityConstraints Class
```python
class HaplotypeCompatibilityChecker:
    def __init__(self, allow_imputation=False):
        # Store configuration about constraint enforcement
```

- Implement efficient batch-compatible constraint checking
- Generate compatibility masks to restrict decoder outputs
- Define special handling for missing/unknown alleles
- Support both hard and soft constraints for differentiability

### 2.3 LatentSpaceExplorer Class
```python
class HaplotypeSpaceExplorer:
    def __init__(self, compatibility_checker, sampling_temperature=1.0):
        # Store configuration for sampling behavior
```

- Generate valid samples from latent space of compatible haplotypes
- Implement diverse sampling strategies (beam search, nucleus sampling)
- Balance exploration vs exploitation in haplotype space
- Track sample statistics for diagnostic purposes

## Component 3: Generative Transformer Decoder

### 3.1 HaplotypeDecoder Architecture
```python
class HaplotypeDecoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize transformer layers, embeddings, etc.
```

- Implement decoder-only transformer with causal masking
- Use separate embedding layers for alleles, locus positions, and covariates
- Support efficient batched forward passes
- Include gradient checkpointing for memory efficiency

### 3.2 PositionalEmbeddings Class
```python
class LocusPositionalEmbedding(nn.Module):
    def __init__(self, num_loci, embedding_dim):
        super().__init__()
        # Initialize locus-specific embeddings
```

- Generate embeddings that encode both sequence position and locus identity
- Support arbitrary k-locus configurations
- Include options for fixed vs learned embeddings
- Implement efficient batched embedding lookups

### 3.3 AlleleEmbedding Class
```python
class AlleleEmbedding(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim):
        super().__init__()
        # Initialize per-locus embedding tables
```

- Support separate embedding spaces for each locus
- Share embedding parameters where appropriate
- Handle special tokens consistently across loci
- Include embedding dropout for regularization

### 3.4 AutoregressiveDecoder Class
```python
class AutoregressiveHaplotypeDecoder:
    def __init__(self, transformer_model, tokenizer, max_length):
        # Store model and configuration
```

- Implement efficient autoregressive generation
- Support beam search with configurable beam width
- Include temperature and top-k/top-p sampling
- Enforce compatibility constraints during generation

## Component 4: Variational Inference Encoder

### 4.1 GenotypeEncoder Architecture
```python
class GenotypeEncoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize transformer layers, embeddings, etc.
```

- Implement bidirectional transformer encoder
- Process unphased genotype data with covariates
- Generate latent representations for posterior parameterization
- Support efficient batched inference

### 4.2 PosteriorDistribution Class
```python
class HaplotypePosteriorDistribution:
    def __init__(self, latent_dim, num_loci, vocab_sizes):
        # Initialize parameters for distribution
```

- Parameterize posterior distribution over haplotype pairs
- Support efficient sampling with reparameterization
- Calculate log probabilities for ELBO computation
- Enforce compatibility with observed genotype

### 4.3 GumbelSoftmaxSampler Class
```python
class GumbelSoftmaxSampler:
    def __init__(self, initial_temperature=1.0, min_temperature=0.1, anneal_rate=0.003):
        # Store temperature parameters
```

- Implement differentiable discrete sampling
- Support annealing schedules for temperature
- Include straight-through estimator option
- Maintain numerical stability during sampling

### 4.4 ConstrainedSampling Class
```python
class ConstrainedHaplotypeSampler:
    def __init__(self, compatibility_checker):
        # Store constraint configuration
```

- Sample haplotype pairs that satisfy genotype constraints
- Apply masking to enforce compatibility
- Handle missing data during sampling
- Support both training-time and inference-time sampling

## Component 5: ELBO Optimization

### 5.1 ELBOLoss Class
```python
class HLAPhasingELBO(nn.Module):
    def __init__(self, kl_weight=1.0, reconstruction_weight=1.0):
        super().__init__()
        # Store loss weights and configuration
```

- Calculate Evidence Lower Bound components
- Support KL annealing with configurable schedules
- Handle batch-level computation efficiently
- Include regularization terms as needed

### 5.2 KLAnnealingScheduler Class
```python
class KLAnnealingScheduler:
    def __init__(self, anneal_type='linear', max_weight=1.0, cycles=1):
        # Store annealing configuration
```

- Implement different annealing schedules (linear, cyclical, sigmoid)
- Track training progress for scheduling
- Provide current annealing weight for loss calculation
- Support schedule adjustment based on validation metrics

### 5.3 TrainingLoop Implementation
```python
class HLAPhasingTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, device):
        # Store training configuration
```

- Implement efficient training loop with gradient accumulation
- Track and log metrics during training
- Include validation steps with early stopping
- Support distributed training for larger models

### 5.4 CheckpointManager Class
```python
class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        # Configure checkpoint storage
```

- Save and load model checkpoints with metadata
- Implement best model selection based on metrics
- Support experiment tracking integration
- Include model export for deployment

## Component 6: Compatibility Enforcer

### 6.1 CompatibilityRules Class
```python
class HLACompatibilityRules:
    def __init__(self, strict_mode=True):
        # Configure enforcement strictness
```

- Define rules mapping genotypes to valid haplotype pairs
- Handle special cases like homozygosity and missing data
- Support both strict and relaxed constraint enforcement
- Implement efficient vectorized constraint checking

### 6.2 MaskGenerator Class
```python
class CompatibilityMaskGenerator:
    def __init__(self, tokenizer, compatibility_rules):
        # Store configuration
```

- Generate masks to restrict decoder predictions to compatible alleles
- Update masks dynamically during autoregressive generation
- Support batched mask generation for efficiency
- Handle special tokens in masking logic

### 6.3 ConstraintPropagation Class
```python
class HaplotypeConstraintPropagator:
    def __init__(self, compatibility_rules):
        # Store configuration
```

- Propagate constraints across loci to reduce search space
- Implement forward and backward propagation of constraints
- Support efficient batch processing of constraints
- Track constraint satisfaction during generation

## Component 7: Missing Data Handler

### 7.1 MissingDataDetector Class
```python
class MissingDataDetector:
    def __init__(self, tokenizer):
        # Store configuration
```

- Identify patterns of missing data in input
- Categorize missing data types (missing locus, partial information)
- Generate appropriate masks for handling missing data
- Track missing data statistics for reporting

### 7.2 MarginalizationEngine Class
```python
class MissingDataMarginalizer:
    def __init__(self, model, sampling_iterations=10):
        # Store configuration
```

- Marginalize over possible values for missing data
- Use importance sampling for efficient approximation
- Track uncertainty introduced by missing data
- Support both training and inference-time marginalization

### 7.3 ImputationStrategies Class
```python
class AlleleImputer:
    def __init__(self, model, imputation_strategy='sampling'):
        # Store configuration
```

- Implement strategies for imputing missing alleles
- Use model posterior to guide imputation
- Include uncertainty estimates for imputed values
- Support multiple imputation for robust analysis

### 7.4 RelaxedCompatibilityLoss Class
```python
class RelaxedCompatibilityLoss(nn.Module):
    def __init__(self, base_weight=1.0, missing_weight=0.5):
        super().__init__()
        # Store loss weights
```

- Implement relaxed constraints for missing data
- Balance strictness vs flexibility based on data quality
- Apply appropriate weighting for different missing patterns
- Support gradient-friendly relaxation mechanisms

## Component 8: Evaluation Tools

### 8.1 MetricCalculator Class
```python
class HLAPhasingMetrics:
    def __init__(self, tokenizer):
        # Store configuration
```

- Calculate Hamming distance between predicted and true haplotypes
- Compute switch error rates and other phasing metrics
- Support batch computation for efficiency
- Include normalization and scaling options

### 8.2 UncertaintyEstimator Class
```python
class PhasingUncertaintyEstimator:
    def __init__(self, model, sampling_iterations=100):
        # Store configuration
```

- Calculate entropy-based uncertainty metrics
- Estimate posterior variance through sampling
- Identify ambiguous phasing cases
- Generate uncertainty profiles across loci

### 8.3 CandidateRanker Class
```python
class HaplotypeCandidateRanker:
    def __init__(self, model, num_candidates=10, diversity_weight=0.1):
        # Store configuration
```

- Rank haplotype pairs by likelihood/probability
- Implement efficient top-k selection
- Include diversity sampling for multiple candidates
- Calculate normalized scores for interpretability

### 8.4 ResultVisualizer Class
```python
class PhasingResultVisualizer:
    def __init__(self, tokenizer):
        # Store configuration
```

- Generate visualizations of phasing results
- Plot likelihood distributions and uncertainties
- Create haplotype alignment visualizations
- Support export to interactive formats

## Component 9: Configuration and Extensibility

### 9.1 ConfigurationSystem Design
```python
class HLAPhasingConfig:
    def __init__(self, config_path=None):
        # Load configuration or use defaults
```

- Use Pydantic models for configuration with validation
- Support hierarchical configuration for components
- Include sensible defaults with documentation
- Allow runtime reconfiguration where appropriate

### 9.2 LocusMetadataManager Class
```python
class LocusMetadataManager:
    def __init__(self, locus_config=None):
        # Initialize with default or custom configuration
```

- Manage metadata about HLA loci and alleles
- Support arbitrary k-locus configurations
- Include population-specific frequency information
- Allow locus-specific parameters for model components

### 9.3 ExternalPriorInterface Design
```python
class AlleleFrequencyPrior:
    def __init__(self, prior_source, prior_weight=1.0):
        # Configure prior source and weight
```

- Define interfaces for incorporating external frequency priors
- Support common HLA frequency database formats
- Include validation and normalization for external data
- Allow mixing of data-driven and external priors

### 9.4 CovariateHandler Class
```python
class CovariateManager:
    def __init__(self, categorical_covariates=None, numerical_covariates=None):
        # Configure covariate handling
```

- Manage categorical and numerical covariates
- Support flexible encoding strategies
- Allow assessment of covariate importance
- Include mechanisms to selectively use covariates

## Output Interface

### ResultFormatter Class
```python
class PhasingResultFormatter:
    def __init__(self, tokenizer, num_candidates=5):
        # Configure output formatting
```

- Format phasing results as structured JSON
- Include ranked haplotype pairs with scores
- Add uncertainty metrics and metadata
- Support customizable output formats

## Integration and Deployment

### 1. Runner Script Design
```python
class HLAPhasingRunner:
    def __init__(self, config_path):
        # Initialize system from configuration
```

- Provide command-line interface for phasing
- Support batch processing of samples
- Include progress tracking and logging
- Manage computational resources efficiently

### 2. Testing Framework
```python
class HLAPhasingTestSuite:
    def __init__(self, model_path, test_data_path):
        # Configure test suite
```

- Generate synthetic test data with known ground truth
- Implement comprehensive test cases for components
- Include performance benchmarks
- Support regression testing during development

### 3. Performance Reporting
```python
class PerformanceReporter:
    def __init__(self, metrics_log, output_dir):
        # Configure reporting
```

- Generate comprehensive performance reports
- Plot training curves and evaluation metrics
- Include runtime performance statistics
- Support comparison between model versions

---

This detailed implementation plan covers the core components necessary for building a transformer-based HLA phasing system with support for arbitrary k-locus configurations and robust handling of missing data. The modular design ensures flexibility, extensibility, and maintainability throughout development.