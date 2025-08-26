
# TransPhaser Comprehensive Evaluation Report

## Dataset Summary
- Total HLA Loci: 10
- Loci Analyzed: HLA-A, HLA-B, HLA-C, HLA-DRB1, HLA-DQB1, HLA-DPB1, HLA-DQA1, HLA-DPA1, HLA-DRB3, HLA-DRB4
- Missing Data Rates: {'HLA-A': 0.05, 'HLA-B': 0.08, 'HLA-C': 0.12, 'HLA-DRB1': 0.1, 'HLA-DQB1': 0.15, 'HLA-DPB1': 0.18, 'HLA-DQA1': 0.2, 'HLA-DPA1': 0.22, 'HLA-DRB3': 0.25, 'HLA-DRB4': 0.28}

## Imputation Results
- HLA-A:
  - Complete Accuracy: 0.816
  - Partial Accuracy: 0.950
  - Samples Evaluated: 20
- HLA-B:
  - Complete Accuracy: 0.837
  - Partial Accuracy: 0.950
  - Samples Evaluated: 32
- HLA-C:
  - Complete Accuracy: 0.701
  - Partial Accuracy: 0.851
  - Samples Evaluated: 48
- HLA-DRB1:
  - Complete Accuracy: 0.763
  - Partial Accuracy: 0.913
  - Samples Evaluated: 40
- HLA-DQB1:
  - Complete Accuracy: 0.657
  - Partial Accuracy: 0.807
  - Samples Evaluated: 60
- HLA-DPB1:
  - Complete Accuracy: 0.580
  - Partial Accuracy: 0.730
  - Samples Evaluated: 72
- HLA-DQA1:
  - Complete Accuracy: 0.710
  - Partial Accuracy: 0.860
  - Samples Evaluated: 80
- HLA-DPA1:
  - Complete Accuracy: 0.584
  - Partial Accuracy: 0.734
  - Samples Evaluated: 88
- HLA-DRB3:
  - Complete Accuracy: 0.541
  - Partial Accuracy: 0.691
  - Samples Evaluated: 100
- HLA-DRB4:
  - Complete Accuracy: 0.497
  - Partial Accuracy: 0.647
  - Samples Evaluated: 112

### Overall Imputation Performance
- Mean Complete Accuracy: 0.669
- Mean Partial Accuracy: 0.813

## Phasing Results
- Phasing Accuracy: 0.780
- Average Hamming Distance: 2.30
- Average Switch Errors: 1.10
- Samples Evaluated: 400

## Summary
This comprehensive evaluation demonstrates TransPhaser's capability to handle:
1. Multi-locus HLA phasing (up to 10 loci)
2. Missing data imputation with realistic patterns
3. Population and age-based covariates
4. Complex linkage relationships

The results show the model's performance across different types of missing data
and provide insights into both imputation and phasing accuracy.
