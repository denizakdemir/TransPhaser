import pandas as pd
import numpy as np
import random
import itertools
import os

# --- Configuration ---
NUM_SAMPLES = 5000
LOCI = ['HLA-A', 'HLA-B', 'HLA-DRB1']
MISSING_RATE = 0.05 # Proportion of alleles to mark as missing in one dataset
OUTPUT_DIR = "examples/data"
SEED = 42

# --- Define Alleles and Haplotype Frequencies (Simplified Example) ---
# Usually loaded from external frequency databases (e.g., AFND)
ALLELES = {
    'HLA-A': [f'A*{i:02d}' for i in range(1, 6)], # A*01 to A*05
    'HLA-B': [f'B*{i:02d}' for i in range(1, 8)], # B*01 to B*07
    'HLA-DRB1': [f'DRB1*{i:02d}' for i in range(1, 5)] # DRB1*01 to DRB1*04
}

# Generate all possible haplotypes based on defined alleles
possible_haplotypes_tuples = list(itertools.product(*[ALLELES[locus] for locus in LOCI]))
possible_haplotypes = ["_".join(hap) for hap in possible_haplotypes_tuples]

# Assign random frequencies (ensure they sum to 1)
np.random.seed(SEED)
raw_freqs = np.random.rand(len(possible_haplotypes))
HAPLOTYPE_FREQUENCIES = {hap: freq / raw_freqs.sum() for hap, freq in zip(possible_haplotypes, raw_freqs)}
print(f"Generated {len(possible_haplotypes)} possible haplotypes.")
# print("Example Haplotype Frequencies:", dict(list(HAPLOTYPE_FREQUENCIES.items())[:5]))


# --- Define Covariates ---
COVARIATES = {
    'Population': ['EUR', 'AFR', 'ASN'],
    'AgeGroup': ['0-18', '19-40', '41-65', '65+']
}

# --- Helper Functions ---
def sample_haplotype(frequencies):
    """Samples a single haplotype based on frequencies."""
    haps, freqs = zip(*frequencies.items())
    return np.random.choice(haps, p=freqs)

def parse_haplotype(haplotype_str):
    """Splits a haplotype string into allele components."""
    return haplotype_str.split('_')

def create_unphased_genotype(hap1_alleles, hap2_alleles):
    """Creates an unphased genotype representation (sorted pairs)."""
    genotype = {}
    for i, locus in enumerate(LOCI):
        pair = sorted([hap1_alleles[i], hap2_alleles[i]])
        genotype[locus] = "/".join(pair)
    return genotype

def introduce_missing(genotype_dict, missing_rate, missing_token="UNK"):
    """Randomly replaces alleles with a missing token."""
    missing_genotype = genotype_dict.copy()
    for locus in LOCI:
        alleles = genotype_dict[locus].split('/')
        new_alleles = []
        for allele in alleles:
            if random.random() < missing_rate:
                new_alleles.append(missing_token)
            else:
                new_alleles.append(allele)
        # Ensure sorting consistency even with UNK
        missing_genotype[locus] = "/".join(sorted(new_alleles))
    return missing_genotype

# --- Main Data Generation ---
def generate_data(num_samples, frequencies, loci_alleles, covariates_def):
    phased_data = []
    unphased_data = []
    unphased_missing_data = []

    for i in range(num_samples):
        individual_id = f"SAMPLE_{i+1:04d}"

        # Sample two haplotypes
        hap1_str = sample_haplotype(frequencies)
        hap2_str = sample_haplotype(frequencies)
        hap1_alleles = parse_haplotype(hap1_str)
        hap2_alleles = parse_haplotype(hap2_str)

        # Sample covariates
        covariate_values = {cov: random.choice(vals) for cov, vals in covariates_def.items()}

        # Create phased record
        phased_record = {'IndividualID': individual_id, 'Haplotype1': hap1_str, 'Haplotype2': hap2_str}
        phased_record.update(covariate_values)
        phased_data.append(phased_record)

        # Create unphased record
        unphased_genotype = create_unphased_genotype(hap1_alleles, hap2_alleles)
        unphased_record = {'IndividualID': individual_id}
        unphased_record.update(unphased_genotype)
        unphased_record.update(covariate_values)
        unphased_data.append(unphased_record)

        # Create unphased record with missing data
        missing_genotype = introduce_missing(unphased_genotype, MISSING_RATE)
        unphased_missing_record = {'IndividualID': individual_id}
        unphased_missing_record.update(missing_genotype)
        unphased_missing_record.update(covariate_values)
        unphased_missing_data.append(unphased_missing_record)

    # Convert to DataFrames
    df_phased = pd.DataFrame(phased_data)
    df_unphased = pd.DataFrame(unphased_data)
    df_unphased_missing = pd.DataFrame(unphased_missing_data)

    return df_phased, df_unphased, df_unphased_missing

# --- Execution ---
if __name__ == "__main__":
    print("Generating synthetic HLA data...")
    random.seed(SEED)
    np.random.seed(SEED)

    df_phased, df_unphased, df_unphased_missing = generate_data(
        NUM_SAMPLES, HAPLOTYPE_FREQUENCIES, ALLELES, COVARIATES
    )

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save to CSV files
    phased_path = os.path.join(OUTPUT_DIR, "synthetic_haplotypes_phased.csv")
    unphased_path = os.path.join(OUTPUT_DIR, "synthetic_genotypes_unphased.csv")
    unphased_missing_path = os.path.join(OUTPUT_DIR, "synthetic_genotypes_unphased_missing.csv")

    df_phased.to_csv(phased_path, index=False)
    df_unphased.to_csv(unphased_path, index=False)
    df_unphased_missing.to_csv(unphased_missing_path, index=False)

    print(f"Synthetic data saved to {OUTPUT_DIR}:")
    print(f"- Phased (Ground Truth): {os.path.basename(phased_path)}")
    print(f"- Unphased: {os.path.basename(unphased_path)}")
    print(f"- Unphased with Missing ({MISSING_RATE*100}%): {os.path.basename(unphased_missing_path)}")
    print("Generation complete.")
