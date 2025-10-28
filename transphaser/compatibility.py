import torch # Needed for mask generation
import logging # Added for potential warnings

class HaplotypeCompatibilityChecker:
    """
    Checks if a pair of haplotypes is compatible with a given genotype.
    Handles homozygosity and potentially missing data based on configuration.
    """
    def __init__(self, allow_imputation=False):
        """
        Initializes the checker.

        Args:
            allow_imputation (bool): If True, allows for special handling or
                                     relaxation of constraints for missing/unknown alleles.
                                     (Specific logic TBD based on requirements).
                                     Defaults to False.
        """
        self.allow_imputation = allow_imputation

    def check(self, genotype, haplotype1, haplotype2):
        """
        Checks if the two haplotypes are compatible with the genotype.

        Args:
            genotype (list): A list representing the genotype for a single locus,
                             containing two allele strings (e.g., ['A*01:01', 'A*02:01']).
                             Alleles should ideally be sorted for consistency.
            haplotype1 (str): The allele string for the first haplotype at this locus.
            haplotype2 (str): The allele string for the second haplotype at this locus.

        Returns:
            bool: True if the haplotype pair is compatible with the genotype, False otherwise.
        """
        # Basic validation
        if not isinstance(genotype, list) or len(genotype) != 2:
             raise TypeError("Genotype must be a list of two allele strings.")
        if not isinstance(haplotype1, str) or not isinstance(haplotype2, str):
             raise TypeError("Haplotypes must be strings.")

        # The core compatibility check:
        # The multiset of alleles in the genotype must equal the multiset
        # of alleles in the proposed haplotype pair.
        # Sorting both ensures order doesn't matter for comparison.
        genotype_sorted = sorted(genotype)
        haplotype_pair_sorted = sorted([haplotype1, haplotype2])

        return genotype_sorted == haplotype_pair_sorted

        # Note: This implementation assumes allow_imputation=False.
        # Handling imputation/missing data would require additional logic here,
        # potentially checking against special 'UNK' or 'PAD' tokens if used,
        # or allowing partial matches based on the self.allow_imputation flag.

    def check_batch(self, genotypes_batch, haplotype1_batch, haplotype2_batch):
        """Checks compatibility for a batch of genotypes and haplotype pairs."""
        # Basic implementation: iterate and call self.check
        results = []
        for genotype, hap1, hap2 in zip(genotypes_batch, haplotype1_batch, haplotype2_batch):
            results.append(self.check(genotype, hap1, hap2))
        return results

    def generate_compatibility_mask(self, genotype_tokens, vocab_size):
        """Generates a mask indicating valid alleles based on genotype tokens."""
        # Placeholder implementation - requires specific logic based on tokenization
        # and how compatibility should restrict the vocabulary.
        logging.warning("generate_compatibility_mask is not fully implemented.")
        # Example: Allow only tokens present in the genotype?
        mask = torch.zeros(genotype_tokens.shape[0], vocab_size, dtype=torch.bool, device=genotype_tokens.device)
        # Logic to populate the mask based on genotype_tokens would go here.
        # For now, return a mask allowing everything (or nothing, depending on desired default)
        mask.fill_(True) # Allow all tokens as a placeholder
        return mask


class HLACompatibilityRules:
    """
    Defines and enforces compatibility rules between genotypes and haplotype pairs.
    Can operate in strict or relaxed modes (e.g., for handling missing data).
    This class might be more focused on generating valid pairs or masks rather
    than just checking like HaplotypeCompatibilityChecker.
    """
    def __init__(self, strict_mode=True):
        """
        Initializes the rules engine.

        Args:
            strict_mode (bool): If True, enforces strict compatibility. If False,
                                allows for relaxed rules (e.g., for missing data).
                                Defaults to True.
        """
        self.strict_mode = strict_mode
        logging.debug("HLACompatibilityRules initialized.")

    def get_valid_haplotype_pairs(self, genotype):
        """
        Given a genotype, returns all possible valid haplotype pairs.

        Args:
            genotype (list): A list representing the genotype for a single locus,
                             containing two allele strings (e.g., ['A*01:01', 'A*02:01']).
                             Assumes alleles are already standardized.

        Returns:
            list: A list of tuples, where each tuple is a valid haplotype pair
                  (haplotype1, haplotype2). Order within the tuple might matter
                  depending on downstream use (e.g., (H1, H2) vs (H2, H1)).
                  Returns unique pairs considering order.
        """
        # Placeholder implementation
        # print(f"Placeholder: Getting valid pairs for genotype {genotype}.") # Reduce noise
        if not isinstance(genotype, list) or len(genotype) != 2:
             raise TypeError("Genotype must be a list of two allele strings.")

        allele1, allele2 = genotype[0], genotype[1]

        if allele1 == allele2:
            # Homozygous case: only one possible pair
            return [(allele1, allele1)]
        else:
            # Heterozygous case: two possible ordered pairs
            return [(allele1, allele2), (allele2, allele1)]

        # Note: Handling missing data (UNK tokens) or relaxed mode would add complexity here.


class CompatibilityMaskGenerator:
    """
    Generates masks to restrict decoder predictions based on genotype compatibility.
    Ensures that the decoder only predicts alleles that are compatible with the
    observed genotype and the haplotype allele already generated for the pair.
    """
    def __init__(self, tokenizer, compatibility_rules):
        """
        Initializes the CompatibilityMaskGenerator.

        Args:
            tokenizer: An initialized AlleleTokenizer instance.
            compatibility_rules: An initialized HLACompatibilityRules instance.
        """
        # Need to import AlleleTokenizer if not done globally
        # from .data_preprocessing import AlleleTokenizer
        # if not isinstance(tokenizer, AlleleTokenizer):
        #      raise TypeError("tokenizer must be an instance of AlleleTokenizer")
        # if not isinstance(compatibility_rules, HLACompatibilityRules):
        #      raise TypeError("compatibility_rules must be an instance of HLACompatibilityRules")

        self.tokenizer = tokenizer
        self.compatibility_rules = compatibility_rules
        logging.debug("CompatibilityMaskGenerator initialized.")

    def generate_mask(self, locus, genotype, partial_haplotype1=None):
        """
        Generates a mask for the next allele prediction for a specific locus.

        Args:
            locus (str): The locus for which to generate the mask.
            genotype (list): The genotype for this locus (e.g., ['A*01:01', 'A*02:01']).
            partial_haplotype1 (str, optional): The allele already chosen for the
                                                first haplotype at this locus. If None,
                                                it's assumed we are predicting the first
                                                haplotype allele. Defaults to None.

        Returns:
            torch.Tensor: A boolean tensor (or float tensor with -inf/0) of shape
                          (vocab_size,) where True (or 0) indicates a valid token
                          and False (or -inf) indicates an invalid token.
        """
        # Placeholder implementation
        # print(f"Placeholder: Generating mask for locus {locus}, genotype {genotype}, partial_hap1={partial_haplotype1}") # Reduce noise

        vocab_size = self.tokenizer.get_vocab_size(locus)
        mask = torch.zeros(vocab_size, dtype=torch.bool) # Initialize mask (all False/invalid)

        # Determine the set of allowed alleles based on genotype and partial haplotype
        allowed_alleles = set()
        allele1, allele2 = genotype[0], genotype[1]

        if partial_haplotype1 is None:
            # Predicting the first haplotype allele: any allele in the genotype is allowed
            allowed_alleles.add(allele1)
            allowed_alleles.add(allele2)
        else:
            # Predicting the second haplotype allele: the remaining allele is allowed
            if partial_haplotype1 == allele1:
                allowed_alleles.add(allele2)
            elif partial_haplotype1 == allele2:
                allowed_alleles.add(allele1)
            else:
                # This case implies the partial_haplotype1 was somehow incompatible
                # with the genotype. Depending on strictness, either raise error
                # or allow nothing/UNK. Allowing nothing by default.
                logging.warning(f"Incompatible partial haplotype '{partial_haplotype1}' provided for genotype {genotype} at locus {locus}.")

        # Add special tokens if they should always be allowed (e.g., EOS)
        # allowed_alleles.add("EOS") # Example

        # Set mask to True for allowed allele tokens
        for allele in allowed_alleles:
            token_id = self.tokenizer.tokenize(locus, allele)
            # Use self.tokenizer.special_tokens directly
            if token_id != self.tokenizer.special_tokens['UNK']: # Don't allow UNK unless explicitly handled
                if 0 <= token_id < vocab_size:
                    mask[token_id] = True
                else:
                     # Use logging if available
                     if logging:
                         logging.warning(f"Token ID {token_id} for allele '{allele}' out of range for locus {locus} vocab size {vocab_size}")
                     else:
                         print(f"Warning: Token ID {token_id} for allele '{allele}' out of range for locus {locus} vocab size {vocab_size}")


        # Convert to float mask if needed for softmax (-inf for masked, 0 for allowed)
        # float_mask = torch.zeros_like(mask, dtype=torch.float)
        # float_mask.masked_fill_(~mask, float('-inf'))
        # return float_mask

        return mask # Return boolean mask for now


class HaplotypeConstraintPropagator:
    """
    Propagates haplotype constraints across loci to potentially reduce the search space
    during generation or inference.

    Note: A meaningful implementation requires specific multi-locus constraint rules
    (e.g., based on Linkage Disequilibrium data), which are not provided by default.
    This implementation provides the structure but performs no actual propagation.
    """
    def __init__(self, compatibility_rules):
        """
        Initializes the HaplotypeConstraintPropagator.

        Args:
            compatibility_rules: An instance potentially defining multi-locus constraints.
                                 Currently, HLACompatibilityRules only handles single-locus checks.
        """
        # TODO: Enhance compatibility_rules or provide a separate mechanism
        #       for defining multi-locus constraints (e.g., LD data).
        self.compatibility_rules = compatibility_rules
        logging.info("HaplotypeConstraintPropagator initialized. Note: Using trivial propagation (no multi-locus rules).")

    def propagate(self, partial_haplotypes, known_genotypes):
        """
        Applies multi-locus constraints based on partially generated haplotypes.

        Note: This is a trivial implementation as no multi-locus rules (e.g., LD data)
              are provided. It currently returns no additional constraints beyond
              what single-locus compatibility implies.

        Args:
            partial_haplotypes (dict): Dictionary mapping locus -> (hap1_allele, hap2_allele)
                                       for loci already determined. Alleles can be None if not yet determined.
            known_genotypes (dict): Dictionary mapping locus -> [allele1, allele2] for all loci.

        Returns:
            dict: A dictionary representing any derived constraints. In this trivial
                  implementation, it always returns an empty dictionary, signifying
                  no cross-locus constraints were derived.
                  A functional implementation using LD data might return:
                  {'B': {'valid_hap2_alleles': {'B*15:01'}},
                   'C': {'valid_pairs': {('C*03:04', 'C*07:01'), ('C*07:01', 'C*03:04')}}}
        """
        logging.debug("Constraint propagation called (trivial implementation - no multi-locus rules applied).")

        # Trivial implementation: Return an empty dictionary as no cross-locus
        # constraints can be derived without specific rules or data (like LD).
        # A real implementation would analyze partial_haplotypes against known_genotypes
        # using LD patterns or other rules to restrict possibilities at undetermined loci.
        propagated_constraints = {}

        return propagated_constraints
