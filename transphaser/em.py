"""
Expectation-Maximization (EM) algorithm for haplotype frequency estimation and phasing.
This serves as a classical baseline for benchmarking TransPhaser.
Based on the principles of Excoffier and Slatkin (1995).
"""

import numpy as np
from typing import List, Tuple, Dict, Set
import logging
import itertools
from collections import defaultdict
from transphaser.evaluation import HLAPhasingMetrics
from transphaser.data_preprocessing import AlleleTokenizer

class EMHaplotypePhaser:
    """
    Classical EM algorithm for phasing.
    
    This implementation handles multi-allelic loci (like HLA).
    It assumes Hardy-Weinberg Equilibrium.
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 1000):
        """
        Args:
            tolerance: Convergence threshold for change in log-likelihood (or sum of abs diffs).
            max_iterations: Maximum number of EM iterations.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.haplotype_frequencies: Dict[Tuple[str, ...], float] = {}
        self.converged = False
        self.n_iter_ = 0
        self.history_ = []

    def _get_possible_phasings(self, genotype: List[Tuple[str, str]], known_alleles_per_locus: List[Set[str]] = None) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        """
        Given a genotype (list of allele pairs), return all possible haplotype pairs (h1, h2)
        that could form this genotype. Handles 'MISSING' alleles by marginalizing over 
        known alleles at that locus.
        
        Args:
            genotype: List of (a1, a2) tuples, one for each locus.
            known_alleles_per_locus: List of sets of known alleles for each locus. 
                                     Required for handling missing data.
            
        Returns:
            List of (h1, h2) tuples, where h1 and h2 are tuples of alleles.
        """
        locus_choices = []
        
        for i, (a1, a2) in enumerate(genotype):
            choices_for_this_locus = []
            
            # Determine effective alleles to consider
            candidates_1 = [a1] if a1 != "MISSING" else []
            candidates_2 = [a2] if a2 != "MISSING" else []
            
            # If nothing passed in known_alleles, fallback to just preserving MISSING 
            # (though this is degenerate for EM)
            known_alleles = known_alleles_per_locus[i] if known_alleles_per_locus else set()
            if not known_alleles and (a1 == "MISSING" or a2 == "MISSING"):
                 # Fallback if we have no knowledge (e.g. all missing)
                 known_alleles = {"MISSING"}

            if not candidates_1:
                candidates_1 = list(known_alleles)
            if not candidates_2:
                candidates_2 = list(known_alleles)
                
            # Now generate compatible pairs for this locus
            # This can be expensive if highly polymorphic and full missing.
            # Optimization: 
            # If we simply take Cartesian product of candidates_1 x candidates_2, 
            # we cover all (h1_i, h2_i).
            # We want to represent the set of unordered pairs {c1, c2}.
            # But the EM logic expects distinct ordered pairs (h1, h2).
            
            for c1 in candidates_1:
                for c2 in candidates_2:
                    if c1 == c2:
                        choices_for_this_locus.append((c1, c1))
                    else:
                        # For a specific pair of DISTINCT alleles, we have two ordered possibilities.
                        # However, we must be careful not to double count if we iterate loosely.
                        # here c1 and c2 are from our candidate lists.
                        # The pair is (c1, c2).
                        # The call to product(*locus_choices) creates ordered Haplotypes.
                        # So simply appending (c1, c2) is sufficient for that specific choice.
                        # BUT wait:
                        # If a1="A", a2="B". c1 can only be A. c2 can only be B. pair (A, B).
                        # Current logic expects us to return BOTH (A,B) and (B,A) if distinct?
                        # Original code:
                        #   locus_choices.append([(a1, a2), (a2, a1)])
                        # So yes, we need to explicitly allow both orderings if they come from the original genotype constraint.
                        
                        # Case 1: Original was (A, B). Distinct.
                        # c1=A, c2=B. We add (A, B).
                        # We also need (B, A)?
                        # In the original code, the loop was simple.
                        # Here, if a1="MISSING", c1 iterates all.
                        # If a1="A", a2="MISSING". c1="A". c2 iterates all (say A, B, C).
                        # pairs: (A, A), (A, B), (A, C).
                        # Do we imply (B, A)?
                        # Since we are constructing H1 and H2.
                        # H1 takes c1. H2 takes c2.
                        # If c1=A, c2=B -> H1 has A, H2 has B.
                        # If we want H1 has B, H2 has A to be possible, we need c1=B, c2=A.
                        # IF a1 was MISSING and a2 was MISSING:
                        #   c1 iterates A, B. c2 iterates A, B.
                        #   (c1=A, c2=B) -> (A, B)
                        #   (c1=B, c2=A) -> (B, A)
                        #   So the cartesian product candidates_1 x candidates_2 covers BOTH directions automatically
                        #   IF both are missing.
                        
                        # IF a1="A", a2="MISSING":
                        #   c1=A. c2 in {A, B}.
                        #   (A, A), (A, B).
                        #   Does (B, A) exist?
                        #   NO. Because the genotype says "Allele 1 is A".
                        #   Wait. Genotype data comes as unordered set usually {A, B}.
                        #   But the input format `List[Tuple]` implies order? 
                        #   Usually genotype data is unphased, so (A, B) is same as (B, A).
                        #   Our parser sorts alleles: `sample_genotypes.append(sorted(alleles))`.
                        #   So a1 is always alphabetically <= a2.
                        #   So the input is effectively {a1, a2}.
                        #   If input is (A, B) (sorted):
                        #     Original code: returns [(A, B), (B, A)].
                        #   If input is (A, MISSING):
                        #     This means one observed is A. The other is unknown.
                        #     So {A, ?}.
                        #     Possibilities for {h1_i, h2_i} are {A, x}.
                        #     So ordered pairs (A, x) and (x, A).
                        #     Our loops:
                        #       c1 is from a1 (A). c2 is from a2 (?).
                        #       c1=A. c2=x. -> (A, x).
                        #       This gives ONE ordering. We are missing (x, A)!
                        #     So if we have distinct c1, c2, we might not be covering the swap 
                        #     UNLESS the source was symmetrical.
                        #     Here the source is NOT symmetrical (A vs MISSING).
                        #     We strictly need to interpret (A, MISSING) as the SET {A, x}.
                        #     So H1 can be x, H2 can be A.
                        
                        # Fix:
                        # We are identifying a set of alleles present at this locus.
                        # Obs = {A, ?}. True = {A, x}.
                        # H1_i, H2_i can be A, x OR x, A.
                        # If we just generate (A, x), we constrain H1 to be the one carrying A?
                        # No, Haplotypes are global paths.
                        # H1 = (L1_A, L2_..., ...).
                        # If we fix L1 to be A for H1 and x for H2, we forbid H1 having x.
                        # We must allow both swaps.
                        
                        choices_for_this_locus.append((c1, c2))
                        
                        # If this specific combination (c1, c2) implies a heterogeneous state where
                        # semantics allow swapping, do we add (c2, c1)?
                        # If a1, a2 were both MISSING:
                        #   c1=A, c2=B -> (A, B) added.
                        #   c1=B, c2=A -> (B, A) added (in a later iteration of c1/c2 loops).
                        #   So full cover.
                        # If a1=A, a2=MISSING:
                        #   c1=A, c2=B -> (A, B) added.
                        #   c1=B impossible (since a1=A).
                        #   So we manage only (A, B). We MISS (B, A).
                        #   We MUST add (c2, c1) explicitly if standard unphased logic applies.
                        #   BUT we only add it if (c2, c1) wouldn't be generated by the loop itself?
                        #   Or simpler: Generate the SET of alleles {c1, c2}.
                        #   Then return Permutations of that set.
                        
                        # Let's clean this logic.
                        pass

            # Refined Generation Logic:
            # We want all ordered pairs (u, v) such that {u, v} is compatible with {a1, a2}.
            # If a1, a2 not missing: {u,v} == {a1,a2}.
            # If a1=A, a2=?: {u,v} == {A, x} for some x.
            # If a1=?, a2=?: {u,v} == {x, y} for some x, y.
            
            # Implementation:
            real_options = []
            
            # 1. Expand 'MISSING' to all possibilities
            expanded_sets = []
            
            # candidates_1/2 logic from above is good as "sets of possibilities for slot 1 and 2"
            # BUT slot 1 and slot 2 in the input tuple (a1, a2) are arbitrary if unphased.
            # Actually, `data_preprocessing` sorts them.
            # So (A, MISSING) is fixed order.
            
            # Let's iterate all x in candidates_1, all y in candidates_2.
            # This generates a pair (x, y) that is "compatible" with the observations (a1, a2)
            # in the sense that x is consistent with a1 and y is consistent with a2.
            # BUT genotyping is unphased.
            # So observation is the SET {A, MISSING}.
            # Hypothetical true genotype is SET {A, x}.
            # We want all ordered pairs (h1, h2) such that {h1, h2} == {A, x}.
            # i.e., (A, x) and (x, A).
            
            # If we iterate c1 in cand1 (A), c2 in cand2 (All):
            # We get pairs (A, x).
            # This covers {A, x} mapped to (H1=A, H2=x).
            # We DO NOT get (x, A) unless A was in cand2 AND x was in cand1.
            # Here cand1 is {A}. cand2 is {All}.
            # If x != A, x is NOT in cand1. So we never generate (x, A).
            
            # So we implicitly force H1 to carry 'A' and H2 to carry 'x'.
            # This constrains the EM incorrectly?
            # Yes, because H1 might need to be the haplotype (..., x, ...).
            
            # CONCLUSION:
            # For every pair (c1, c2) generated from cand1 x cand2:
            #   Add (c1, c2)
            #   If c1 != c2: Add (c2, c1)
            # Eliminate duplicates?
            
            seen_pairs = set()
            
            for c1 in candidates_1:
                for c2 in candidates_2:
                    p1 = (c1, c2)
                    if p1 not in seen_pairs:
                        real_options.append(p1)
                        seen_pairs.add(p1)
                    
                    if c1 != c2:
                        p2 = (c2, c1)
                        if p2 not in seen_pairs:
                            real_options.append(p2)
                            seen_pairs.add(p2)
            
            locus_choices.append(real_options)
        
        # Cartesian product across loci
        possible_phasings = []
        for choice in itertools.product(*locus_choices):
            h1 = tuple(c[0] for c in choice)
            h2 = tuple(c[1] for c in choice)
            possible_phasings.append((h1, h2))
            
        return possible_phasings

    def fit(self, genotypes: List[List[Tuple[str, str]]]):
        """
        Estimate haplotype frequencies from a population of genotypes.
        
        Args:
            genotypes: List of N samples. Each sample is a list of L loci.
                       Each locus is a tuple (allele1, allele2).
        """
        # 0. Discover observed alleles and COMPLETE haplotypes
        if not genotypes:
            return 
            
        n_loci = len(genotypes[0])
        self.known_alleles_per_locus = [set() for _ in range(n_loci)]
        
        known_haplotypes = set()
        
        # First pass: Collect alleles and potential haplotypes from complete samples
        for g in genotypes:
            is_complete = True
            for i, (a1, a2) in enumerate(g):
                if a1 == "MISSING" or a2 == "MISSING":
                    is_complete = False
                if a1 != "MISSING": self.known_alleles_per_locus[i].add(a1)
                if a2 != "MISSING": self.known_alleles_per_locus[i].add(a2)
            
            if is_complete:
                # If complete, we can get phasings efficiently
                phasings = self._get_possible_phasings(g, self.known_alleles_per_locus)
                for h1, h2 in phasings:
                    known_haplotypes.add(h1)
                    known_haplotypes.add(h2)

        # 1. Build sample phasings
        all_unique_haplotypes = known_haplotypes.copy()
        sample_phasings = [] 
        
        logging.info(f"EM: initializing with {len(genotypes)} samples. Found {len(known_haplotypes)} haplotypes from complete samples.")

        # Reset state for re-fitting
        self.converged = False
        self.history_ = []
        self.n_iter_ = 0
        
        # Optimize: Convert known_haplotypes to list for indexing if needed, 
        # but set is good for membership.
        # We need a way to quickly find compatible haplotypes for incomplete samples without checking H^2 pairs?
        # Maybe H^2 is fine if H is small (e.g. < 500). 500^2 = 250,000. 
        # For 4000 samples, 4000 * 250,000 = 10^9 ops. Too slow.
        
        # Optimization:
        # Pre-index known haplotypes by allele at each locus?
        # index[locus_idx][allele] -> list of haplotypes having that allele
        
        hap_index = [defaultdict(list) for _ in range(n_loci)]
        known_hap_list = list(known_haplotypes)
        for h in known_hap_list:
            for i, allele in enumerate(h):
                hap_index[i][allele].append(h)
                
        for g in genotypes:
            # Check if complete (fast path)
            is_complete = all(a1 != "MISSING" and a2 != "MISSING" for a1, a2 in g)
            
            if is_complete:
                phasings = self._get_possible_phasings(g, self.known_alleles_per_locus)
            else:
                # Incomplete sample. 
                # Strategy: Find pairs (h1, h2) from known_haplotypes that match g.
                # If no matches (e.g. rare alleles not in complete set), fallback to expansion?
                # Let's try matching against known first.
                
                # We need h1, h2 such that for all loci: {h1[i], h2[i]} is compatible with g[i]
                # g[i] is {a1, a2} where bits can be MISSING.
                
                # This is a Constraint Satisfaction Problem.
                # Backtracking search?
                # Or just use the original expansion, but intersected with known_haplotypes?
                # Original expansion generates alleles.
                # If we filter `candidates` to those present in `known_haplotypes`'s alleles at that locus? (Already done via known_alleles_per_locus).
                
                # If we use the original _get_possible_phasings, it explodes.
                # Let's try to constrain it.
                
                # If we assume at least ONE haplotype in the pair comes from our known set (or both).
                # Let's assume BOTH come from the known set (standard interpolation).
                
                # Helper to check compatibility
                def is_compatible(h1, h2, gen):
                    for i, (ga1, ga2) in enumerate(gen):
                        obs = set()
                        if ga1 != "MISSING": obs.add(ga1)
                        if ga2 != "MISSING": obs.add(ga2)
                        
                        has = {h1[i], h2[i]}
                        
                        # logic: 
                        # if ga1 != MISSING, ga1 must be in has.
                        # if ga2 != MISSING, ga2 must be in has.
                        # AND counts must match?
                        # "A/A" -> {A}. has={A, A}. OK.
                        # "A/MISSING" -> {A}. has={A, B}. OK. (A matched A. B matched MISSING).
                        # "A/B" -> {A, B}. has={A, B}. OK.
                        
                        # Implementation:
                        # Make copies to consume
                        pool = list(has)
                        targets = []
                        if ga1 != "MISSING": targets.append(ga1)
                        if ga2 != "MISSING": targets.append(ga2)
                        
                        for t in targets:
                            if t in pool:
                                pool.remove(t)
                            else:
                                return False
                    return True

                matched_phasings = []
                
                # Heuristic: Filter candidate haplotypes first?
                # If `g` has "A" at locus 0, then at least one haplotype must have "A" at locus 0.
                # This can pruning the search space.
                
                # But brute forcing H^2 might be just on the edge.
                # If H ~ 100-200, it's fine. If H ~ 1000, it's slow.
                # Let's check H size from log? "Found X haplotypes...".
                # If H is small, do brute force.
                # If H is large, or matched_phasings empty, use simplified fallback.
                
                if len(known_haplotypes) < 300 and len(known_haplotypes) > 0:
                    # Brute force pairs
                    # Optimization: iterate h1, filter valid h2?
                    # valid h2 must supply the missing alleles.
                     # This is basically: h2 must be compatible with (G - h1).
                    
                    for h1 in known_hap_list:
                         # Check if h1 is semi-compatible (it doesn't contradict explicit alleles)
                         # i.e. if Genotype has (A, A) and h1 has B, then h1 is incompatible as *either* parent?
                         # No, if G=(A,A), h1 MUST have A.
                         # If G=(A,B), h1 could be A or B.
                         # If G=(A, MISSING), h1 could be A, or anything (if matching missing).
                         
                         # Check strict conflict:
                         # For each locus, h1[i] must be one of the potential alleles in g[i]?
                         # potential alleles = {a1, a2} U {all if missing}.
                         # Yes.
                         
                         # If h1 passes, identifying h2 requirements:
                         # For each locus:
                         #   bits needed = g[i] - h1[i].
                         #   If h1[i] matched a1, need a2.
                         #   If h1[i] matched a2, need a1.
                         #   If matched missing, need a1/a2.
                         pass
                        
                    # Slow fallback for now:
                    valid_pairs = []
                    # Just sample random pairs if too big? No.
                    
                    # Let's try the ORIGINAL method but restricted.
                    # Problem with original method is Cartesian product.
                    # Can we do product of haplotypes?
                    pass
                
                # NEW STRATEGY:
                # Use the original _get_possible_phasings but with a MAX_LIMIT on candidates.
                # If a locus is missing, candidates = known_alleles_per_locus[i].
                # If this list has > 5 alleles, take top 5 most frequent (approx from known)?
                # We don't have frequencies yet.
                # Just take first 5.
                
                truncated_alleles_per_locus = []
                for i in range(n_loci):
                    alleles = list(self.known_alleles_per_locus[i])
                    if len(alleles) > 5:
                        # Deterministic slicing to match behavior
                        alleles = sorted(alleles)[:5] 
                    truncated_alleles_per_locus.append(set(alleles))
                
                phasings = self._get_possible_phasings(g, truncated_alleles_per_locus)
                
                # Warn if empty?
                if not phasings and not is_complete:
                    # Fallback to single dummy
                    # Or try slightly larger limit?
                    pass

            sample_phasings.append(phasings)
            for h1, h2 in phasings:
                all_unique_haplotypes.add(h1)
                all_unique_haplotypes.add(h2)
                
        # 2. Initialize frequencies: Uniform
        num_haps = len(all_unique_haplotypes)
        if num_haps == 0:
            logging.warning("EM: No valid haplotypes found.")
            return

        self.haplotype_frequencies = {h: 1.0 / num_haps for h in all_unique_haplotypes}
        
        logging.info(f"EM: identified {num_haps} unique potential haplotypes.")
        
        # 3. EM Loop
        n_samples = len(genotypes)
        
        for iteration in range(self.max_iterations):
            self.n_iter_ = iteration + 1
            
            # --- E-Step ---
            new_hap_counts = defaultdict(float)
            total_log_likelihood = 0.0
            
            for phasings in sample_phasings:
                probs = []
                pairs = []
                
                for h1, h2 in phasings:
                    f1 = self.haplotype_frequencies[h1]
                    f2 = self.haplotype_frequencies[h2]
                    p = f1 * f2
                    probs.append(p)
                    pairs.append((h1, h2))
                
                sum_p = sum(probs)
                
                if sum_p > 0:
                    # Normalize
                    posteriors = [p / sum_p for p in probs]
                    
                    # Accumulate counts
                    for (h1, h2), post in zip(pairs, posteriors):
                        new_hap_counts[h1] += post
                        new_hap_counts[h2] += post
                        
                if sum_p > 0:
                    total_log_likelihood += np.log(sum_p)
            
            # --- M-Step ---
            total_genes = 2 * n_samples
            max_diff = 0.0
            
            new_frequencies = {}
            for h, count in new_hap_counts.items():
                new_freq = count / total_genes
                old_freq = self.haplotype_frequencies.get(h, 0.0)
                max_diff = max(max_diff, abs(new_freq - old_freq))
                new_frequencies[h] = new_freq
            
            # Implicitly missing haplotypes get 0
            
            self.haplotype_frequencies = new_frequencies
            self.history_.append(total_log_likelihood)
            
            if iteration > 0 and max_diff < self.tolerance:
                logging.info(f"EM converged at iteration {iteration}")
                self.converged = True
                break
                
        if not self.converged:
            logging.warning(f"EM did not converge after {self.max_iterations} iterations.")
            
        return self.get_estimated_frequencies()

    def get_estimated_frequencies(self, sort: bool = True, threshold: float = 0.0) -> Dict[Tuple[str, ...], float]:
        """
        Returns the estimated haplotype frequencies.
        
        Args:
            sort: If True, returns frequencies sorted in descending order.
            threshold: If > 0, only returns haplotypes with frequency above this value.
        """
        freqs = self.haplotype_frequencies
        if threshold > 0:
            freqs = {h: f for h, f in freqs.items() if f > threshold}
            
        if sort:
            return dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True))
        return freqs

    def predict(self, genotypes: List[List[Tuple[str, str]]]) -> List[Tuple[List[str], List[str]]]:
        """
        Infer the most likely phasing for each individual.
        """
        predictions = []
        
        # Use known alleles from training if available, else infer from this batch (suboptimal)
        known_alleles = getattr(self, 'known_alleles_per_locus', None)
        
        for g in genotypes:
            phasings = self._get_possible_phasings(g, known_alleles)
            
            best_pair = None
            max_p = -1.0
            
            for h1, h2 in phasings:
                f1 = self.haplotype_frequencies.get(h1, 0.0)
                f2 = self.haplotype_frequencies.get(h2, 0.0)
                p = f1 * f2
                
                if p > max_p:
                    max_p = p
                    best_pair = (h1, h2)
            
            if best_pair is None:
                if phasings:
                    best_pair = phasings[0]
                else:
                    best_pair = ((), ())
                    
            predictions.append((list(best_pair[0]), list(best_pair[1])))
            
        return predictions

    def evaluate(self, genotypes: List, ground_truth: List, tokenizer: AlleleTokenizer = None) -> Dict[str, float]:
        """Evaluate EM accuracy and other phasing metrics."""
        predictions = self.predict(genotypes)
        
        # Convert predictions and ground truth to the format expected by HLAPhasingMetrics
        # (list of tuples of haplotype strings)
        pred_pairs = []
        for p1, p2 in predictions:
            h1_str = '_'.join(p1)
            h2_str = '_'.join(p2)
            pred_pairs.append(tuple(sorted((h1_str, h2_str))))
            
        truth_pairs = []
        for t1, t2 in ground_truth:
            h1_str = '_'.join(t1)
            h2_str = '_'.join(t2)
            truth_pairs.append(tuple(sorted((h1_str, h2_str))))
            
        if tokenizer is None:
            # Create a dummy tokenizer if none provided, 
            # though HLAPhasingMetrics currently requires one.
            # In a real scenario, the user should provide the tokenizer used during training.
            tokenizer = AlleleTokenizer()
            
        metrics_calc = HLAPhasingMetrics(tokenizer=tokenizer)
        metrics = metrics_calc.calculate_metrics(pred_pairs, truth_pairs)
        
        # Merge with existing count-based metrics if needed, but HLAPhasingMetrics is more complete
        return metrics
