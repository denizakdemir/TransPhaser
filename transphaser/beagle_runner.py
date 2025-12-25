import pandas as pd
import numpy as np
import subprocess
import os
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import sys

class BeagleRunner:
    """
    Wrapper for running Beagle 5.x for HLA phasing.
    Converts HLA genotypes (CSV) to VCF, runs Beagle (Java), and parses output VCF back to haplotypes.
    """
    
    def __init__(self, beagle_jar_path: str = "beagle.jar", output_dir: str = "beagle_output"):
        self.beagle_jar_path = beagle_jar_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find java
        self.java_path = self._find_java()
        if not self.java_path:
            logging.warning("Java not found in PATH or Conda environment. Beagle cannot run without Java.")
            self.java_available = False
        else:
            self.java_available = True
            logging.info(f"Using Java at: {self.java_path}")
            
        # Check if beagle jar exists
        if not os.path.exists(self.beagle_jar_path):
            # Try to look in current dir or likely spots
            candidates = [
                "beagle.jar", 
                "beagle.22Jul22.46e.jar", 
                "../beagle.jar",
                "examples/beagle.jar"
            ]
            found = False
            for c in candidates:
                if os.path.exists(c):
                    self.beagle_jar_path = c
                    found = True
                    break
            
            if not found:
                logging.warning(f"Beagle JAR not found at {self.beagle_jar_path}. Please install Beagle 5.4.")
                self.jar_available = False
            else:
                self.jar_available = True
        
        self.locus_allele_map = {} # Stores locus -> {dna_seq: hla_allele}
    
    def _int_to_dna(self, n: int) -> str:
        """Bijective mapping from integer to DNA sequence (A,C,G,T)."""
        mapping = ['A', 'C', 'G', 'T']
        if n < 0: return "N"
        if n < 4: return mapping[n]
        s = []
        while n >= 0:
            n, r = divmod(n, 4)
            s.append(mapping[r])
            n -= 1 # Adjust for 1-based digit position in bijection
        return "".join(reversed(s))
            
    def _find_java(self) -> Optional[str]:
        """Finds the java executable in PATH or Conda env."""
        # 1. Check Conda environment (typically lib/jvm/bin/java for openjdk)
        try:
            conda_java = Path(sys.prefix) / "lib" / "jvm" / "bin" / "java"
            if conda_java.exists():
                return str(conda_java)
        except Exception:
            pass
            
        # 2. Check PATH
        path = shutil.which("java")
        if path:
            return path
            
        return None

            
    def run(self, df_unphased: pd.DataFrame, locus_columns: List[str]) -> pd.DataFrame:
        """
        Run Beagle on the unphased dataframe.
        Returns a DataFrame with 'Haplotype1' and 'Haplotype2' columns (comma-separated strings).
        """
        if not (self.java_available and self.jar_available):
            logging.error("Cannot run Beagle: Java or Beagle JAR missing.")
            return None
        
        # 1. Convert to VCF
        vcf_path = self.output_dir / "input.vcf"
        # Clear map
        self.locus_allele_map = {}
        self._df_to_vcf(df_unphased, locus_columns, vcf_path)
        
        # 2. Run Beagle
        out_prefix = self.output_dir / "beagle_output"
        cmd = [

            self.java_path, 
            "-Xmx4g", 
            "-jar", str(self.beagle_jar_path),
            f"gt={vcf_path}",
            f"out={out_prefix}",
            "nthreads=4"
        ]
        
        logging.info(f"Running Beagle: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # logging.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Beagle failed: {e.stderr}")
            logging.error(f"Beagle stdout: {e.stdout}")
            return None
            
        # 3. Parse output VCF (Beagle adds .vcf.gz)
        out_vcf = Path(str(out_prefix) + ".vcf.gz")
        if not out_vcf.exists():
             logging.error(f"Beagle output file not found: {out_vcf}")
             return None
             
        # Need to handle gzip. But pandas can read gzip. 
        # Actually it's intricate to parse VCF lines properly.
        # We'll use a custom parser since pandas read_csv with comment='#' might miss headers.
        
        return self._vcf_to_haplotypes(out_vcf, locus_columns, len(df_unphased))

    def _df_to_vcf(self, df: pd.DataFrame, locus_columns: List[str], output_path: Path):
        """
        Converts DataFrame to a VCF suitable for Beagle.
        Treats each HLA locus as a variant line.
        Since Beagle handles multiallelic sites, we list all alleles in ALT.
        """
        
        # Header
        header_lines = [
            "##fileformat=VCFv4.2",
            "##source=TransPhaser_BeagleRunner",
            '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join([f"Sample{i}" for i in range(len(df))])
        ]
        
        with open(output_path, "w") as f:
            for line in header_lines:
                f.write(line + "\n")
            
            # Content
            # For each locus
            for idx, locus in enumerate(locus_columns):
                # 1. Collect all unique alleles
                all_alleles = set()
                # Also collect sample genotypes
                sample_gts = []
                
                for _, row in df.iterrows():
                    val = row[locus]
                    try:
                        a1, a2 = val.split('/') # Assuming "A*01:01/A*02:01" format
                    except:
                        # Handle missing or single
                        if pd.isna(val) or val == "MISSING":
                            a1, a2 = ".", "."
                        else:
                            a1, a2 = val, val # Homozygous assumption if single? Or split error.
                            
                    all_alleles.add(a1)
                    all_alleles.add(a2)
                    sample_gts.append((a1, a2))
                
                # Filter out "."
                if "." in all_alleles:
                    all_alleles.remove(".")
                
                # Map alleles to DNA sequences
                # We need a stable sort so mapping is deterministic
                sorted_alleles = sorted(list(all_alleles))
                
                # Map each HLA allele to a unique DNA sequence
                # 0 -> REF, 1..N -> ALT
                dna_map = {} # hla -> dna
                self.locus_allele_map[locus] = {} # dna -> hla
                
                for i, hla_allele in enumerate(sorted_alleles):
                    dna_seq = self._int_to_dna(i)
                    dna_map[hla_allele] = dna_seq
                    self.locus_allele_map[locus][dna_seq] = hla_allele
                
                # Pick REF and ALTs based on DNA sequences
                # First allele is REF
                if not sorted_alleles:
                    ref_dna = "A"
                    alts_dna = ["C"] # Dummy
                else:
                    ref_dna = dna_map[sorted_alleles[0]]
                    # Remaining are variants
                    alts_dna = [dna_map[a] for a in sorted_alleles[1:]]
                
                # Map allele string to VCF index (for GT field construction)
                # REF (sorted_alleles[0]) -> 0
                # Others -> 1, 2...
                allele_to_vcf_idx = {sorted_alleles[0]: 0}
                for i, allele in enumerate(sorted_alleles[1:]):
                     allele_to_vcf_idx[allele] = i + 1
                allele_to_vcf_idx["."] = "."
                
                # Construct ALT string
                alt_str = ",".join(alts_dna) if alts_dna else "."
                
                # Construct Sample GT strings "0/1"
                gt_strs = []
                for (a1, a2) in sample_gts:
                    # Note: a1/a2 are HLA strings
                    idx1 = allele_to_vcf_idx.get(a1, ".")
                    idx2 = allele_to_vcf_idx.get(a2, ".")
                    gt_strs.append(f"{idx1}/{idx2}")
                
                # Write VCF line
                # Chrom "6", Pos idx+1
                chrom = "6" # HLA on chr6
                pos = (idx + 1) * 1000 
                
                line_parts = [
                    chrom, 
                    str(pos), 
                    locus, 
                    ref_dna, 
                    alt_str, 
                    ".", 
                    "PASS", 
                    f"NS={len(df)}", 
                    "GT"
                ] + gt_strs
                
                f.write("\t".join(line_parts) + "\n")

    def _vcf_to_haplotypes(self, vcf_path: Path, locus_columns: List[str], n_samples: int) -> pd.DataFrame:
        """
        Parses phased VCF back to DataFrame.
        """
        import gzip
        
        # Map Locus -> {index: allele_str}
        locus_allele_maps = {} 
        
        # Store phased alleles: {sample_idx: {locus: (h1, h2)}}
        phased_data = defaultdict(dict)
        
        with gzip.open(vcf_path, "rt") as f:
            for line in f:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    # Parse header if needed (sample names)
                    continue
                
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                    
                chrom = parts[0]
                locus = parts[2] # We stored Locus name in ID
                ref = parts[3]
                alt_str = parts[4]
                alts = alt_str.split(",")
                
                # Reconstruct allele map for this locus
                idx_to_allele = {0: ref}
                for i, alt in enumerate(alts):
                    idx_to_allele[i+1] = alt
                idx_to_allele["."] = "MISSING"
                
                # Parse sample GTs
                sample_cols = parts[9:]
                for i, gt_field in enumerate(sample_cols):
                    # Beagle outputs phased as 0|1
                    gt = gt_field.split(":")[0] # take GT part
                    sep = "|" if "|" in gt else "/"
                    
                    try:
                        idx1_s, idx2_s = gt.split(sep)
                        idx1 = int(idx1_s) if idx1_s != "." else "."
                        idx2 = int(idx2_s) if idx2_s != "." else "."
                        
                        a1_dna = idx_to_allele.get(idx1, "MISSING")
                        a2_dna = idx_to_allele.get(idx2, "MISSING")
                        
                        # Decode DNA -> HLA
                        if a1_dna == "MISSING":
                            h1 = "MISSING"
                        else:
                            h1 = self.locus_allele_map[locus].get(a1_dna, a1_dna)
                            
                        if a2_dna == "MISSING":
                            h2 = "MISSING"
                        else:
                            h2 = self.locus_allele_map[locus].get(a2_dna, a2_dna)
                        
                        phased_data[i][locus] = (h1, h2)
                    except Exception as e:
                        logging.warning(f"Failed to parse GT {gt} for locus {locus}: {e}")
                        phased_data[i][locus] = ("MISSING", "MISSING")
        
        # Convert to DataFrame
        h1_list = []
        h2_list = []
        
        for i in range(n_samples):
            # Reconstruct full haplotype strings
            # Order matters: use locus_columns order
            row_h1 = []
            row_h2 = []
            for loc in locus_columns:
                pair = phased_data[i].get(loc, ("MISSING", "MISSING"))
                row_h1.append(pair[0])
                row_h2.append(pair[1])
            
            h1_list.append("_".join(row_h1))
            h2_list.append("_".join(row_h2))
            
        return pd.DataFrame({
            "Haplotype1": h1_list,
            "Haplotype2": h2_list
        })
