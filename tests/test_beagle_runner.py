import pytest
import pandas as pd
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from transphaser.beagle_runner import BeagleRunner

class TestBeagleRunner:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample unphased dataframe."""
        data = {
            "HLA-A": ["A*01:01/A*02:01", "A*03:01/A*03:01"],
            "HLA-B": ["B*07:02/B*08:01", "B*44:02/B*07:02"],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def locus_columns(self):
        return ["HLA-A", "HLA-B"]

    @pytest.fixture
    def mock_beagle_env(self):
        """Mock environment where java and beagle.jar exist."""
        with patch("shutil.which", return_value="/usr/bin/java"), \
             patch("os.path.exists", return_value=True):
            yield

    def test_initialization(self, mock_beagle_env):
        runner = BeagleRunner(beagle_jar_path="beagle.jar")
        assert runner.java_available
        assert runner.jar_available

    def test_initialization_no_java(self):
        with patch("shutil.which", return_value=None):
            runner = BeagleRunner()
            assert not runner.java_available

    def test_df_to_vcf_generation(self, tmp_path, sample_data, locus_columns):
        """Test conversion of DataFrame to VCF format."""
        runner = BeagleRunner(output_dir=str(tmp_path))
        output_vcf = tmp_path / "test.vcf"
        
        runner._df_to_vcf(sample_data, locus_columns, output_vcf)
        
        assert output_vcf.exists()
        content = output_vcf.read_text()
        
        # Check headers
        assert "##fileformat=VCFv4.2" in content
        assert "#CHROM" in content
        assert "HLA-A" in content  # Locus name in ID column
        assert "HLA-B" in content
        
        # Check sample data lines
        lines = content.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 2  # Two loci
        
        for line in data_lines:
            assert len(line.split('\t')) == 11

    @patch("gzip.open")
    def test_vcf_to_haplotypes_parsing(self, mock_gzip, tmp_path):
        """Test parsing of Beagle VCF output back to DataFrame."""
        runner = BeagleRunner(output_dir=str(tmp_path))
        
        vcf_content = [
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample0\tSample1",
            "6\t1000\tHLA-A\tA*01:01\tA*02:01,A*03:01\t.\tPASS\t.\tGT\t0|1\t2|2",
            "6\t2000\tHLA-B\tB*07:02\tB*08:01,B*44:02\t.\tPASS\t.\tGT\t0|1\t2|0"
        ]
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value = vcf_content
        mock_gzip.return_value = mock_file
        
        locus_columns = ["HLA-A", "HLA-B"]
        df = runner._vcf_to_haplotypes(Path("dummy.vcf.gz"), locus_columns, n_samples=2)
        
        assert len(df) == 2
        assert "Haplotype1" in df.columns
        assert df.iloc[0]["Haplotype1"] == "A*01:01,B*07:02"
        assert df.iloc[0]["Haplotype2"] == "A*02:01,B*08:01"
        assert df.iloc[1]["Haplotype1"] == "A*03:01,B*44:02"
        assert df.iloc[1]["Haplotype2"] == "A*03:01,B*07:02"

    @patch("subprocess.run")
    def test_run_integration(self, mock_subprocess, tmp_path, sample_data, locus_columns):
        """Test the full run method with mocks."""
        output_dir = tmp_path / "beagle_out"
        runner = BeagleRunner(output_dir=str(output_dir))
        
        runner.java_available = True
        runner.jar_available = True
        
        out_vcf = output_dir / "beagle_output.vcf.gz"
        out_vcf.parent.mkdir(parents=True, exist_ok=True)
        out_vcf.touch()
        
        with patch.object(runner, "_df_to_vcf") as mock_to_vcf, \
             patch.object(runner, "_vcf_to_haplotypes") as mock_from_vcf:
            
            mock_from_vcf.return_value = pd.DataFrame({"Haplotype1": [], "Haplotype2": []})
            runner.run(sample_data, locus_columns)
            mock_to_vcf.assert_called_once()
            mock_subprocess.assert_called_once()
            mock_from_vcf.assert_called_once()

    def test_df_to_vcf_missing_and_varied(self, tmp_path, locus_columns):
        """Test VCF generation with missing data and different separators."""
        data = {
            "HLA-A": ["MISSING", "A*01:01,A*02:01"],
            "HLA-B": ["B*07:02", float("nan")], # Single allele and NaN
        }
        df = pd.DataFrame(data)
        runner = BeagleRunner(output_dir=str(tmp_path))
        output_vcf = tmp_path / "test_varied.vcf"
        
        runner._df_to_vcf(df, locus_columns, output_vcf)
        
        content = output_vcf.read_text()
        lines = [l for l in content.split('\n') if not l.startswith("#") and l.strip()]
        
        # Line 1: HLA-A
        hla_a_line = lines[0]
        parts_a = hla_a_line.split('\t')
        s0_a = parts_a[9] # Sample 0
        s1_a = parts_a[10] # Sample 1
        
        assert s0_a == "./."
        assert "/" in s1_a and s1_a != "./."
        
        # Line 2: HLA-B
        hla_b_line = lines[1]
        parts_b = hla_b_line.split('\t')
        s0_b = parts_b[9]
        s1_b = parts_b[10]
        
        assert s0_b == "0/0"
        assert s1_b == "./."

    @patch("gzip.open")
    def test_vcf_parsing_multiallelic(self, mock_gzip, tmp_path):
        """Test parsing VCF with many alleles and varied GTs."""
        runner = BeagleRunner(output_dir=str(tmp_path))
        
        # 4 alleles: REF, ALT1, ALT2, ALT3
        vcf_content = [
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2",
            "6\t1000\tHLA-A\tA*01\tA*02,A*03,A*04\t.\tPASS\t.\tGT\t0|1\t2|3\t.|0"
        ]
        
        mock_file = MagicMock()
        mock_file.__enter__.return_value = vcf_content
        mock_gzip.return_value = mock_file
        
        df = runner._vcf_to_haplotypes(Path("dummy.vcf.gz"), ["HLA-A"], n_samples=3)
        
        # S0: 0|1 -> A*01 | A*02
        assert df.iloc[0]["Haplotype1"] == "A*01"
        assert df.iloc[0]["Haplotype2"] == "A*02"
        
        # S1: 2|3 -> A*03 | A*04
        assert df.iloc[1]["Haplotype1"] == "A*03"
        assert df.iloc[1]["Haplotype2"] == "A*04"
        
        # S2: .|0 -> MISSING | A*01
        assert df.iloc[2]["Haplotype1"] == "MISSING"
        assert df.iloc[2]["Haplotype2"] == "A*01"
