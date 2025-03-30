import unittest

# Placeholder for HaplotypeCompatibilityChecker (already exists)
from src.compatibility import HaplotypeCompatibilityChecker
# Placeholder for the class we are about to create
from src.latent_space import HaplotypeSpaceExplorer

class TestHaplotypeSpaceExplorer(unittest.TestCase):

    def test_initialization(self):
        """Test HaplotypeSpaceExplorer initialization."""
        mock_checker = HaplotypeCompatibilityChecker() # Use the real one as a mock for init

        # Default initialization
        explorer_default = HaplotypeSpaceExplorer(compatibility_checker=mock_checker)
        self.assertEqual(explorer_default.sampling_temperature, 1.0)
        self.assertIs(explorer_default.compatibility_checker, mock_checker)

        # Initialization with specific temperature
        explorer_temp = HaplotypeSpaceExplorer(compatibility_checker=mock_checker, sampling_temperature=0.7)
        self.assertEqual(explorer_temp.sampling_temperature, 0.7)
        self.assertIs(explorer_temp.compatibility_checker, mock_checker)

if __name__ == '__main__':
    unittest.main()
