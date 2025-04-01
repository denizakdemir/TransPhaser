import unittest
import torch
import torch.nn as nn
import math # Added for annealing test

# Placeholder for HaplotypeCompatibilityChecker
from transphaser.compatibility import HaplotypeCompatibilityChecker # Reverted to src.
# Placeholder for the classes we are about to create
from transphaser.samplers import GumbelSoftmaxSampler, ConstrainedHaplotypeSampler # Reverted to src.

class TestGumbelSoftmaxSampler(unittest.TestCase):

    def test_initialization(self):
        """Test GumbelSoftmaxSampler initialization."""

        # Default initialization
        sampler_default = GumbelSoftmaxSampler()
        self.assertEqual(sampler_default.initial_temperature, 1.0)
        self.assertEqual(sampler_default.min_temperature, 0.1)
        self.assertEqual(sampler_default.anneal_rate, 0.003)
        self.assertEqual(sampler_default.current_temperature, 1.0) # Starts at initial
        self.assertEqual(sampler_default.step, 0)

        # Initialization with specific values
        sampler_custom = GumbelSoftmaxSampler(
            initial_temperature=2.0,
            min_temperature=0.5,
            anneal_rate=0.01
        )
        self.assertEqual(sampler_custom.initial_temperature, 2.0)
        self.assertEqual(sampler_custom.min_temperature, 0.5)
        self.assertEqual(sampler_custom.anneal_rate, 0.01)
        self.assertEqual(sampler_custom.current_temperature, 2.0)
        self.assertEqual(sampler_custom.step, 0)

    def test_anneal_temperature(self):
        """Test temperature annealing."""
        initial_temp = 1.0
        min_temp = 0.1
        rate = 0.01
        sampler = GumbelSoftmaxSampler(
            initial_temperature=initial_temp,
            min_temperature=min_temp,
            anneal_rate=rate
        )

        temp1 = sampler.current_temperature
        self.assertEqual(temp1, initial_temp)

        # Step 1
        sampler.anneal_temperature()
        temp2 = sampler.current_temperature
        expected_temp2 = max(min_temp, initial_temp * math.exp(-rate * 0)) # Step was 0 before anneal
        # Note: The implementation increments step *after* calculation in anneal_temperature
        # Let's adjust the test to reflect the implementation: step used is self.step *before* increment
        expected_temp2_impl = max(min_temp, initial_temp * math.exp(-rate * 0)) # Step 0 used
        self.assertEqual(sampler.step, 1) # Step incremented
        # After step 0 calculation, temp should still be initial_temp * exp(0) = initial_temp
        self.assertAlmostEqual(temp2, expected_temp2_impl, places=5)
        self.assertAlmostEqual(temp2, initial_temp, places=5) # Should still be initial temp after first call

        # Step 2
        sampler.anneal_temperature()
        temp3 = sampler.current_temperature
        expected_temp3_impl = max(min_temp, initial_temp * math.exp(-rate * 1)) # Step 1 used
        self.assertEqual(sampler.step, 2)
        self.assertLess(temp3, temp2) # Temperature should decrease
        self.assertAlmostEqual(temp3, expected_temp3_impl, places=5)

        # Anneal many steps until min_temp is reached
        max_steps = int(math.log(min_temp / initial_temp) / -rate) + 5 # Steps needed + buffer
        for _ in range(max_steps):
            sampler.anneal_temperature()

        self.assertAlmostEqual(sampler.current_temperature, min_temp, places=5)
        # Check it doesn't go below min_temp
        sampler.anneal_temperature()
        self.assertAlmostEqual(sampler.current_temperature, min_temp, places=5)


    def test_sample(self):
        """Test Gumbel-Softmax sampling output properties."""
        sampler = GumbelSoftmaxSampler()
        batch_size = 4
        num_categories = 5
        logits = torch.randn(batch_size, num_categories)

        # Test soft sampling
        y_soft = sampler.sample(logits, hard=False)
        self.assertEqual(y_soft.shape, (batch_size, num_categories))
        # Check if probabilities sum to 1 (approximately)
        self.assertTrue(torch.allclose(y_soft.sum(dim=-1), torch.ones(batch_size)))
        # Check if values are between 0 and 1
        self.assertTrue(torch.all(y_soft >= 0))
        self.assertTrue(torch.all(y_soft <= 1))

        # Test hard sampling (Straight-Through)
        y_hard = sampler.sample(logits, hard=True)
        self.assertEqual(y_hard.shape, (batch_size, num_categories))
        # Check if it's one-hot
        self.assertTrue(torch.allclose(y_hard.sum(dim=-1), torch.ones(batch_size)))
        # Check that each row contains only 0s and a single 1
        is_one_hot = torch.all((y_hard == 0) | (y_hard == 1)) and torch.all(y_hard.sum(dim=-1) == 1)
        self.assertTrue(is_one_hot)
        # Removed incorrect check: The argmax of the hard sample is based on logits + noise,
        # so it won't necessarily match the argmax of the original logits.
        # self.assertTrue(torch.all(torch.argmax(y_hard, dim=-1) == torch.argmax(logits, dim=-1)))


class TestConstrainedHaplotypeSampler(unittest.TestCase):

    def test_initialization(self):
        """Test ConstrainedHaplotypeSampler initialization."""
        mock_checker = HaplotypeCompatibilityChecker() # Use real checker instance

        sampler = ConstrainedHaplotypeSampler(compatibility_checker=mock_checker)

        # Check if checker is stored
        self.assertIs(sampler.compatibility_checker, mock_checker)


if __name__ == '__main__':
    unittest.main()
