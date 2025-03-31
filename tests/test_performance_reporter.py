import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock # For mocking file I/O and plotting

# Assuming the class will be created in src/performance_reporter.py
from src.performance_reporter import PerformanceReporter

class TestPerformanceReporter(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test if the reporter initializes correctly."""
        reporter = PerformanceReporter(output_dir=self.test_dir)
        self.assertEqual(reporter.output_dir, self.test_dir)
        self.assertIsNotNone(reporter.metrics_log) # Check if metrics log is initialized

    def test_log_metric_and_generate_report_json(self):
        """Test logging metrics and generating a JSON report."""
        reporter = PerformanceReporter(output_dir=self.test_dir)
        reporter.log_metric("epoch_1", {"loss": 0.5, "accuracy": 0.8})
        reporter.log_metric("epoch_2", {"loss": 0.4, "accuracy": 0.85})
        reporter.log_metric("summary", {"best_accuracy": 0.85, "total_time": 120.5})

        report_path = os.path.join(self.test_dir, "performance_report.json")
        reporter.generate_report(report_format="json", report_filename="performance_report.json")

        self.assertTrue(os.path.exists(report_path))

        with open(report_path, 'r') as f:
            report_data = json.load(f)

        expected_data = {
            "epoch_1": {"loss": 0.5, "accuracy": 0.8},
            "epoch_2": {"loss": 0.4, "accuracy": 0.85},
            "summary": {"best_accuracy": 0.85, "total_time": 120.5}
        }
        self.assertEqual(report_data, expected_data)

    def test_generate_report_txt(self):
        """Test generating a simple text report."""
        reporter = PerformanceReporter(output_dir=self.test_dir)
        reporter.log_metric("epoch_1", {"loss": 0.5, "accuracy": 0.8})
        reporter.log_metric("summary", {"best_accuracy": 0.85})

        report_path = os.path.join(self.test_dir, "report.txt")
        reporter.generate_report(report_format="txt", report_filename="report.txt")

        self.assertTrue(os.path.exists(report_path))

        with open(report_path, 'r') as f:
            content = f.read()
            # Basic check for content existence
            self.assertIn("epoch_1", content)
            self.assertIn("loss: 0.5", content) # Removed single quotes around key
            self.assertIn("summary", content)
            self.assertIn("best_accuracy: 0.85", content) # Removed single quotes around key

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.xlabel')
    @patch('matplotlib.pyplot.ylabel')
    @patch('matplotlib.pyplot.title')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.grid')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_training_curves_mocked(self, mock_close, mock_savefig, mock_grid, mock_legend, mock_title, mock_ylabel, mock_xlabel, mock_plot, mock_figure):
        """Test plotting training curves with mocked matplotlib."""
        reporter = PerformanceReporter(output_dir=self.test_dir)
        # Log some sample training history (e.g., list of losses per epoch)
        reporter.log_metric("training_history", {"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.2, 0.9, 0.7]})

        plot_path = os.path.join(self.test_dir, "training_curves.png")
        reporter.plot_training_curves(plot_filename="training_curves.png")

        # Assert that plotting functions were called
        mock_figure.assert_called_once()
        self.assertEqual(mock_plot.call_count, 2) # Called for train_loss and val_loss
        mock_xlabel.assert_called_once_with("Epoch")
        mock_ylabel.assert_called_once_with("Loss")
        mock_title.assert_called_once()
        mock_legend.assert_called_once()
        mock_grid.assert_called_once_with(True)
        mock_savefig.assert_called_once_with(plot_path)
        mock_close.assert_called_once()

    def test_unsupported_report_format(self):
        """Test handling of unsupported report formats."""
        reporter = PerformanceReporter(output_dir=self.test_dir)
        with self.assertRaises(ValueError):
            reporter.generate_report(report_format="xml") # Assuming xml is not supported


if __name__ == '__main__':
    unittest.main()
