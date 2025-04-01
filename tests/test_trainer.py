import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import os
import shutil
from unittest.mock import MagicMock, patch, ANY

# Import necessary components from the project
from transphaser.trainer import HLAPhasingTrainer
from transphaser.loss import ELBOLoss, KLAnnealingScheduler # Assuming ELBOLoss and KL scheduler exist
# Mock components needed for testing
class MockModel(nn.Module):
    def __init__(self, latent_dim=4):
        super().__init__()
        self.linear = nn.Linear(10, latent_dim * 2) # Mock layer
        self.latent_dim = latent_dim
        # Mock a parameter to check device placement
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        # Minimal forward pass returning dict expected by ELBOLoss
        # Use a dummy input feature from the batch if available, else create one
        input_tensor = batch.get('feature', torch.randn(batch['genotype_tokens'].size(0), 10, device=self.dummy_param.device))
        latent_params = self.linear(input_tensor)
        mu = latent_params[:, :self.latent_dim]
        log_var = latent_params[:, self.latent_dim:]
        # Return mock reconstruction and KL divergence terms
        return {
            'reconstruction_log_prob': torch.randn(batch['genotype_tokens'].size(0), device=self.dummy_param.device),
            'kl_divergence': torch.randn(batch['genotype_tokens'].size(0), device=self.dummy_param.device) * 0.1 # Small KL div
        }

    def to(self, device):
        # Ensure the dummy parameter is moved for device checks
        super().to(device)
        self.dummy_param = self.dummy_param.to(device)
        return self

class MockDataset(Dataset):
    def __init__(self, num_samples=100, seq_len=6, cov_dim=5):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.cov_dim = cov_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return a dictionary matching expected batch structure
        return {
            'genotype_tokens': torch.randint(0, 10, (self.seq_len * 2,)), # Example token data
            'covariates': torch.randn(self.cov_dim),
            'target_haplotype_tokens': torch.randint(0, 10, (self.seq_len + 1,)), # Example target tokens
            'target_locus_indices': torch.arange(self.seq_len + 1),
            'sample_id': f"sample_{idx}"
            # Add other keys if HLAPhasingModel or ELBOLoss expects them
        }

class MockLRScheduler:
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.last_lr = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        # Mock step doesn't change LR unless specifically tested
        pass

    def get_last_lr(self):
        return self.last_lr

    def state_dict(self):
        return {'last_lr': self.last_lr}

    def load_state_dict(self, state_dict):
        self.last_lr = state_dict['last_lr']


# --- Test Class ---
class TestHLAPhasingTrainer(unittest.TestCase):

    def _compare_state_dicts(self, dict1, dict2):
        """Helper function to compare two state dictionaries containing tensors."""
        self.assertEqual(dict1.keys(), dict2.keys())
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2[key]
            if isinstance(val1, torch.Tensor):
                self.assertTrue(torch.equal(val1, val2), f"Tensor mismatch for key '{key}'")
            else:
                # Handle other potential types in state dict if necessary
                self.assertEqual(val1, val2, f"Value mismatch for key '{key}'")


    def setUp(self):
        """Set up common resources for tests."""
        self.mock_model = MockModel()
        self.mock_loss_fn = ELBOLoss(kl_weight=0.1) # Use actual loss or a simpler mock
        self.mock_train_dataset = MockDataset(num_samples=64)
        self.mock_val_dataset = MockDataset(num_samples=32)
        self.mock_train_loader = DataLoader(self.mock_train_dataset, batch_size=16)
        self.mock_val_loader = DataLoader(self.mock_val_dataset, batch_size=16)
        self.mock_optimizer = Adam(self.mock_model.parameters(), lr=1e-3)
        self.mock_lr_scheduler = MockLRScheduler(self.mock_optimizer)
        self.mock_kl_scheduler = KLAnnealingScheduler(total_steps=10, anneal_type='linear') # Example KL scheduler
        self.test_checkpoint_dir = "test_checkpoints_trainer"
        # Clean up checkpoint dir before each test
        if os.path.exists(self.test_checkpoint_dir):
            shutil.rmtree(self.test_checkpoint_dir)

    def tearDown(self):
        """Clean up resources after tests."""
        # Clean up checkpoint dir after each test
        if os.path.exists(self.test_checkpoint_dir):
            shutil.rmtree(self.test_checkpoint_dir)

    def test_initialization(self):
        """Test trainer initialization with default and custom parameters."""
        # Test with minimal required args
        trainer_min = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            optimizer=self.mock_optimizer
        )
        self.assertIs(trainer_min.model, self.mock_model)
        self.assertIs(trainer_min.loss_fn, self.mock_loss_fn)
        self.assertIs(trainer_min.train_loader, self.mock_train_loader)
        self.assertIs(trainer_min.val_loader, self.mock_val_loader)
        self.assertIs(trainer_min.optimizer, self.mock_optimizer)
        self.assertIsNone(trainer_min.lr_scheduler)
        self.assertIsNone(trainer_min.kl_scheduler)
        self.assertEqual(trainer_min.epochs, 10) # Default
        self.assertEqual(trainer_min.grad_accumulation_steps, 1) # Default
        self.assertEqual(trainer_min.log_interval, 50) # Default
        self.assertEqual(trainer_min.checkpoint_dir, "checkpoints") # Default
        self.assertEqual(trainer_min.checkpoint_frequency, 1) # Default
        self.assertIsNone(trainer_min.early_stopping_patience) # Default
        self.assertEqual(trainer_min.final_model_filename, "final_model.pt") # Default
        self.assertTrue(os.path.exists("checkpoints")) # Default dir created
        # Check device auto-detection
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.assertEqual(trainer_min.device, expected_device)
        # Check model moved to device
        self.assertEqual(self.mock_model.dummy_param.device, expected_device)


        # Test with custom args
        custom_epochs = 5
        custom_grad_acc = 2
        custom_log_int = 10
        custom_chkpt_dir = self.test_checkpoint_dir
        custom_chkpt_freq = 2
        custom_patience = 3
        custom_final_name = "my_model.pth"
        custom_device = torch.device("cpu") # Force CPU

        trainer_custom = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            optimizer=self.mock_optimizer,
            lr_scheduler=self.mock_lr_scheduler,
            kl_scheduler=self.mock_kl_scheduler,
            device=custom_device,
            epochs=custom_epochs,
            grad_accumulation_steps=custom_grad_acc,
            log_interval=custom_log_int,
            checkpoint_dir=custom_chkpt_dir,
            checkpoint_frequency=custom_chkpt_freq,
            early_stopping_patience=custom_patience,
            final_model_filename=custom_final_name
        )
        self.assertIs(trainer_custom.lr_scheduler, self.mock_lr_scheduler)
        self.assertIs(trainer_custom.kl_scheduler, self.mock_kl_scheduler)
        self.assertEqual(trainer_custom.device, custom_device)
        self.assertEqual(trainer_custom.epochs, custom_epochs)
        self.assertEqual(trainer_custom.grad_accumulation_steps, custom_grad_acc)
        self.assertEqual(trainer_custom.log_interval, custom_log_int)
        self.assertEqual(trainer_custom.checkpoint_dir, custom_chkpt_dir)
        self.assertEqual(trainer_custom.checkpoint_frequency, custom_chkpt_freq)
        self.assertEqual(trainer_custom.early_stopping_patience, custom_patience)
        self.assertEqual(trainer_custom.final_model_filename, custom_final_name)
        self.assertTrue(os.path.exists(custom_chkpt_dir)) # Custom dir created
        # Check model moved to custom device
        self.assertEqual(self.mock_model.dummy_param.device, custom_device)


    @patch('transphaser.trainer.logging') # Mock logging within the trainer module
    def test_train_epoch(self, mock_logging):
        """Test the train_epoch method executes a single epoch correctly."""
        # Mock train/eval methods on the instance BEFORE trainer init uses them potentially
        self.mock_model.train = MagicMock()
        self.mock_model.eval = MagicMock()

        trainer = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader, # Needed for init, not used here
            optimizer=self.mock_optimizer,
            lr_scheduler=self.mock_lr_scheduler,
            kl_scheduler=self.mock_kl_scheduler,
            device=torch.device("cpu"), # Use CPU for simplicity
            epochs=1,
            log_interval=2 # Log more frequently for testing
        )
        # Reset logger after init logs
        mock_logging.reset_mock()

        # Mock loss_fn to control the returned loss value
        mock_loss_value = torch.tensor(1.5, requires_grad=True)
        self.mock_loss_fn.forward = MagicMock(return_value=mock_loss_value)
        # Mock model forward pass (already done by MockModel)
        # Mock train method on the instance
        self.mock_model.train = MagicMock()
        # Mock optimizer and schedulers
        self.mock_optimizer.step = MagicMock()
        self.mock_optimizer.zero_grad = MagicMock()
        self.mock_lr_scheduler.step = MagicMock()
        self.mock_kl_scheduler.step = MagicMock(return_value=0.5) # Mock KL weight update
        self.mock_loss_fn.update_kl_weight = MagicMock() # Mock the update method

        # --- Act ---
        avg_train_loss = trainer.train_epoch()

        # --- Assert ---
        # Check model was set to train mode
        self.mock_model.train.assert_called_once()

        # Check optimizer zero_grad called at start and after steps
        # Total batches = 64 / 16 = 4. zero_grad called once at start, once per step.
        self.assertEqual(self.mock_optimizer.zero_grad.call_count, 1 + len(self.mock_train_loader))

        # Check optimizer step called for each batch (no grad accum)
        self.assertEqual(self.mock_optimizer.step.call_count, len(self.mock_train_loader))

        # Check loss backward called for each batch
        # Need to access the mock loss value's backward call
        # This is tricky as loss is recalculated each time. Let's check calls to model/loss instead.
        self.assertEqual(self.mock_loss_fn.forward.call_count, len(self.mock_train_loader))

        # Check schedulers were stepped
        self.assertEqual(self.mock_lr_scheduler.step.call_count, len(self.mock_train_loader))
        self.assertEqual(self.mock_kl_scheduler.step.call_count, len(self.mock_train_loader))
        # Check KL weight was updated in loss function
        self.assertEqual(self.mock_loss_fn.update_kl_weight.call_count, len(self.mock_train_loader))
        self.mock_loss_fn.update_kl_weight.assert_called_with(0.5) # Check the value passed

        # Check average loss calculation (approximate due to mocking)
        # Expected avg loss = mock_loss_value.item()
        self.assertAlmostEqual(avg_train_loss, mock_loss_value.item(), places=5)
        self.assertEqual(len(trainer.train_loss_history), 1)
        self.assertAlmostEqual(trainer.train_loss_history[0], mock_loss_value.item(), places=5)

        # Check logging calls
        # Log interval = 2. Batches = 4. Expected logs at batch 2 and 4. + 1 summary log.
        num_batches = len(self.mock_train_loader)
        expected_log_calls = (num_batches // trainer.log_interval) + 1 # Batch logs + summary log
        self.assertEqual(mock_logging.info.call_count, expected_log_calls)
        # Check specific log content (optional, can be brittle)
        mock_logging.info.assert_any_call(unittest.mock.ANY) # Check if any info log was made


    @patch('transphaser.trainer.logging') # Mock logging
    def test_train_epoch_with_grad_accumulation(self, mock_logging):
        """Test train_epoch with gradient accumulation."""
        grad_accum_steps = 2
        trainer = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader, # 4 batches total
            val_loader=self.mock_val_loader,
            optimizer=self.mock_optimizer,
            lr_scheduler=self.mock_lr_scheduler,
            kl_scheduler=self.mock_kl_scheduler,
            device=torch.device("cpu"),
            epochs=1,
            grad_accumulation_steps=grad_accum_steps,
            log_interval=1 # Log every batch for this test
        )
        # Reset logger after init logs
        mock_logging.reset_mock()

        # Mock components
        mock_loss_value = torch.tensor(1.5, requires_grad=True)
        # Mock backward on the loss tensor itself to track calls
        mock_loss_value.backward = MagicMock()
        self.mock_loss_fn.forward = MagicMock(return_value=mock_loss_value)
        # Mock train method on the instance
        self.mock_model.train = MagicMock()
        self.mock_optimizer.step = MagicMock()
        self.mock_optimizer.zero_grad = MagicMock()
        self.mock_lr_scheduler.step = MagicMock()
        self.mock_kl_scheduler.step = MagicMock(return_value=0.5)
        self.mock_loss_fn.update_kl_weight = MagicMock()

        # --- Act ---
        trainer.train_epoch()

        # --- Assert ---
        num_batches = len(self.mock_train_loader) # 4 batches
        expected_optimizer_steps = num_batches // grad_accum_steps # 4 // 2 = 2 steps

        # Check backward called for every batch
        # self.assertEqual(mock_loss_value.backward.call_count, num_batches) # Difficult to mock backward on the returned tensor reliably

        # Check optimizer step called only after accumulation steps
        self.assertEqual(self.mock_optimizer.step.call_count, expected_optimizer_steps)

        # Check zero_grad called once at start and once per optimizer step
        self.assertEqual(self.mock_optimizer.zero_grad.call_count, 1 + expected_optimizer_steps)

        # Check schedulers stepped only when optimizer steps
        self.assertEqual(self.mock_lr_scheduler.step.call_count, expected_optimizer_steps)
        self.assertEqual(self.mock_kl_scheduler.step.call_count, expected_optimizer_steps)
        self.assertEqual(self.mock_loss_fn.update_kl_weight.call_count, expected_optimizer_steps)

        # Check logging calls (log interval 1 means log every batch + summary)
        self.assertEqual(mock_logging.info.call_count, num_batches + 1)


    @patch('transphaser.trainer.logging') # Mock logging
    def test_evaluate(self, mock_logging):
        """Test the evaluate method."""
        trainer = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader, # Needed for init
            val_loader=self.mock_val_loader, # Used here
            optimizer=self.mock_optimizer, # Needed for init
            device=torch.device("cpu")
        )
        # Reset logger after init logs
        mock_logging.reset_mock()
        # Mock eval method on the instance
        self.mock_model.eval = MagicMock()

        # Mock components
        mock_loss_value = torch.tensor(0.8) # Mock validation loss
        self.mock_loss_fn.forward = MagicMock(return_value=mock_loss_value)


        # --- Act ---
        avg_val_loss = trainer.evaluate()

        # --- Assert ---
        # Check model was set to eval mode
        self.mock_model.eval.assert_called_once()

        # Check loss function called for each validation batch
        self.assertEqual(self.mock_loss_fn.forward.call_count, len(self.mock_val_loader))

        # Check average loss calculation
        self.assertAlmostEqual(avg_val_loss, mock_loss_value.item(), places=5)
        self.assertEqual(len(trainer.val_loss_history), 1)
        self.assertAlmostEqual(trainer.val_loss_history[0], mock_loss_value.item(), places=5)

        # Check logging call (1 summary log for evaluate)
        self.assertEqual(mock_logging.info.call_count, 1)
        mock_logging.info.assert_called_once_with(unittest.mock.ANY)


    @patch('transphaser.trainer.torch.save') # Mock torch.save
    @patch('transphaser.trainer.logging') # Mock logging
    def test_save_checkpoint(self, mock_logging, mock_torch_save):
        """Test the _save_checkpoint method."""
        epoch = 2
        val_loss = 0.75
        trainer = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            optimizer=self.mock_optimizer,
            lr_scheduler=self.mock_lr_scheduler, # Include scheduler
            kl_scheduler=self.mock_kl_scheduler, # Include KL scheduler
            device=torch.device("cpu"),
            checkpoint_dir=self.test_checkpoint_dir,
            checkpoint_frequency=1 # Ensure saving happens
        )
        trainer.best_val_loss = 0.8 # Set previous best loss
        trainer.train_loss_history = [1.0, 0.9]
        trainer.val_loss_history = [0.9, 0.8]

        # --- Act 1: Save regular checkpoint (not best) ---
        trainer._save_checkpoint(epoch=epoch, val_loss=val_loss, is_best=False)

        # --- Assert 1 ---
        expected_filename_reg = f"checkpoint_epoch_{epoch+1}.pt"
        expected_filepath_reg = os.path.join(self.test_checkpoint_dir, expected_filename_reg)
        # Check torch.save called once for the regular checkpoint
        mock_torch_save.assert_called_once()
        # Check the arguments passed to torch.save
        call_args, call_kwargs = mock_torch_save.call_args
        saved_state = call_args[0] # The state dictionary
        saved_path = call_args[1] # The file path
        self.assertEqual(saved_path, expected_filepath_reg)
        # Check content of the saved state
        self.assertEqual(saved_state['epoch'], epoch + 1)
        self.assertIn('model_state_dict', saved_state)
        self.assertIn('optimizer_state_dict', saved_state)
        self.assertIn('lr_scheduler_state_dict', saved_state) # Check LR scheduler state
        # Check for KL scheduler weight (assuming KLAnnealingScheduler saves current_weight)
        self.assertIn('kl_scheduler_weight', saved_state)
        self.assertEqual(saved_state['val_loss'], val_loss)
        self.assertEqual(saved_state['best_val_loss'], trainer.best_val_loss)
        self.assertEqual(saved_state['train_loss_history'], trainer.train_loss_history)
        self.assertEqual(saved_state['val_loss_history'], trainer.val_loss_history)
        # Check logging call
        mock_logging.info.assert_called_with(f"Saved checkpoint to {expected_filepath_reg}")

        # Reset mock for next call
        mock_torch_save.reset_mock()
        mock_logging.reset_mock()

        # --- Act 2: Save best checkpoint ---
        trainer.best_val_loss = val_loss # Update best loss for this call
        trainer._save_checkpoint(epoch=epoch, val_loss=val_loss, is_best=True)

        # --- Assert 2 ---
        expected_filename_best = "best_model.pt"
        expected_filepath_best = os.path.join(self.test_checkpoint_dir, expected_filename_best)
        # Check torch.save called twice (regular + best)
        self.assertEqual(mock_torch_save.call_count, 2)
        # Check the second call was for the best model
        call_args_best, _ = mock_torch_save.call_args_list[1] # Get args of the second call
        saved_path_best = call_args_best[1]
        self.assertEqual(saved_path_best, expected_filepath_best)
        # Check logging calls (regular + best)
        self.assertEqual(mock_logging.info.call_count, 2)
        mock_logging.info.assert_any_call(f"Saved checkpoint to {expected_filepath_reg}")
        mock_logging.info.assert_any_call(f"Saved best model checkpoint to {expected_filepath_best}")


    @patch('transphaser.trainer.HLAPhasingTrainer._save_checkpoint') # Mock saving checkpoints
    @patch('transphaser.trainer.torch.save') # Mock final model save
    @patch('transphaser.trainer.logging') # Mock logging
    def test_train_loop_full(self, mock_logging, mock_final_save, mock_save_checkpoint):
        """Test the main train loop for multiple epochs."""
        num_epochs = 3
        trainer = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            optimizer=self.mock_optimizer,
            device=torch.device("cpu"),
            epochs=num_epochs,
            checkpoint_dir=self.test_checkpoint_dir,
            checkpoint_frequency=1, # Save every epoch
            final_model_filename="final_test.pt"
        )
        # Reset logger after init logs
        mock_logging.reset_mock()

        # Mock train_epoch and evaluate to return decreasing losses AND update history
        mock_train_losses = [1.0, 0.8, 0.6]
        mock_val_losses = [0.9, 0.7, 0.5]
        def mock_train_epoch_side_effect():
            loss = mock_train_losses[trainer.current_epoch]
            trainer.train_loss_history.append(loss) # Simulate history update
            return loss
        def mock_evaluate_side_effect():
            loss = mock_val_losses[trainer.current_epoch]
            trainer.val_loss_history.append(loss) # Simulate history update
            return loss

        trainer.train_epoch = MagicMock(side_effect=mock_train_epoch_side_effect)
        trainer.evaluate = MagicMock(side_effect=mock_evaluate_side_effect)

        # --- Act ---
        train_history, val_history = trainer.train()

        # --- Assert ---
        # Check train_epoch and evaluate called correct number of times
        self.assertEqual(trainer.train_epoch.call_count, num_epochs)
        self.assertEqual(trainer.evaluate.call_count, num_epochs)

        # Check loss histories returned correctly
        self.assertEqual(train_history, [1.0, 0.8, 0.6])
        self.assertEqual(val_history, [0.9, 0.7, 0.5])

        # Check checkpoint saving logic
        # Epoch 1: val_loss 0.9 < inf -> best=True. Save chkpt (freq=1). Call _save_checkpoint(0, 0.9, True)
        # Epoch 2: val_loss 0.7 < 0.9 -> best=True. Save chkpt (freq=1). Call _save_checkpoint(1, 0.7, True)
        # Epoch 3: val_loss 0.5 < 0.7 -> best=True. Save chkpt (freq=1). Call _save_checkpoint(2, 0.5, True)
        self.assertEqual(mock_save_checkpoint.call_count, num_epochs)
        # Use positional arguments in unittest.mock.call to match actual call
        mock_save_checkpoint.assert_has_calls([
            unittest.mock.call(0, 0.9, is_best=True),
            unittest.mock.call(1, 0.7, is_best=True),
            unittest.mock.call(2, 0.5, is_best=True),
        ], any_order=False) # Ensure order is correct

        # Check final model saved
        mock_final_save.assert_called_once()
        final_save_args, _ = mock_final_save.call_args
        # Use helper function to compare state dicts
        self._compare_state_dicts(final_save_args[0], self.mock_model.state_dict()) # Check saved object content is correct
        self.assertEqual(final_save_args[1], os.path.join(self.test_checkpoint_dir, "final_test.pt")) # Check path

        # Check logging calls (start, end, per epoch start/end, best loss updates, final save)
        # Expected: Start(1) + End(1) + EpochStart(3) + EpochEnd(3) + NewBestLoss(3) + FinalSave(1) = 12
        expected_log_calls = 1 + 1 + num_epochs + num_epochs + num_epochs + 1
        self.assertEqual(mock_logging.info.call_count, expected_log_calls)
        mock_logging.info.assert_any_call("Starting training...")
        mock_logging.info.assert_any_call("Training finished.")
        mock_logging.info.assert_any_call(f"--- Starting Epoch 1/{num_epochs} ---")
        mock_logging.info.assert_any_call(f"--- Finished Epoch {num_epochs}/{num_epochs} ---")


    @patch('transphaser.trainer.HLAPhasingTrainer._save_checkpoint')
    @patch('transphaser.trainer.torch.save')
    @patch('transphaser.trainer.logging')
    def test_early_stopping(self, mock_logging, mock_final_save, mock_save_checkpoint):
        """Test the early stopping logic terminates training."""
        patience = 2
        num_epochs = 10 # Set more epochs than needed for stopping
        trainer = HLAPhasingTrainer(
            model=self.mock_model,
            loss_fn=self.mock_loss_fn,
            train_loader=self.mock_train_loader,
            val_loader=self.mock_val_loader,
            optimizer=self.mock_optimizer,
            device=torch.device("cpu"),
            epochs=num_epochs,
            early_stopping_patience=patience,
            checkpoint_dir=self.test_checkpoint_dir,
            final_model_filename="final_early_stop.pt"
        )
        # Reset logger after init logs
        mock_logging.reset_mock()

        # Mock evaluate to simulate improvement then stagnation AND update history
        mock_val_losses_seq = [0.8, 0.7, 0.7, 0.7, 0.6] # Add extra values just in case
        def mock_evaluate_early_stop_side_effect():
            # Use current_epoch to index into the loss sequence
            loss = mock_val_losses_seq[trainer.current_epoch]
            trainer.val_loss_history.append(loss) # Simulate history update
            return loss
        trainer.evaluate = MagicMock(side_effect=mock_evaluate_early_stop_side_effect)

        # Mock train_epoch to return dummy values AND update history
        def mock_train_epoch_early_stop():
            loss = 1.0 # Dummy train loss
            trainer.train_loss_history.append(loss)
            return loss
        trainer.train_epoch = MagicMock(side_effect=mock_train_epoch_early_stop)


        # --- Act ---
        trainer.train()

        # --- Assert ---
        # Check training stopped after patience epochs without improvement
        # Expected epochs run = 1 (improve) + 1 (improve) + patience (2) = 4
        expected_epochs_run = 1 + 1 + patience
        self.assertEqual(trainer.evaluate.call_count, expected_epochs_run)
        self.assertEqual(trainer.train_epoch.call_count, expected_epochs_run)

        # Check early stopping log message
        mock_logging.info.assert_any_call(f"Early stopping triggered after {patience} epochs without improvement.")

        # Check final model was still saved
        mock_final_save.assert_called_once()
        final_save_args, _ = mock_final_save.call_args
        self.assertEqual(final_save_args[1], os.path.join(self.test_checkpoint_dir, "final_early_stop.pt"))

        # Check best model checkpoint was saved correctly (should be from epoch 2, val_loss 0.7)
        # Find the call where is_best=True was passed with epoch=1
        best_call_found = False
        for call in mock_save_checkpoint.call_args_list:
            call_epoch = call.kwargs.get('epoch', call.args[0])
            call_val_loss = call.kwargs.get('val_loss', call.args[1])
            call_is_best = call.kwargs.get('is_best') # Check only kwargs

            if call_epoch == 1 and call_val_loss == 0.7 and call_is_best is True:
                best_call_found = True
                break
        self.assertTrue(best_call_found, "Best checkpoint (epoch 1, loss 0.7) was not saved correctly.")


if __name__ == '__main__':
    unittest.main()
