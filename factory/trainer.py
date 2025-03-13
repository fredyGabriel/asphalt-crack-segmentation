import os
import time
import torch

from factory.train_one import train_one_epoch
from validators.evaluate import validate, calculate_metrics
from utils.logger import Logger


class Trainer:
    """
    Trainer class that manages the training process including metrics tracking,
    checkpointing, early stopping, and scheduling.
    """
    def __init__(self, model_manager, train_loader, val_loader, criterion,
                 optimizer, scheduler, config, device):
        """
        Initialize the trainer with all necessary components.

        Args:
            model_manager: ModelManager instance containing the model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Configuration dictionary
            device: Device to run training on
        """
        # Store all required components
        self.model_manager = model_manager
        self.model = model_manager.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.model_type = model_manager.model_type

        # Initialize logger
        self.logger = Logger(log_dir=config['model']['save_path'],
                             config=config)

        # Configure mixed precision
        self.use_mixed_precision = config.get('training', {}
                                              ).get('mixed_precision', True)
        self.scaler = torch.amp.GradScaler() if self.use_mixed_precision and\
            device.type == 'cuda' else None

        # Configure early stopping
        self.early_stopping_patience = config.get(
            'training', {}).get('early_stopping_patience', 10)
        self.early_stopping_counter = 0

        # Other training parameters
        self.epochs = config['training']['num_epochs']
        self.accumulation_steps = config.get('training', {}
                                             ).get('accumulation_steps', 4)
        self.save_frequency = config.get('training', {}).get('save_frequency',
                                                             5)

    def train(self):
        """
        Run the full training loop for the specified number of epochs.

        Returns:
            dict: Training history
        """
        start_time = time.time()

        for epoch in range(self.epochs):
            # Clear CUDA cache at the beginning of each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Train for one epoch
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                self.scaler,
                accumulation_steps=self.accumulation_steps
            )

            # Validate
            val_loss = validate(self.model, self.val_loader, self.criterion,
                                self.device)

            # Calculate metrics
            metrics = calculate_metrics(self.model, self.val_loader,
                                        self.device, self.config)

            # Check if we should unfreeze next stage
            self.model_manager.unfreeze_next_stage_if_needed(metrics['iou'])

            # Save periodic checkpoints and best models
            es_count, improved = self.model_manager.save_progress(
                epoch,
                train_loss,
                val_loss,
                metrics,
                self.optimizer,
                self.scheduler
            )

            # Update early stopping counter
            if improved:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += es_count

            # Check for early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Determine if we should save metrics in this epoch
            save_epoch = (epoch % self.save_frequency == 0) or (
                epoch == self.epochs - 1)

            # Display progress
            print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: \
{train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, IoU: {metrics["iou"]:.4f}, F1: \
{metrics["f1"]:.4f}' f'{" - Saving metrics" if save_epoch else ""}')

            # Log metrics with our logger
            self.logger.log_loss(train_loss, val_loss, epoch)
            self.logger.log_metrics(metrics, epoch)
            self.logger.log_learning_rate(self.optimizer, epoch)
            epoch_time = self.logger.log_epoch_time(epoch)

            # Save metrics to history if needed
            if save_epoch:
                self.logger.update_history(
                    epoch=epoch + 1,  # 1-indexed for history
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=metrics
                )

            # Update learning rate based on validation loss
            if isinstance(self.scheduler,
                          torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Register current learning rates in TensorBoard
            if hasattr(self, 'writer') and self.writer:
                for i, group in enumerate(self.optimizer.param_groups):
                    group_name = "encoder" if i == 0 and len(
                        self.optimizer.param_groups) > 1 else "decoder"
                    self.writer.add_scalar(f'LearningRate/\
{group_name}', group['lr'], epoch)

            print(f"Epoch time: {epoch_time:.2f}s")

        # Save final history
        self.logger.save_history()
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s. History and models \
saved.")
        return self.logger.history

    def evaluate(self, test_loader):
        """
        Evaluate the model on the test dataset.

        Args:
            test_loader: DataLoader for test data

        Returns:
            tuple: (test_metrics, test_loss)
        """
        print("\nEvaluating on test set...")

        # Load best model by IoU for evaluation
        best_model_path = os.path.join(self.config['model']['save_path'],
                                       'best_iou_model.pth')
        checkpoint = self.model_manager.load_checkpoint(best_model_path)
        print(f"Loaded best model from epoch {checkpoint.get('epoch',
                                                             'unknown')}")

        # Calculate metrics and loss on test set
        test_metrics = calculate_metrics(self.model, test_loader, self.device,
                                         self.config)
        test_loss = validate(self.model, test_loader, self.criterion,
                             self.device)

        # Print test results
        print("\nFinal evaluation on test set:")
        print(f"Test Loss: {test_loss:.4f}")
        for metric, value in test_metrics.items():
            print(f"Test {metric}: {value:.4f}")

        # Log test results
        self.logger.log_test_results(test_metrics, test_loss, self.model_type)

        # Close the logger
        self.logger.close()

        return test_metrics, test_loss
