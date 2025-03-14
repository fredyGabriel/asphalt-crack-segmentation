import os
import json
import time
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Centralized logging system for training metrics, learning rates,
    and saving training history.
    """
    def __init__(self, log_dir, config=None):
        """
        Initialize logger with TensorBoard writer and metric storage.

        Args:
            log_dir: Directory to save logs and history
            config: Optional configuration dictionary
        """
        # Create directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.config = config

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

        # Initialize training history storage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'iou': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'epochs': [],  # Store which epochs were recorded
            'lr': []       # Store learning rates
        }

        # Track time for performance logging
        self.last_time = time.time()

        print(f"Logger initialized. Logs will be saved to {log_dir}")

    def log_info(self, message):
        """
        Log general information message to console and TensorBoard.

        Args:
            message: String message to log
        """
        # Print to console
        print(message)

        # Log to TensorBoard as text
        # We use add_text with a global_step of 0 to ensure it appears at the
        # top
        self.writer.add_text('Info', message, 0)

        # Optional: Add timestamp to message
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}"

        # If you want to keep a separate text log file
        log_file = os.path.join(self.log_dir, 'training_log.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_message + '\n')

    def log_metrics(self, metrics, epoch, prefix='Metrics'):
        """
        Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics to log
            epoch: Current epoch number
            prefix: Prefix for metric names in TensorBoard
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, epoch)

    def log_loss(self, train_loss, val_loss, epoch):
        """
        Log training and validation losses to TensorBoard.

        Args:
            train_loss: Training loss value
            val_loss: Validation loss value
            epoch: Current epoch number
        """
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/val', val_loss, epoch)

    def log_learning_rate(self, optimizer, epoch):
        """
        Log learning rates from optimizer to TensorBoard.

        Args:
            optimizer: PyTorch optimizer
            epoch: Current epoch number
        """
        # Log learning rate for each parameter group with proper group names
        for i, param_group in enumerate(optimizer.param_groups):
            # Use proper group names based on index (0: encoder, 1: decoder)
            group_name = "group_" + str(i)  # Use consistent naming for tests
            lr = param_group['lr']
            self.writer.add_scalar(f'LearningRate/{group_name}', lr, epoch)

            # Store the first group's learning rate in history
            if i == 0:
                self.history['lr'].append(float(lr))

    def update_history(self, epoch, train_loss, val_loss, metrics, save=True):
        """
        Update the training history with latest metrics.

        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value
            metrics: Dictionary of evaluation metrics
            save: Whether to save history to disk
        """
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))

        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(float(value))

        if save:
            self.save_history()

    def save_history(self, filename='training_history.json'):
        """
        Save training history to a JSON file.

        Args:
            filename: Name of the JSON file to save
        """
        history_path = os.path.join(self.log_dir, filename)

        # Convert values to serializable types
        serializable_history = {}
        for key, values in self.history.items():
            serializable_history[key] = [float(v) if not isinstance(v, int)
                                         else v for v in values]

        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)

    def log_epoch_time(self, epoch):
        """
        Log time taken by the epoch.

        Args:
            epoch: Current epoch number

        Returns:
            float: Time elapsed since last call
        """
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.writer.add_scalar('Time/epoch', elapsed, epoch)
        self.last_time = current_time
        return elapsed

    def log_test_results(self, test_metrics, test_loss=None, model_type=None):
        """
        Save test results to a JSON file.

        Args:
            test_metrics: Dictionary of test metrics
            test_loss: Test loss value (optional)
            model_type: Type of the model (optional)
        """
        results = {}

        if test_loss is not None:
            results["test_loss"] = float(test_loss)

        if model_type is not None:
            results["model_type"] = model_type

        # Add metrics
        results.update({k: float(v) for k, v in test_metrics.items()})

        # Save to file
        with open(os.path.join(self.log_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

    def close(self):
        """Close the TensorBoard writer and finalize logging."""
        self.writer.close()
