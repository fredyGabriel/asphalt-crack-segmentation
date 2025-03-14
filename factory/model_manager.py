import os
import torch
from models.model_factory import get_model
from factory.custom_schedulers import WarmupCosineAnnealingLR


class ModelManager:
    """
    Manages model creation, checkpoint operations, and gradual unfreezing
    for training deep learning models.
    """
    def __init__(self, config, device, logger=None):
        """
        Initialize the model manager.

        Args:
            config (dict): Configuration dictionary
            device (torch.device): Device to run the model on
            logger (Logger, optional): Logger for output messages
        """
        self.config = config
        self.device = device
        self.model = None
        self.model_type = config['model'].get('type', 'unet').lower()
        self.save_path = config['model']['save_path']
        self.logger = logger

        # Create directory for model checkpoints
        os.makedirs(self.save_path, exist_ok=True)

        # Tracking best model performance
        self.best_val_loss = float('inf')
        self.best_iou = 0.0

    def log(self, message):
        """Helper method to log messages using logger if available, or print \
otherwise"""
        if self.logger:
            self.logger.log_info(message)
        else:
            print(message)

    def create_model(self):
        """
        Create and return a model based on configuration.

        Returns:
            torch.nn.Module: The created model
        """
        # Create model using the factory
        # Add debug information for ASPP rates if using Swin UNet
        if self.model_type == 'swin2_unet' and 'swin2_unet' in\
                self.config['model']:
            swin_config = self.config['model']['swin2_unet']
            if 'aspp_rates' in swin_config:
                self.log(f"ASPP rates from config: {swin_config['aspp_rates']}"
                         )
            else:
                self.log("Warning: No ASPP rates found in config, will use \
defaults")

        self.model = get_model(self.config, logger=self.logger)
        self.model = self.model.to(self.device)
        self.log(f"Model transferred to {self.device}")
        return self.model

    def setup_optimizer(self, base_lr, weight_decay=1e-4):
        """
        Set up optimizer with different learning rates for encoder and decoder.

        Args:
            base_lr (float): Base learning rate for decoder
            weight_decay (float): Weight decay factor

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if self.model is None:
            raise ValueError("Model must be created before setting up \
optimizer")

        # Separate encoder and decoder parameters
        encoder_params = []
        decoder_params = []

        # Extract encoder and decoder parameters based on model type
        if self.model_type in ["unet_resnet", "swin2_unet"]:
            # For models with explicit encoder component
            encoder_params = list(self.model.encoder.parameters())
            # All other parameters belong to decoder
            decoder_params = [p for p in self.model.parameters()
                              if not any(p is ep for ep in encoder_params)]
        else:
            # For other models without clear separation, use all parameters as
            # decoder
            encoder_params = []
            decoder_params = list(self.model.parameters())

        # Configure learning rates
        encoder_lr_factor = self.config.get('training', {}
                                            ).get('encoder_lr_factor', 0.1)
        encoder_lr = base_lr * encoder_lr_factor
        decoder_lr = base_lr

        self.log(f"Learning rates: encoder={encoder_lr}, decoder={decoder_lr}")

        # Set up parameter groups for optimizer
        param_groups = []

        # Always add encoder group, even if empty (for test compatibility)
        encoder_group = {
            'params': [p for p in encoder_params if p.requires_grad],
            'lr': encoder_lr,
            'weight_decay': weight_decay
        }
        param_groups.append(encoder_group)
        self.log(f"Added encoder parameter group with \
{len(encoder_group['params'])} parameters")

        # Always add decoder group
        decoder_group = {
            'params': [p for p in decoder_params if p.requires_grad],
            'lr': decoder_lr,
            'weight_decay': weight_decay
        }
        param_groups.append(decoder_group)
        self.log(f"Added decoder parameter group with \
{len(decoder_group['params'])} parameters")

        # Create optimizer with parameter groups
        optimizer = torch.optim.Adam(param_groups)
        return optimizer

    def setup_scheduler(self, optimizer):
        """
        Set up learning rate scheduler based on configuration.

        Args:
            optimizer: The optimizer to schedule

        Returns:
            torch.optim.lr_scheduler: Configured scheduler
        """
        scheduler_config = self.config.get('training', {}).get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'ReduceLROnPlateau')

        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 0.00001)
            )
            self.log(f"Created ReduceLROnPlateau scheduler with patience="
                     f"{scheduler_config.get('patience', 5)}")
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.get('T_0', 10),
                T_mult=scheduler_config.get('T_mult', 1),
                eta_min=scheduler_config.get('min_lr', 0.00001)
            )
            self.log(f"Created CosineAnnealingWarmRestarts scheduler with T_0="
                     f"{scheduler_config.get('T_0', 10)}")
        elif scheduler_type == 'WarmupCosineAnnealing':
            # Get configuration parameters with appropriate defaults
            warmup_epochs = scheduler_config.get('warmup_epochs', 5)
            total_epochs = self.config['training'].get('num_epochs', 100)
            min_lr_decoder = scheduler_config.get('min_lr_decoder', 0.00001)
            min_lr_encoder = scheduler_config.get('min_lr_encoder', 0.000001)

            scheduler = WarmupCosineAnnealingLR(
                optimizer,
                T_warmup=warmup_epochs,
                T_max=total_epochs,
                eta_min_decoder=min_lr_decoder,
                eta_min_encoder=min_lr_encoder
            )
            self.log(f"Created WarmupCosineAnnealing scheduler: warmup=\
{warmup_epochs}, "
                     f"total={total_epochs}, min_lr_decoder={min_lr_decoder}, "
                     f"min_lr_encoder={min_lr_encoder}")
        else:
            self.log(f"Warning: Unknown scheduler type '{scheduler_type}', \
using constant learning rate")
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                          lambda epoch: 1.0)

        return scheduler

    def save_checkpoint(self, path, **kwargs):
        """
        Save model checkpoint with additional information.

        Args:
            path (str): Path to save the checkpoint
            **kwargs: Additional information to include in checkpoint
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            **kwargs
        }

        # Save checkpoint
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load model from checkpoint.

        Args:
            path (str): Path to the checkpoint file

        Returns:
            dict: Loaded checkpoint data
        """
        # Check if checkpoint exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Create model if it doesn't exist
        if self.model is None:
            self.model = self.create_model()

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.log(f"Loaded checkpoint: {path}")

        return checkpoint

    def save_regular_checkpoint(self, epoch, optimizer, scheduler, train_loss,
                                val_loss, metrics):
        """Saves a regular checkpoint at specified intervals."""
        save_frequency = self.config.get('training', {}).get('save_frequency',
                                                             5)
        if (epoch % save_frequency == 0) or\
                (epoch == self.config['training']['num_epochs'] - 1):
            checkpoint_path = os.path.join(self.save_path,
                                           f'checkpoint_epoch_{epoch+1}.pth')
            self.save_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict() if scheduler else
                None,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics
            )
            self.log(f"Saved regular checkpoint at epoch {epoch+1}")

    def save_best_model_by_loss(self, epoch, val_loss):
        """Saves the best model based on validation loss."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_loss_path = os.path.join(self.save_path,
                                          'best_loss_model.pth')
            self.save_checkpoint(
                best_loss_path,
                epoch=epoch + 1,
                val_loss=val_loss
            )
            self.log(f"Saved new best model by loss (epoch {epoch+1})")
            return True
        return False

    def save_best_model_by_iou(self, epoch, metrics, optimizer, train_loss,
                               val_loss):
        """Saves the best model based on IoU metric."""
        if metrics['iou'] > self.best_iou:
            self.best_iou = metrics['iou']
            best_iou_path = os.path.join(self.save_path, 'best_iou_model.pth')
            self.save_checkpoint(
                best_iou_path,
                epoch=epoch + 1,
                optimizer_state_dict=optimizer.state_dict(),
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics
            )
            self.log(f"Saved new best model by IoU (epoch {epoch+1}, IoU: \
{metrics['iou']:.4f})")
            return True
        return False

    def save_final_model(self, epoch):
        """Saves the final model at the end of training."""
        if epoch == self.config['training']['num_epochs'] - 1:
            final_path = os.path.join(self.save_path, 'last_model.pth')
            self.save_checkpoint(
                final_path,
                epoch=epoch + 1
            )
            self.log("Saved final model")

    def save_progress(self, epoch, train_loss, val_loss, metrics, optimizer,
                      scheduler=None):
        """
        Save training progress, including periodic checkpoints and best models.

        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            metrics (dict): Performance metrics
            optimizer (torch.optim.Optimizer): Optimizer state
            scheduler: Learning rate scheduler (optional)

        Returns:
            tuple: (early_stop_counter, improved) - Counter for early stopping
                and whether metrics improved
        """
        # Save regular checkpoint
        self.save_regular_checkpoint(epoch, optimizer, scheduler, train_loss,
                                     val_loss, metrics)

        # Track if we've improved (for early stopping)
        improved = False
        early_stop_counter = 0

        # Save best model by loss
        if not self.save_best_model_by_loss(epoch, val_loss):
            early_stop_counter = 1

        # Save best model by IoU
        improved = self.save_best_model_by_iou(epoch, metrics, optimizer,
                                               train_loss, val_loss) or\
            improved

        # Save final model
        self.save_final_model(epoch)

        return early_stop_counter, improved

    def unfreeze_next_stage_if_needed(self, current_metric, patience=3):
        """
        Unfreezes the next stage if model supports it and metrics haven't
        improved.

        Args:
            current_metric (float): Current metric value (e.g., IoU)
            patience (int): Patience for unfreezing

        Returns:
            bool: True if unfreezing occurred, False otherwise
        """
        # Check if model supports unfreezing
        supports_unfreezing = self.model_type in ["unet_resnet", "swin2_unet"]

        if supports_unfreezing and hasattr(self.model,
                                           'unfreeze_next_stage_if_needed'):
            # Get unfreezing patience from config or use default
            if self.model_type == "unet_resnet":
                unfreezing_patience = self.config['model'].get(
                    'unet_resnet', {}).\
                    get('unfreezing_patience', patience)
            elif self.model_type == "swin2_unet":
                unfreezing_patience = self.config['model'].get(
                    'swin2_unet', {}).\
                    get('unfreezing_patience', patience)
            else:
                unfreezing_patience = patience

            # Delegate to model's unfreezing logic
            did_unfreeze = self.model.unfreeze_next_stage_if_needed(
                current_metric=current_metric,
                patience=unfreezing_patience
            )

            if did_unfreeze:
                self.log(f"Unfroze next stage of encoder (stage \
{self.model.current_unfreeze_stage})")
                return True

        return False


# For backward compatibility
def create_model(config, device, logger=None):
    """Legacy function for creating models, delegates to ModelManager."""
    manager = ModelManager(config, device, logger)
    return manager.create_model()
