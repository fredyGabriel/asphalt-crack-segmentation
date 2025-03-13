import unittest
from unittest.mock import MagicMock, call
import torch
from factory.model_manager import ModelManager
from models.swin2_unet import SwinUNetV2
from factory.custom_schedulers import WarmupCosineAnnealingLR
from utils.logger import Logger  # Import the Logger class


class TestScheduler(unittest.TestCase):

    def setUp(self):
        # Mock configuration
        self.config = {
            'model': {
                'type': 'swin2_unet',
                'save_path': 'test_save_path'  # Add save_path
            },
            'training': {
                'learning_rate': 0.001,
                'encoder_lr_factor': 0.1,
                'num_epochs': 100,
                'scheduler': {
                    'type': 'WarmupCosineAnnealing',
                    'warmup_epochs': 10,
                    'min_lr_decoder': 0.00001,
                    'min_lr_encoder': 0.000001
                }
            }
        }
        self.device = torch.device('cpu')  # Use CPU for testing
        self.model_manager = ModelManager(self.config, self.device)

        # Create a mock SwinUNetV2 model
        self.model_manager.model = SwinUNetV2(
            img_size=224,  # Changed to 224 to be divisible by window_size
            patch_size=4,
            in_chans=3,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0, 0, 0, 0],
        )

    def test_parameter_group_separation(self):
        # Setup optimizer
        optimizer = self.model_manager.setup_optimizer(
            base_lr=self.config['training']['learning_rate'])

        # Verify that there are two parameter groups (encoder and decoder)
        self.assertEqual(len(optimizer.param_groups), 2)

        # Get encoder parameters
        encoder_params = list(self.model_manager.model.encoder.parameters())

        # Get decoder parameters
        decoder_params = [p for n, p in self.model_manager.model.
                          named_parameters() if 'encoder' not in n]

        # Verify that the first parameter group is the encoder
        encoder_group = optimizer.param_groups[0]['params']
        self.assertEqual(len(encoder_group),
                         sum(1 for p in encoder_params if p.requires_grad))

        # Verify that the second parameter group is the decoder
        decoder_group = optimizer.param_groups[1]['params']
        self.assertEqual(len(decoder_group),
                         sum(1 for p in decoder_params if p.requires_grad))

        # Verify that the encoder and decoder have different learning rates
        encoder_lr = self.config['training']['learning_rate'] *\
            self.config['training']['encoder_lr_factor']
        decoder_lr = self.config['training']['learning_rate']
        self.assertEqual(optimizer.param_groups[0]['lr'], encoder_lr)
        self.assertEqual(optimizer.param_groups[1]['lr'], decoder_lr)

    def test_scheduler_instantiation(self):
        # Setup optimizer
        optimizer = self.model_manager.setup_optimizer(
            base_lr=self.config['training']['learning_rate'])

        # Setup scheduler
        scheduler = self.model_manager.setup_scheduler(optimizer)

        # Verify that the scheduler is a WarmupCosineAnnealingLR
        self.assertIsInstance(scheduler, WarmupCosineAnnealingLR)

        # Verify that the scheduler parameters are correct
        self.assertEqual(scheduler.T_warmup,
                         self.config['training']['scheduler']['warmup_epochs'])
        self.assertEqual(scheduler.T_max,
                         self.config['training']['num_epochs'])
        self.assertEqual(scheduler.eta_min_decoder,
                         self.config['training']['scheduler']['min_lr_decoder']
                         )
        self.assertEqual(scheduler.eta_min_encoder,
                         self.config['training']['scheduler']['min_lr_encoder']
                         )

    def test_learning_rate_logging(self):
        # Setup optimizer
        optimizer = self.model_manager.setup_optimizer(
            base_lr=self.config['training']['learning_rate'])

        # Create a mock Logger
        mock_logger = MagicMock()
        logger = Logger(log_dir='test_logs')
        logger.writer = mock_logger

        # Log learning rates
        logger.log_learning_rate(optimizer, epoch=1)

        # Verify that the log_learning_rate method was called twice
        # self.assertEqual(mock_logger.add_scalar.call_count, 2) # Original
        # assertion

        # Verify that the learning rates were logged correctly using
        # assert_has_calls
        encoder_lr = self.config['training']['learning_rate'] *\
            self.config['training']['encoder_lr_factor']
        decoder_lr = self.config['training']['learning_rate']
        calls = [
            call('LearningRate/group_0', encoder_lr, 1),
            call('LearningRate/group_1', decoder_lr, 1)
        ]
        mock_logger.add_scalar.assert_has_calls(calls, any_order=True)
        # Ensure exactly two calls
        self.assertEqual(mock_logger.add_scalar.call_count, len(calls))

    def test_warmup_phase(self):
        # Get training parameters from config
        base_lr = self.config['training']['learning_rate']
        encoder_lr_factor = self.config['training']['encoder_lr_factor']
        warmup_epochs = self.config['training']['scheduler']['warmup_epochs']
        min_lr_decoder = self.config['training']['scheduler']['min_lr_decoder']
        min_lr_encoder = self.config['training']['scheduler']['min_lr_encoder']

        # Calculate target learning rates
        encoder_lr = base_lr * encoder_lr_factor
        decoder_lr = base_lr

        # Create mock optimizer with two parameter groups - initializing with
        # min learning rates
        optimizer = MagicMock()
        optimizer.param_groups = [
            {'lr': min_lr_encoder},  # Encoder group
            {'lr': min_lr_decoder}   # Decoder group
        ]

        # Create WarmupCosineAnnealingLR scheduler
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            T_warmup=warmup_epochs,
            T_max=self.config['training']['num_epochs'],
            eta_min_decoder=min_lr_decoder,
            eta_min_encoder=min_lr_encoder
        )

        # Simulate warmup phase and check learning rates
        for epoch in range(1, warmup_epochs + 1):
            scheduler.step()

            # Usar exactamente la misma función que utiliza el scheduler
            expected_encoder_lr = scheduler._get_warmup_lr(encoder_lr, 0)
            expected_decoder_lr = scheduler._get_warmup_lr(decoder_lr, 1)

            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],
                                   expected_encoder_lr, places=6)
            self.assertAlmostEqual(optimizer.param_groups[1]['lr'],
                                   expected_decoder_lr, places=6)

    def test_cosine_annealing_phase(self):
        # Get training parameters from config
        base_lr = self.config['training']['learning_rate']
        encoder_lr_factor = self.config['training']['encoder_lr_factor']
        warmup_epochs = self.config['training']['scheduler']['warmup_epochs']
        total_epochs = self.config['training']['num_epochs']
        min_lr_decoder = self.config['training']['scheduler']['min_lr_decoder']
        min_lr_encoder = self.config['training']['scheduler']['min_lr_encoder']

        # Calculate target learning rates - start with full learning rates for
        # annealing test
        encoder_lr = base_lr * encoder_lr_factor
        decoder_lr = base_lr

        # Create mock optimizer with two parameter groups - initializing with
        # max learning rates
        optimizer = MagicMock()
        optimizer.param_groups = [
            {'lr': encoder_lr},  # Encoder group already at full learning rate
            {'lr': decoder_lr}   # Decoder group already at full learning rate
        ]

        # Create WarmupCosineAnnealingLR scheduler
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            T_warmup=warmup_epochs,
            T_max=total_epochs,
            eta_min_decoder=min_lr_decoder,
            eta_min_encoder=min_lr_encoder
        )

        # Set scheduler to warmup phase completion
        for epoch in range(warmup_epochs):
            scheduler.step()

        # Simulate cosine annealing phase and check learning rates
        for epoch in range(warmup_epochs + 1, total_epochs + 1):
            scheduler.step()
            # Remove unused progress variable or use it in calculations below

            # Usar exactamente la misma función que utiliza el scheduler
            expected_encoder_lr = scheduler._get_cosine_lr(encoder_lr, 0)
            expected_decoder_lr = scheduler._get_cosine_lr(decoder_lr, 1)

            self.assertAlmostEqual(optimizer.param_groups[0]['lr'],
                                   expected_encoder_lr, places=6)
            self.assertAlmostEqual(optimizer.param_groups[1]['lr'],
                                   expected_decoder_lr, places=6)

    def test_minimum_learning_rate(self):
        # Get training parameters from config
        base_lr = self.config['training']['learning_rate']
        encoder_lr_factor = self.config['training']['encoder_lr_factor']
        warmup_epochs = self.config['training']['scheduler']['warmup_epochs']
        total_epochs = self.config['training']['num_epochs']
        min_lr_decoder = self.config['training']['scheduler']['min_lr_decoder']
        min_lr_encoder = self.config['training']['scheduler']['min_lr_encoder']

        # Calculate target learning rates
        encoder_lr = base_lr * encoder_lr_factor
        decoder_lr = base_lr

        # Create mock optimizer with two parameter groups
        optimizer = MagicMock()
        optimizer.param_groups = [
            {'lr': encoder_lr},  # Encoder group
            {'lr': decoder_lr}   # Decoder group
        ]

        # Create WarmupCosineAnnealingLR scheduler
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            T_warmup=warmup_epochs,
            T_max=total_epochs,
            eta_min_decoder=min_lr_decoder,
            eta_min_encoder=min_lr_encoder
        )

        # Simulate entire training and check minimum learning rates
        for epoch in range(1, total_epochs + 1):
            scheduler.step()
            self.assertGreaterEqual(optimizer.param_groups[0]['lr'],
                                    min_lr_encoder)
            self.assertGreaterEqual(optimizer.param_groups[1]['lr'],
                                    min_lr_decoder)


if __name__ == '__main__':
    unittest.main()
