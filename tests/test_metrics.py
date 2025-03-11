import unittest
import torch
from factory.metrics import (
    confusion_matrix, iou, pixel_accuracy,
    dice_coefficient, precision, recall, f1_score
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        # Case 1: Perfect prediction
        self.perfect_pred = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        self.perfect_target = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Case 2: Completely wrong prediction
        self.wrong_pred = torch.tensor([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=torch.float32)
        self.wrong_target = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Case 3: Partial overlap
        self.partial_pred = torch.tensor([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1]
        ], dtype=torch.float32)
        self.partial_target = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # Edge cases
        self.all_zeros_pred = torch.zeros((3, 3), dtype=torch.float32)
        self.all_ones_pred = torch.ones((3, 3), dtype=torch.float32)
        self.all_ones_target = torch.ones((3, 3), dtype=torch.float32)
        self.all_zeros_target = torch.zeros((3, 3), dtype=torch.float32)

    def test_confusion_matrix(self):
        # Test confusion matrix with perfect prediction
        tp, fp, tn, fn = confusion_matrix(self.perfect_pred,
                                          self.perfect_target)
        self.assertEqual(tp, 3)  # 3 true positives
        self.assertEqual(fp, 0)  # 0 false positives
        self.assertEqual(tn, 6)  # 6 true negatives
        self.assertEqual(fn, 0)  # 0 false negatives

        # Test confusion matrix with completely wrong prediction
        tp, fp, tn, fn = confusion_matrix(self.wrong_pred, self.wrong_target)
        self.assertEqual(tp, 0)  # 0 true positives
        self.assertEqual(fp, 6)  # 6 false positives
        self.assertEqual(tn, 0)  # 0 true negatives
        self.assertEqual(fn, 3)  # 3 false negatives

        # Test with all zeros and ones
        tp, fp, tn, fn = confusion_matrix(self.all_zeros_pred,
                                          self.all_zeros_target)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(tn, 9)
        self.assertEqual(fn, 0)

        tp, fp, tn, fn = confusion_matrix(self.all_ones_pred,
                                          self.all_ones_target)
        self.assertEqual(tp, 9)
        self.assertEqual(fp, 0)
        self.assertEqual(tn, 0)
        self.assertEqual(fn, 0)

    def test_iou(self):
        # Test IoU with perfect prediction (should be 1.0)
        perfect_iou = iou(self.perfect_pred, self.perfect_target)
        self.assertAlmostEqual(perfect_iou.item(), 1.0, places=5)

        # Test IoU with completely wrong prediction (should be close to 0)
        wrong_iou = iou(self.wrong_pred, self.wrong_target)
        self.assertAlmostEqual(wrong_iou.item(), 0.0, places=5)

        # Test IoU with partial overlap
        partial_iou = iou(self.partial_pred, self.partial_target)
        self.assertGreater(partial_iou.item(), 0.0)
        self.assertLess(partial_iou.item(), 1.0)

        # Test with edge cases
        zero_iou = iou(self.all_zeros_pred, self.all_ones_target)
        self.assertAlmostEqual(zero_iou.item(), 0.0, places=5)

    def test_pixel_accuracy(self):
        # Test accuracy with perfect prediction (should be 1.0)
        perfect_acc = pixel_accuracy(self.perfect_pred, self.perfect_target)
        self.assertAlmostEqual(perfect_acc.item(), 1.0, places=5)

        # Test accuracy with completely wrong prediction
        wrong_acc = pixel_accuracy(self.wrong_pred, self.wrong_target)
        self.assertAlmostEqual(wrong_acc.item(), 0.0, places=5)

        # Edge case: all zeros prediction and target
        all_zeros_acc = pixel_accuracy(self.all_zeros_pred,
                                       self.all_zeros_target)
        self.assertAlmostEqual(all_zeros_acc.item(), 1.0, places=5)

    def test_dice_coefficient(self):
        # Test Dice with perfect prediction (should be 1.0)
        perfect_dice = dice_coefficient(self.perfect_pred, self.perfect_target)
        self.assertAlmostEqual(perfect_dice.item(), 1.0, places=5)

        # Test Dice with completely wrong prediction (should be 0.0)
        wrong_dice = dice_coefficient(self.wrong_pred, self.wrong_target)
        self.assertAlmostEqual(wrong_dice.item(), 0.0, places=5)

        # Test Dice with partial overlap
        partial_dice = dice_coefficient(self.partial_pred, self.partial_target)
        expected_dice = 2 * 3 / (5 + 3)  # 2 * TP / (total_pred + total_target)
        self.assertAlmostEqual(partial_dice.item(), expected_dice, places=5)

    def test_precision(self):
        # Test precision with perfect prediction (should be 1.0)
        perfect_prec = precision(self.perfect_pred, self.perfect_target)
        self.assertAlmostEqual(perfect_prec.item(), 1.0, places=5)

        # Test precision with completely wrong prediction (should be 0.0)
        wrong_prec = precision(self.wrong_pred, self.wrong_target)
        self.assertAlmostEqual(wrong_prec.item(), 0.0, places=5)

        # Edge case: zeros prediction, any target
        # With smoothing factor, precision = (0+smooth)/(0+smooth) = 1.0
        zero_pred_prec = precision(self.all_zeros_pred, self.all_ones_target)
        self.assertAlmostEqual(zero_pred_prec.item(), 1.0, places=5)

    def test_recall(self):
        # Test recall with perfect prediction (should be 1.0)
        perfect_rec = recall(self.perfect_pred, self.perfect_target)
        self.assertAlmostEqual(perfect_rec.item(), 1.0, places=5)

        # Test recall with completely wrong prediction (should be 0.0)
        wrong_rec = recall(self.wrong_pred, self.wrong_target)
        self.assertAlmostEqual(wrong_rec.item(), 0.0, places=5)

        # Edge case: any prediction, zeros target
        # With smoothing factor, recall = (0+smooth)/(0+smooth) = 1.0
        zero_target_rec = recall(self.all_ones_pred, self.all_zeros_target)
        self.assertAlmostEqual(zero_target_rec.item(), 1.0, places=5)

    def test_f1_score(self):
        # Test F1 with perfect prediction (should be 1.0)
        perfect_f1 = f1_score(self.perfect_pred, self.perfect_target)
        self.assertAlmostEqual(perfect_f1.item(), 1.0, places=5)

        # Test F1 with completely wrong prediction (should be 0.0)
        wrong_f1 = f1_score(self.wrong_pred, self.wrong_target)
        self.assertAlmostEqual(wrong_f1.item(), 0.0, places=5)

        # Check if F1 equals Dice coefficient
        partial_f1 = f1_score(self.partial_pred, self.partial_target)
        partial_dice = dice_coefficient(self.partial_pred,
                                        self.partial_target)
        self.assertAlmostEqual(partial_f1.item(), partial_dice.item(),
                               places=5)


if __name__ == '__main__':
    unittest.main()
