# src/models/loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Computes the Sørensen–Dice loss.
    Assumes model output is logits and applies sigmoid internally.
    Focuses on the positive class (label=1).
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits (N, 1, H, W)
            targets: Ground truth labels (N, 1, H, W)
        """
        probas = torch.sigmoid(logits)
        targets = targets.float() # Ensure targets are float

        probas = probas.view(probas.size(0), -1) # Shape: (N, H*W)
        targets = targets.view(targets.size(0), -1) # Shape: (N, H*W)

        intersection = (probas * targets).sum(dim=1)
        dice_coeff = (2. * intersection + self.smooth) / (probas.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        dice_loss = 1. - dice_coeff.mean()
        return dice_loss

class CombinedLoss(nn.Module):
    """
    Combines Weighted Binary Cross Entropy and Dice Loss.
    """
    def __init__(self, pos_weight, bce_weight=1.0, dice_weight=1.0, dice_smooth=1e-6):
        super(CombinedLoss, self).__init__()
        print(f"Initializing CombinedLoss with BCE Weight: {bce_weight}, Dice Weight: {dice_weight}")
        # Ensure pos_weight is prepared for BCEWithLogitsLoss constructor
        if isinstance(pos_weight, (int, float)):
             pos_weight = torch.tensor([pos_weight])
        elif isinstance(pos_weight, list):
             pos_weight = torch.tensor(pos_weight)
        # pos_weight should now be a tensor

        print(f"Using BCE pos_weight: {pos_weight.item() if pos_weight.numel() == 1 else pos_weight}")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Instantiate BCE loss; pos_weight will be moved to device in forward()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits: Model output logits (N, 1, H, W)
            targets: Ground truth labels (N, 1, H, W)
        """
        # Ensure pos_weight is on the correct device just before use
        self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logits.device)

        bce = self.bce_loss(logits, targets.float()) # BCE expects float targets
        dice = self.dice_loss(logits, targets)

        combined_loss = (self.bce_weight * bce) + (self.dice_weight * dice)
        return combined_loss

if __name__ == '__main__':
     # Example usage (keep unchanged)
     logits = torch.randn(4, 1, 256, 256, requires_grad=True) # Batch=4
     labels = (torch.rand(4, 1, 256, 256) > 0.8).long()    # Example binary labels
     print("Testing Dice Loss")
     dice = DiceLoss()
     dice_l = dice(logits, labels)
     print(f"Dice Loss: {dice_l.item()}")
     print("\nTesting Combined Loss")
     pos_w = torch.tensor([10.0]) # Example weight
     combined = CombinedLoss(pos_weight=pos_w, bce_weight=0.5, dice_weight=0.5)
     combined_l = combined(logits, labels)
     print(f"Combined Loss: {combined_l.item()}")