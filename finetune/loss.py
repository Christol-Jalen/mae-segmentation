import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def dice_score(self, ps, ts):
        """
        Compute the Dice score, a measure of overlap between two sets.
        """
        numerator = torch.sum(ts * ps, dim=(1, 2, 3)) * 2 + self.eps
        denominator = torch.sum(ts, dim=(1, 2, 3)) + torch.sum(ps, dim=(1, 2, 3)) + self.eps
        return numerator / denominator

    def dice_loss(self, ps, ts):
        """
        Compute the Dice loss, which is -1 times the Dice score.
        """
        return 1 - self.dice_score(ps, ts) 

    def dice_binary(self, ps, ts):
        """
        Threshold predictions and true values at 0.5, convert to float, and compute the Dice score.
        """
        ps = (ps >= 0.5).float()
        ts = (ts >= 0.5).float()
        return self.dice_score(ps, ts)