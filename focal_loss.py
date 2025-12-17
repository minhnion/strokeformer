import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for binary classification.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits from the model (before sigmoid)
        targets: binary ground truth labels (0 or 1)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # pt is the probability of the correct class
        p = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, p, 1 - p)
        
        # alpha factor
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # modulating factor
        modulating_factor = (1.0 - pt).pow(self.gamma)
        
        # final focal loss
        focal_loss = alpha_factor * modulating_factor * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
