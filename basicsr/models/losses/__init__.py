from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss, SCSCRLoss,PerceptualLoss)
from .gan_loss import GANLoss
__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss', 'SCSCRLoss','PerceptualLoss','GANLoss'
]
