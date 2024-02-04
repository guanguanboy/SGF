import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from basicsr.models.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vgg16
import kornia as K
import numpy as np

from basicsr.models.losses.color_transfer import pre_process, transfer_chrom, transfer_lum

def inv_norm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    if tensor.get_device() > -1:
        mean = mean.to(tensor.get_device())
        std = std.to(tensor.get_device())
    if len(tensor.size()) == 3 and tensor.size(0) == 3:
        out = tensor * std[:,None,None] + mean[:,None,None]
    elif len(tensor.size()) == 4 and tensor.size(1) == 3:
        out = tensor * std[None,:,None,None] + mean[None,:,None,None]
    else:
        out = None
        print(f"Error! Can't normalize shape {tensor.size()}")

    return out

class ParamSmoothness(nn.Module):
    def __init__(self, pred_name='params'):
        super(self).__init__()
        self.pred_outputs = (pred_name, )
        self.gt_outputs = ()

    def forward(self, params):
        loss = torch.mean(K.filters.sobel(params))
        return loss
    
class SCSCRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', pred_name='images', gt_image_name='target_images', gt_comp_name='images', k=5, writer=None):
        super(SCSCRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.pred_outputs = (pred_name, )
        self.gt_outputs = (gt_image_name, gt_comp_name)

        self.D = nn.L1Loss()
        self.k = k
        self.writer = writer
        VGG16_model = vgg16(pretrained=True)
        self.SR_extractor = VGG16_model.features[:23]

    def forward(self, pred, gt_image, input_lowlight, return_neg_samp=False):
        '''
        composition (N, 3, H, W):           composition image (input image to the network)
        pred        (N, 3, H, W):           predicted image (output of the network)
        gt_image    (N, 3, H, W):           ground truth image
        mask        (N, 1, H, W):           image mask         
        '''
        composition = input_lowlight

        if self.k > pred.size(0)-1:
            self.k = pred.size(0)-1

        # create negative samples
        with torch.no_grad():
            neg_samp = self.create_negative_samples(gt_image.detach(), composition.detach())

        if return_neg_samp:
            return neg_samp

        # calculate foreground and background style representations    
        self.SR_extractor = self.SR_extractor.to(pred.get_device())
        self.SR_extractor.eval()

        f = self.SR_extractor(pred)                                            # (N, 512, 32, 32)
        f_plus = self.SR_extractor(gt_image)                                   # (N, 512, 32, 32)
        f_minus = [self.SR_extractor(neg_samp[:,k]) for k in range(self.k)]    # (K, N, 512, 32, 32)
        #b_plus = self.SR_extractor(gt_image * (1-mask))                               # (N, 512, 32, 32)
        
        # self-style contrastistive regularization (SS-CR)
        l_ss_cr = self.D(f, f_plus) / ( self.D(f, f_plus) + torch.sum(torch.tensor([self.D(f, f_minus_k) for f_minus_k in f_minus])) + 1e-8 )

        return self.loss_weight * l_ss_cr

    def create_negative_samples(self, gt_images, composition, gamma=2.2):
        '''
        samples K-1 negative samples for each composited image based on the other images in the batch

        gt_images   (N, 3, H, W):           ground truth images
        composition (N, 3, H, W):           composition image (input image to the network)
        mask        (N, 1, H, W):           image mask   

        neg_samp    (N, K, 3, H, W):        K negative samples 
        '''

        # Pre-processing
        N, C, H, W, = gt_images.size()        
        img = inv_norm(gt_images)
        img = torch.pow(img, gamma)
        img = K.color.rgb_to_lab(img)

        for n in range(N):
            img[n, 0] = pre_process(img[n, 0].clone())

        # Change color statistics for each composition
        neg_samp = torch.ones((N, self.k-1, C, H, W)).to(gt_images.get_device())
        ref_img = np.zeros((N, self.k-1), np.int16)
        for n in range(N):
            for j, n2 in enumerate(np.random.choice([i for i in range(N) if i!=n], self.k-1, replace=False)):

                ## No mask consideration for color transformation                
                c_img = transfer_chrom(img[n], img[n2])                                                 # (3, H, W)                           
                t_img = transfer_lum(img[n, 0], img[n2, 0])
                
                neg_img = torch.stack([t_img, c_img[1], c_img[2]])                # (3, H, W)
      
                neg_samp[n, j] = neg_img
                ref_img[n, j] = n2

        # Post-processing
        neg_samp = K.color.lab_to_rgb(neg_samp)
        neg_samp = torch.pow(neg_samp, 1/gamma)
        normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        neg_samp = normalize(neg_samp)
        neg_samp = torch.cat([composition.unsqueeze(1), neg_samp], dim=1)

        return neg_samp

    def Gram(self, mat1, mat2):
        '''
        caculates the Gram matrix

        mat1 (N, 512, 32, 32):              feature map
        mat2 (N, 512, 32, 32):              feature map

        out (N, 512, 512):                  Gram matrix of both feature maps
        '''

        out = []
        for f1, f2 in zip(mat1, mat2):
            out.append( torch.matmul(f1.view(512, -1).T, f2.view(512, -1)) )

        return torch.stack(out)
    

# def gradient(input_tensor, direction):
#     smooth_kernel_x = torch.reshape(torch.tensor([[0, 0], [-1, 1]], dtype=torch.float32), [2, 2, 1, 1])
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, 0, 1)
#     if direction == "x":
#         kernel = smooth_kernel_x
#     elif direction == "y":
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(torch.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
#     return grad_norm

# class SmoothLoss(nn.Moudle):
#     """ illumination smoothness"""

#     def __init__(self, loss_weight=0.15, reduction='mean', eps=1e-2):
#         super(SmoothLoss,self).__init__()
#         self.loss_weight = loss_weight
#         self.eps = eps
#         self.reduction = reduction
    
#     def forward(self, illu, img):
#         # illu: b×c×h×w   illumination map
#         # img:  b×c×h×w   input image
#         illu_gradient_x = gradient(illu, "x")
#         img_gradient_x  = gradient(img, "x")
#         x_loss = torch.abs(torch.div(illu_gradient_x, torch.maximum(img_gradient_x, 0.01)))

#         illu_gradient_y = gradient(illu, "y")
#         img_gradient_y  = gradient(img, "y")
#         y_loss = torch.abs(torch.div(illu_gradient_y, torch.maximum(img_gradient_y, 0.01)))

#         loss = torch.mean(x_loss + y_loss) * self.loss_weight

#         return loss

# class MultualLoss(nn.Moudle):
#     """ Multual Consistency"""

#     def __init__(self, loss_weight=0.20, reduction='mean'):
#         super(MultualLoss,self).__init__()

#         self.loss_weight = loss_weight
#         self.reduction = reduction
    

#     def forward(self, illu):
#         # illu: b x c x h x w
#         gradient_x = gradient(illu,"x")
#         gradient_y = gradient(illu,"y")

#         x_loss = gradient_x * torch.exp(-10*gradient_x)
#         y_loss = gradient_y * torch.exp(-10*gradient_y)

#         loss = torch.mean(x_loss+y_loss) * self.loss_weight
#         return loss

@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
