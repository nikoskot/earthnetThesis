import torch
from torch import nn
import torch.nn.functional as F
from kornia.filters import spatial_gradient
from piqa import SSIM, ssim
from focal_frequency_loss import FocalFrequencyLoss as FFL

def setupLossFunctions(config, device):

    lossFunctions = {}

    if 'mse' in config['trainLossFunctions']:
        lossFunctions['mse'] = MaskedLoss(lossType='mse', lossFunction=nn.MSELoss(reduction='sum'), maskFlag=False) # Use reduction 'sum' because it is divided by the number of valid pixels based on the mask
    if 'maskedL1' in config['trainLossFunctions']:
        lossFunctions['maskedL1'] = MaskedLoss(lossType='l1', lossFunction=nn.L1Loss(reduction='sum'), maskFlag=True)
    if 'l1' in config['trainLossFunctions']:
        lossFunctions['l1'] = MaskedLoss(lossType='l1', lossFunction=nn.L1Loss(reduction='sum'), maskFlag=False)
    if 'maskedSSIM' in config['trainLossFunctions']:
        lossFunctions['maskedSSIM'] = MaskedLoss(lossType='ssim', lossFunction=SSIM(n_channels=config['modelOutputCh'], reduction='sum').to(device), maskFlag=True)
    if 'SSIM' in config['trainLossFunctions']:
        lossFunctions['SSIM'] = MaskedLoss(lossType='ssim', lossFunction=SSIM(n_channels=config['modelOutputCh'], reduction='mean').to(device), maskFlag=False)
    if 'maskedFFL' in config['trainLossFunctions']:
        lossFunctions['maskedFFL'] = MaskedLoss(lossType='ffl', lossFunction=FFL(loss_weight=1.0, alpha=1.0).to(device), maskFlag=True)
    if 'FFL' in config['trainLossFunctions']:
        lossFunctions['FFL'] = MaskedLoss(lossType='ffl', lossFunction=FFL(loss_weight=1.0, alpha=1.0).to(device), maskFlag=False)
    if 'maskedGradient' in config['trainLossFunctions']:
        lossFunctions['maskedGradient'] = MaskedLoss(lossType='gradient', lossFunction=GradientLoss(), maskFlag=True)
    if 'gradient' in config['trainLossFunctions']:
        lossFunctions['gradient'] = MaskedLoss(lossType='gradient', lossFunction=GradientLoss(), maskFlag=False)

    return lossFunctions

class MaskedLoss(nn.Module):
    def __init__(self, lossType, lossFunction):
        super(MaskedLoss, self).__init__()

        self.lossFunction = lossFunction
        self.lossType = lossType

    def forward(self, preds, targets, mask):

        assert(preds.shape == targets.shape)

        mask = torch.repeat_interleave(mask, repeats=4, dim=1)

        preds = preds * mask
        targets = targets * mask

        if self.lossType == 'ffl':
            l = 0
            for t in range(preds.shape[2]):
                l += self.lossFunction(preds[:, :, t, :, :], targets[:, :, t, :, :])
            return l / ((mask > 0).sum() + 1e-7)

        if self.lossType == 'ssim':
            B, _, _, _, _ = preds.shape
            return ((B - self.lossFunction(preds, targets)) / B) ** 10.31885115

        return self.lossFunction(preds, targets) / ((mask > 0).sum() + 1e-7)
        
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targs):
        loss = 0
        for t in range(preds.shape[2]):
            predsGrads = spatial_gradient(preds[:, :, t, :, :]) # B, C, 2, H, W
            targsGrads = spatial_gradient(targs[:, :, t, :, :]) # B, C, 2, H, W

            loss += F.l1_loss(predsGrads[:, :, 0, :, :], targsGrads[:, :, 0, :, :]) # x direction
            loss += F.l1_loss(predsGrads[:, :, 1, :, :], targsGrads[:, :, 1, :, :]) # y direction
        
        return loss
    
if __name__ == "__main__":
    
    img1 = torch.rand(8, 4, 20, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    img2 = torch.rand(8, 4, 20, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    masksOnes = torch.ones(8, 1, 20, 128, 128).to(torch.device('cuda'))
    masksZeros = torch.zeros(8, 1, 20, 128, 128).to(torch.device('cuda'))
    ones = torch.ones(8, 4, 20, 128, 128).to(torch.device('cuda'))
    zeros = torch.zeros(8, 4, 20, 128, 128).to(torch.device('cuda'))

    device = torch.device('cuda')

    mseLoss = MaskedLoss(lossType='mse', lossFunction=nn.MSELoss(reduction='sum'))
    print("Mean masked MSE loss between different images with mask equal to one: {}".format(mseLoss(img1, img2, masksOnes)))
    print("Mean masked MSE loss between same images with mask equal to one: {}".format(mseLoss(img1, img1, masksOnes)))
    print("Mean masked MSE loss between ones and zeros with mask equal to one: {}".format(mseLoss(ones, zeros, masksOnes)))
    print("Mean masked MSE loss between different images with mask equal to zeros: {}".format(mseLoss(img1, img2, masksZeros)))

    mseLossOriginal = nn.MSELoss(reduction='mean')
    print("Mean original MSE loss between different images: {}".format(mseLossOriginal(img1, img2)))
    print("Mean original MSE loss between same images: {}".format(mseLossOriginal(img1, img1)))
    print("Mean original MSE loss between ones and zeros: {}".format(mseLossOriginal(ones, zeros)))
    print('\n')

    l1Loss = MaskedLoss(lossType='l1', lossFunction=nn.L1Loss(reduction='sum'))
    print("Mean masked L1 loss between different images with mask equal to one: {}".format(l1Loss(img1, img2, masksOnes)))
    print("Mean masked L1 loss between same images with mask equal to one: {}".format(l1Loss(img1, img1, masksOnes)))
    print("Mean masked L1 loss between ones and zeros with mask equal to one: {}".format(l1Loss(ones, zeros, masksOnes)))
    print("Mean masked L1 loss between different images with mask equal to zeros: {}".format(l1Loss(img1, img2, masksZeros)))

    l1LossOriginal = nn.L1Loss(reduction='mean')
    print("Mean original L1 loss between different images: {}".format(l1LossOriginal(img1, img2)))
    print("Mean original L1 loss between same images: {}".format(l1LossOriginal(img1, img1)))
    print("Mean original L1 loss between ones and zeros: {}".format(l1LossOriginal(ones, zeros)))
    print('\n')

    ssimLoss = MaskedLoss(lossType='ssim', lossFunction=SSIM(n_channels=4, reduction='sum').to(torch.device('cuda')))
    print("Mean masked SSIM loss between different images with mask equal to one: {}".format(ssimLoss(img1, img2, masksOnes)))
    print("Mean masked SSIM loss between same images with mask equal to one: {}".format(ssimLoss(img1, img1, masksOnes)))
    print("Mean masked SSIM loss between ones and zeros with mask equal to one: {}".format(ssimLoss(ones, zeros, masksOnes)))
    print("Mean masked SSIM loss between different images with mask equal to zeros: {}".format(ssimLoss(img1, img2, masksZeros)))

    ssimLossOriginal = SSIM(n_channels=4, reduction='mean').to(torch.device('cuda'))
    print("Mean original SSIM loss between different images: {}".format(1-ssimLossOriginal(img1, img2)))
    print("Mean original SSIM loss between same images: {}".format(1-ssimLossOriginal(img1, img1)))
    print("Mean original SSIM loss between ones and zeros: {}".format(1-ssimLossOriginal(ones, zeros)))
    print('\n')