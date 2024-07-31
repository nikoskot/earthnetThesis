import torch
from torch import nn
import torch.nn.functional as F
from kornia.filters import spatial_gradient
from piqa import SSIM
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
    def __init__(self, lossType, lossFunction, maskFlag=True):
        super(MaskedLoss, self).__init__()

        self.lossFunction = lossFunction
        self.lossType = lossType
        self.maskFlag = maskFlag

    def forward(self, preds, targets, mask):

        assert(preds.shape == targets.shape)

        mask = torch.repeat_interleave(mask, repeats=4, dim=1)

        if not self.maskFlag:
            mask = torch.ones_like(mask)

        preds = preds * mask
        targets = targets * mask

        if self.lossType == 'ffl':
            l = 0
            for t in range(preds.shape[2]):
                l += self.lossFunction(preds[:, :, t, :, :], targets[:, :, t, :, :])
            return l / ((mask > 0).sum() + 1)

        if self.lossType == 'ssim':
            l = 0
            for t in range(preds.shape[2]):
                l += 1 - self.lossFunction(preds[:, :, t, :, :], targets[:, :, t, :, :])
            return l / t

        return self.lossFunction(preds, targets)/ ((mask > 0).sum() + 1)
        
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
    masks = torch.ones(8, 1, 20, 128, 128).to(torch.device('cuda'))
    ones = torch.ones(8, 4, 20, 128, 128).to(torch.device('cuda'))
    zeros = torch.zeros(8, 4, 20, 128, 128).to(torch.device('cuda'))

    config = {'trainLossFunctions': ["mse", "maskedL1", "l1", "maskedSSIM", "SSIM", "maskedFFL", "FFL", "maskedGradient", "gradient"],
              'modelOutputCh': 4
              }
    config = {'trainLossFunctions': ["SSIM"],
              'modelOutputCh': 4
              }
    device = torch.device('cuda')

    lossFunctions = setupLossFunctions(config, device)

    lossesSame = {l:lossFunctions[l](img1, img1, masks) for l in config['trainLossFunctions']}
    lossesSameZeroMask = {l:lossFunctions[l](img1, img1, 1-masks) for l in config['trainLossFunctions']}
    lossesDifferent = {l:lossFunctions[l](img1, img2, masks) for l in config['trainLossFunctions']}
    lossesOnesZeros = {l:lossFunctions[l](ones, zeros, masks) for l in config['trainLossFunctions']}

    for l in config['trainLossFunctions']:
        print("Mean loss {} between same images : {}".format(l, lossesSame[l]))
        print("Mean loss {} between same images with zero mask: {}".format(l, lossesSameZeroMask[l]))
        print("Mean loss {} between different images: {}".format(l, lossesDifferent[l]))
        print("Mean loss {} between ones and zeros: {}".format(l, lossesOnesZeros[l]))

