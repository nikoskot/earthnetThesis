import torch
from torch import nn
import torch.nn.functional as F
from kornia.filters import spatial_gradient
from piqa import SSIM, ssim
from focal_frequency_loss import FocalFrequencyLoss as FFL
import numpy as np
from skimage import metrics
from typing import Tuple
import torchvision

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
    
def maskedSSIMLoss(preds, targets, mask):
        """Structural similarity index score

        !!!!DESCRIPTION NEEDS CHANGE!!!!

        Structural similarity between predicted and target cube computed for all channels and frames individually if the given target is less than 30% masked. Scaled by a scaling factor such that a mean SSIM of 0.8 is scaled to a ssim-score of 0.1. The ssim-score is mean(ssim), it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): Predictions, shape h,w,c,t
            targs (np.ndarray): Targets, shape h,w,c,t
            masks (np.ndarray): Masks, shape h,w,c,t, 1 if non-masked, else 0

        Returns:
            Tuple[float, dict]: ssim-score, debugging information
        """        
        ssim = SSIM(window_size=7, n_channels=4, reduction='sum').to(torch.device('cuda'))

        mask = torch.repeat_interleave(mask, repeats=4, dim=1) # BCTHW
        targets = torch.where(mask==1, targets, preds)

        running_ssim = 0
        counts = 0

        for j in range(mask.shape[0]):
            tmpMask = mask[j, :, :, :, :]
            tmpMaskSumsPerFrame = torch.sum(tmpMask, dim=[0, 2, 3])
            tmpMaskFramesToKeep = torch.argwhere(tmpMaskSumsPerFrame > 0.7 * (tmpMask.shape[0] * tmpMask.shape[2] * tmpMask.shape[3]))
            tmpMaskFramesToKeep = tmpMaskFramesToKeep.squeeze(dim=1)
            if len(tmpMaskFramesToKeep) == 0:
                continue
            tmpPreds = preds[j, :, tmpMaskFramesToKeep, :, :].permute(1, 0, 2, 3)
            tmpTargs = targets[j, :, tmpMaskFramesToKeep, :, :].permute(1, 0, 2, 3)

            running_ssim += ssim(tmpTargs, tmpPreds)
            counts += len(tmpMaskFramesToKeep)
        
        if counts == 0:
            ssim = 0
        else:
            ssim = max(0,(running_ssim/max(counts,1)))
            scaling_factor = 10.31885115 # Scales SSIM=0.8 down to 0.1
            ssim = ssim ** scaling_factor

        return 1 - ssim

def SSIMEarthnetToolkit(preds: np.ndarray, targs: np.ndarray, masks: np.ndarray) -> Tuple[float, dict]:
        """Structural similarity index score

        Structural similarity between predicted and target cube computed for all channels and frames individually if the given target is less than 30% masked. Scaled by a scaling factor such that a mean SSIM of 0.8 is scaled to a ssim-score of 0.1. The ssim-score is mean(ssim), it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): Predictions, shape h,w,c,t
            targs (np.ndarray): Targets, shape h,w,c,t
            masks (np.ndarray): Masks, shape h,w,c,t, 1 if non-masked, else 0

        Returns:
            Tuple[float, dict]: ssim-score, debugging information
        """        

        ssim_targs = np.where(masks, targs, preds)
        new_shape = (-1, preds.shape[0], preds.shape[1])
        ssim_targs = np.transpose(np.reshape(np.transpose(ssim_targs, (3,2,0,1)), new_shape),(1,2,0))
        ssim_preds = np.transpose(np.reshape(np.transpose(preds, (3,2,0,1)), new_shape),(1,2,0))
        ssim_masks = np.transpose(np.reshape(np.transpose(masks, (3,2,0,1)), new_shape),(1,2,0))
        running_ssim = 0
        counts = 0
        ssim_frames = []
        for i in range(ssim_targs.shape[-1]):
            if ssim_masks[:,:,i].sum() > 0.7*ssim_masks[:,:,i].size:
                curr_ssim = metrics.structural_similarity(ssim_targs[:,:,i], ssim_preds[:,:,i], data_range=1.0)
                running_ssim += curr_ssim
                counts += 1
            else:
                curr_ssim = 1000
            ssim_frames.append(curr_ssim)
        
        if counts == 0:
            ssim = None
        else:
            ssim = max(0,(running_ssim/max(counts,1)))

            scaling_factor = 10.31885115 # Scales SSIM=0.8 down to 0.1

            ssim = float(ssim ** scaling_factor)
        
        debug_info = {
                        #"framewise SSIM, 1000 if frame was too much masked": ssim_frames, 
                        "Min SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).min(), np.nan)),
                        "Max SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).max(), np.nan)),
                        "Mean SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).mean(), np.nan)),
                        "Standard deviation SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).std(), np.nan)),
                        "Valid SSIM frames": counts,
                        "SSIM score": ssim,
                        "frames": ssim_frames
                    }

        return ssim, debug_info

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights='VGG19_Weights.DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class MaskedVGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(MaskedVGGLoss, self).__init__()

        # VGG architecter, used for the perceptual loss using a pretrained VGG network from spade paper
        # https://github.com/NVlabs/SPADE

        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [8., 1., 1., 1., 1.]

    def forward(self, x, y, mask):

        assert(x.shape == y.shape)

        B, C, T, H, W = x.shape

        mask = torch.repeat_interleave(mask, repeats=4, dim=1)

        x[:, 0, :, :, :] = (x[:, 0, :, :, :] - 0.18194012) / 0.27229501
        x[:, 1, :, :, :] = (x[:, 1, :, :, :] - 0.19730986) / 0.25477614
        x[:, 2, :, :, :] = (x[:, 2, :, :, :] - 0.19899204) / 0.25042932
        x[:, 3, :, :, :] = (x[:, 3, :, :, :] - 0.36337873) / 0.20255798
        x = x * mask
        y[:, 0, :, :, :] = (y[:, 0, :, :, :] - 0.18194012) / 0.27229501
        y[:, 1, :, :, :] = (y[:, 1, :, :, :] - 0.19730986) / 0.25477614
        y[:, 2, :, :, :] = (y[:, 2, :, :, :] - 0.19899204) / 0.25042932
        y[:, 3, :, :, :] = (y[:, 3, :, :, :] - 0.36337873) / 0.20255798
        y = y * mask

        xBGR = x[:, 0:3, :, :, :] # isolate bgr channels
        xBGR = xBGR.permute(0, 2, 1, 3, 4).reshape(B*T, C-1, H, W) # Change shape to be able to pass through VGG

        xNIF = x[:, 3, :, :, :] # isolate nif
        xNIF = xNIF.unsqueeze(1).repeat(1, 3, 1, 1, 1) # make it 3 channel
        xNIF = xNIF.permute(0, 2, 1, 3, 4).reshape(B*T, C-1, H, W) # Change shape to be able to pass through VGG

        yBGR = y[:, 0:3, :, :, :] # isolate bgr channels
        yBGR = yBGR.permute(0, 2, 1, 3, 4).reshape(B*T, C-1, H, W) # Change shape to be able to pass through VGG

        yNIF = y[:, 3, :, :, :] # isolate nif
        yNIF = yNIF.unsqueeze(1).repeat(1, 3, 1, 1, 1) # make it 3 channel
        yNIF = yNIF.permute(0, 2, 1, 3, 4).reshape(B*T, C-1, H, W) # Change shape to be able to pass through VGG

        xBGR_vgg, yBGR_vgg, xNIF_vgg, yNIF_vgg = self.vgg(xBGR), self.vgg(yBGR), self.vgg(xNIF), self.vgg(yNIF)
        lossBGR = 0
        lossNIF = 0
        for i in range(len(xBGR_vgg)):
            lossBGR += self.weights[i] * self.criterion(xBGR_vgg[i], yBGR_vgg[i].detach())
            lossNIF += self.weights[i] * self.criterion(xNIF_vgg[i], yNIF_vgg[i].detach())

        return (lossBGR + lossNIF) #/ ((mask > 0).sum() + 1e-7)
    
if __name__ == "__main__":
    
    img1 = torch.rand(8, 4, 20, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    img2 = torch.rand(8, 4, 20, 128, 128).to(torch.device('cuda')) # B, C, T, H, W
    masksOnes = torch.ones(8, 1, 20, 128, 128).to(torch.device('cuda'))
    masksZeros = torch.zeros(8, 1, 20, 128, 128).to(torch.device('cuda'))
    ones = torch.ones(8, 4, 20, 128, 128).to(torch.device('cuda'))
    zeros = torch.zeros(8, 4, 20, 128, 128).to(torch.device('cuda'))

    realTargetImage = np.load('/home/nikoskot/earthnetThesis/EarthnetDataset/iid_test_split/target/29SND/target_29SND_2017-06-20_2017-11-16_953_1081_3641_3769_14_94_56_136.npz')
    targetHighresdynamic = np.nan_to_num(realTargetImage['highresdynamic'], copy=False, nan=0.0, posinf=1.0, neginf=0.0) 
    targetHighresdynamic = np.clip(targetHighresdynamic, a_min=0.0, a_max=1.0)
    realMask = torch.from_numpy(targetHighresdynamic.astype('float32')).permute(2, 3, 0, 1).unsqueeze(0)[:, 4, :, :, :].unsqueeze(1).to(torch.device('cuda'))
    realTargetImage = torch.from_numpy(targetHighresdynamic.astype('float32')).permute(2, 3, 0, 1).unsqueeze(0)[:, 0:4, :, :, :].to(torch.device('cuda'))
    realPredictedImage = np.load('/home/nikoskot/earthnetThesis/experiments/videoSwinUnetV1_03-09-2024_07-46-11/predictions/iid_test_split/29SND/target_29SND_2017-06-20_2017-11-16_953_1081_3641_3769_14_94_56_136.npz')
    predHighresdynamic = np.nan_to_num(realPredictedImage['highresdynamic'], copy=False, nan=0.0, posinf=1.0, neginf=0.0) 
    predHighresdynamic = np.clip(predHighresdynamic, a_min=0.0, a_max=1.0)
    realPredictedImage = torch.from_numpy(predHighresdynamic.astype('float32')).permute(2, 3, 0, 1).unsqueeze(0).to(torch.device('cuda'))
    
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
    print('\n')

    ssimLossOriginal = SSIM(n_channels=4, reduction='mean').to(torch.device('cuda'))
    print("Mean original SSIM loss between different images: {}".format(1-ssimLossOriginal(img1, img2)))
    print("Mean original SSIM loss between same images: {}".format(1-ssimLossOriginal(img1, img1)))
    print("Mean original SSIM loss between ones and zeros: {}".format(1-ssimLossOriginal(ones, zeros)))
    print('\n')

    print("Mean new SSIM loss between different images with mask equal to one: {}".format(maskedSSIMLoss(img1, img2, masksOnes)))
    print("Mean new SSIM loss between same images with mask equal to one: {}".format(maskedSSIMLoss(img1, img1, masksOnes)))
    print("Mean new SSIM loss between ones and zeros with mask equal to one: {}".format(maskedSSIMLoss(ones, zeros, masksOnes)))
    print("Mean new SSIM loss between different images with mask equal to zeros: {}".format(maskedSSIMLoss(img1, img2, masksZeros)))
    print('\n')

    vggLoss = MaskedVGGLoss()
    print("Mean masked VGG loss between different images with mask equal to one: {}".format(vggLoss(img1, img2, masksOnes)))
    print("Mean masked VGG loss between same images with mask equal to one: {}".format(vggLoss(img1, img1, masksOnes)))
    print("Mean masked VGG loss between ones and zeros with mask equal to one: {}".format(vggLoss(ones, zeros, masksOnes)))
    print("Mean masked VGG loss between different images with mask equal to zeros: {}".format(vggLoss(img1, img2, masksZeros)))
    print("Mean masked VGG loss between target and predicted image with real mask: {}".format(vggLoss(realPredictedImage.clone(), realTargetImage.clone(), realMask)))
    print("Mean masked VGG loss between target and predicted image with mask one: {}".format(vggLoss(realPredictedImage.clone(), realTargetImage.clone(), masksOnes[0].unsqueeze(0))))
    print("Mean masked VGG loss between target and predicted image with mask zero: {}".format(vggLoss(realPredictedImage.clone(), realTargetImage.clone(), masksZeros[0].unsqueeze(0))))
    print("Mean masked VGG loss between target and target image with real mask: {}".format(vggLoss(realTargetImage.clone(), realTargetImage.clone(), realMask)))
    print('\n')

    # tmp = SSIMEarthnetToolkit(preds=img1[0].permute(2, 3, 0, 1).cpu().numpy(), targs=img2[0].permute(2, 3, 0, 1).cpu().numpy(), masks=masksOnes[0].repeat(4, 1, 1, 1).permute(2, 3, 0, 1).cpu().numpy())
    # print("SSIMEarthnetToolkit: {}".format(1- tmp[0]))

    # tmp = ssimLoss(img1[0].unsqueeze(0), img2[0].unsqueeze(0), masksOnes[0].unsqueeze(0))
    # print("SSIMMaskedLoss: {}".format(tmp))

    # tmp = 1-ssimLossOriginal(img1[0].unsqueeze(0), img2[0].unsqueeze(0))
    # print("SSIM Original: {}".format(tmp))

    # tmp = maskedSSIMLoss(img1[0].unsqueeze(0), img2[0].unsqueeze(0), masksOnes[0].unsqueeze(0))
    # print("SSIM New: {}".format(tmp))