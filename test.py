# Test if ENS calculation of earthnet toolkit produces the same results with ENS from earthformer.
import earthnet as en
from swinUnet.earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
from swinUnet.earthnetDataloader import EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
from swinUnet.videoSwinUnetMMActionV1 import VideoSwinUNet
import os
import logging
import yaml
import matplotlib.pyplot as plt
from swinUnet.losses import MaskedLoss, SSIMEarthnetToolkit, maskedSSIMLoss
from piqa import SSIM
import time

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def newSSIM(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray):
        """Structural similarity index score

        Structural similarity between predicted and target cube computed for all channels and frames individually if the given target is less than 30% masked. Scaled by a scaling factor such that a mean SSIM of 0.8 is scaled to a ssim-score of 0.1. The ssim-score is mean(ssim), it is scaled from 0 (worst) to 1 (best).

        Args:
            preds (np.ndarray): Predictions, shape h,w,c,t
            targs (np.ndarray): Targets, shape h,w,c,t
            masks (np.ndarray): Masks, shape h,w,c,t, 1 if non-masked, else 0

        Returns:
            Tuple[float, dict]: ssim-score, debugging information
        """        
        ssim = SSIM(window_size=7, n_channels=4, reduction='mean').to(torch.device('cuda'))
        # ssim_targs = np.where(masks, targs, preds)
        # new_shape = (-1, preds.shape[0], preds.shape[1])
        # ssim_targs = np.transpose(np.reshape(np.transpose(ssim_targs, (3,2,0,1)), new_shape),(1,2,0))
        # ssim_preds = np.transpose(np.reshape(np.transpose(preds, (3,2,0,1)), new_shape),(1,2,0))
        # ssim_masks = np.transpose(np.reshape(np.transpose(masks, (3,2,0,1)), new_shape),(1,2,0))
        mask = torch.repeat_interleave(mask, repeats=4, dim=1) # BCTHW
        # print(mask.shape)
        preds = preds * mask
        targets = targets * mask

        running_ssim = 0
        counts = 0
        # ssim_frames = []
        for j in range(mask.shape[0]):
            tmpMask = mask[j, :, :, :, :]
            tmpMaskSumsPerFrame = torch.sum(tmpMask, dim=[0, 2, 3])
            tmpMaskFramesToKeep = torch.argwhere(tmpMaskSumsPerFrame > 0.7 * (tmpMask.shape[0] * tmpMask.shape[2] * tmpMask.shape[3]))
            # print(tmpMaskFramesToKeep)
            # print(targets[j, :, tmpMaskFramesToKeep.squeeze(), :, :].shape)
            # print(len(tmpMaskFramesToKeep))
            running_ssim += ssim(targets[j, :, tmpMaskFramesToKeep.squeeze(), :, :].unsqueeze(0), preds[j, :, tmpMaskFramesToKeep.squeeze(), :, :].unsqueeze(0))
            counts += len(tmpMaskFramesToKeep)
            # for i in range(mask.shape[2]):
            #     if mask[j, :, i, :, :].sum() > 0.7*torch.numel(mask[j, :, i, :, :]):
            #         curr_ssim = ssim(targets[j, :, i, :, :].unsqueeze(0), preds[j, :, i, :, :].unsqueeze(0))
            #         running_ssim += curr_ssim
            #         counts += 1
            #     else:
            #         curr_ssim = 1000
                    # ssim_frames.append(curr_ssim)
        # breakpoint()
        # Find frames per batch to use for the calculation
        # for j in range(mask.shape[0]):
        #     tmpMask = mask[j, :, :, :, :]
        #     print(tmpMask.shape)
        #     tmpMaskSumsPerFrame = torch.sum(tmpMask, dim=[0, 2, 3])
        #     print(tmpMaskSumsPerFrame)
        #     tmpMaskNumElemPerFrame = tmpMask.shape[0] * tmpMask.shape[2] * tmpMask.shape[3]
        #     print(tmpMaskNumElemPerFrame)
        #     tmpMaskFramesToKeep = torch.argwhere(tmpMaskSumsPerFrame > 0.7 * tmpMaskNumElemPerFrame)
        #     print(tmpMaskFramesToKeep)
        
        if counts == 0:
            ssim = None
        else:
            ssim = max(0,(running_ssim/max(counts,1)))

            scaling_factor = 10.31885115 # Scales SSIM=0.8 down to 0.1

            ssim = float(ssim ** scaling_factor)
        
        # debug_info = {
        #                 #"framewise SSIM, 1000 if frame was too much masked": ssim_frames, 
        #                 "Min SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).min(), np.nan)),
        #                 "Max SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).max(), np.nan)),
        #                 "Mean SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).mean(), np.nan)),
        #                 "Standard deviation SSIM": str(np.ma.filled(np.ma.masked_equal(np.array(ssim_frames), 1000.0).std(), np.nan)),
        #                 "Valid SSIM frames": counts,
        #                 "SSIM score": ssim,
        #                 "frames": ssim_frames
        #             }

        return ssim

# dataset = EarthnetTrainDataset(dataDir='/home/nikoskot/earthnetThesis/EarthnetDataset/train', dtype='float32', transform=Preprocessing(), cropMesodynamic=True)
# dataset = Subset(dataset, range(10))

# dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# # earthformerENS = EarthNet2021ScoreUpdateWithoutCompute(layout='NTHWC', eps=1E-4)

trainingFolder = '/home/nikoskot/earthnetThesis/experiments/videoSwinUnetV1_30-08-2024_06-09-13'

config = load_config(os.path.join(trainingFolder, 'savedConfig.yml'))

logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(trainingFolder, 'testing_{}.log'.format('train_part')), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

# Load training checkpoint
checkpoint = torch.load(os.path.join(trainingFolder, 'checkpoint.pth'))

# Intiialize Video Swin Unet model and move to GPU
model = VideoSwinUNet(config, logger).to(torch.device('cuda'))
model.load_state_dict(checkpoint['state_dict'])

# predsFolder = os.path.join(trainingFolder, 'predictions', 'train_part') # .../experiments/modelType_trainDatetime/predictions/split/
# os.makedirs(predsFolder, exist_ok=True)

model.eval()

# ssimLoss = MaskedLoss(lossType='ssim', lossFunction=SSIM(n_channels=4, reduction='sum').to(torch.device('cuda')))
# ssimLossOriginal = SSIM(window_size=7, n_channels=4, reduction='mean').to(torch.device('cuda'))

# for data in tqdm.tqdm(dataloader):

#     x = data['x'].to(torch.device('cuda'))
#     y = data['y'].to(torch.device('cuda'))
#     masks = data['targetMask'].to(torch.device('cuda'))
#     tiles = data['tile']
#     cubenames = data['cubename']

#     # Compute prediction and loss
#     pred = model(x)

#     pred = torch.clamp(pred, min=0, max=1)

#     # Save predictions
#     for i in range(len(tiles)):
#         # path = os.path.join(predsFolder, tiles[i])
#         # os.makedirs(path, exist_ok=True)
#         # np.savez_compressed(os.path.join(path, cubenames[i]), highresdynamic=pred[i].permute(2, 3, 0, 1).detach().cpu().numpy().astype(np.float16))
#         start = time.time()
#         tmp = SSIMEarthnetToolkit(preds=pred[i].permute(2, 3, 0, 1).detach().cpu().numpy(), targs=y[i].permute(2, 3, 0, 1).detach().cpu().numpy(), masks=masks[i].repeat(4, 1, 1, 1).permute(2, 3, 0, 1).detach().cpu().numpy())
#         print(time.time() - start)
#         print("SSIMEarthnetToolkit: {}".format(1 - tmp[0]))

#         start = time.time()
#         tmp = ssimLoss(pred[i].unsqueeze(0), y[i].unsqueeze(0), masks[i].unsqueeze(0))
#         print(time.time() - start)
#         print("SSIMMaskedLoss: {}".format(tmp))

#         start = time.time()
#         tmp = 1-ssimLossOriginal(pred[i].unsqueeze(0), y[i].unsqueeze(0))
#         print(time.time() - start)
#         print("SSIM Original: {}".format(tmp))

#         start = time.time()
#         tmp = newSSIM(pred[i].unsqueeze(0), y[i].unsqueeze(0), masks[i].unsqueeze(0))
#         print(time.time() - start)
#         print("SSIM New: {}".format(1 - tmp))

#         start = time.time()
#         tmp = maskedSSIMLoss(pred[i].unsqueeze(0), y[i].unsqueeze(0), masks[i].unsqueeze(0))
#         print(time.time() - start)
#         print("Masked SSIM Loss New : {}".format(tmp))
    
    # earthformerENS(pred.permute(0, 2, 3, 4, 1), y.permute(0, 2, 3, 4, 1), 1-masks.permute(0, 2, 3, 4, 1))

# # Calculate ENS Score the earthformer way
# ensDict = earthformerENS.compute()
# print(ensDict)
# earthformerENS.reset()

# # Calculate ENS Score the earthnet toolkit way
# en.EarthNetScore.get_ENS(pred_dir=predsFolder, targ_dir='/home/nikoskot/earthnetThesis/EarthnetDataset/train', data_output_file=os.path.join(trainingFolder, 'data_output.json'), ens_output_file=os.path.join(trainingFolder, 'score_output.json'))

############################################################################################################
# Print lr schedulers functions
epochs = 30

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
cosineAnnealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=0)
lrCAvalues = []

for i in range(1, epochs + 1):

    optimizer.step()

    lrCA = cosineAnnealing.get_last_lr()
    lrCAvalues.append(lrCA)

    cosineAnnealing.step()

plt.figure()
plt.plot(range(1, epochs + 1), lrCAvalues)
plt.title('cosine annealing')
plt.savefig('cosineAnnealing.png')


optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
cosineAnnealingWarmRest = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=0)
lrCAWRvalues = []

for i in range(1, epochs + 1):

    optimizer.step()

    lrCAWR = cosineAnnealingWarmRest.get_last_lr()
    lrCAWRvalues.append(lrCAWR)

    cosineAnnealingWarmRest.step()

plt.figure()
plt.plot(range(1, epochs + 1), lrCAWRvalues)
plt.title('cosine annealing warm restarts')
plt.savefig('cosineAnnealingWarmRestarts.png')
############################################################################################################