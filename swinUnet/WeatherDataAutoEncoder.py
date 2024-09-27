import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
# from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, random_split, Subset
import torchmetrics
import tqdm
import time
import datetime
from earthnet.parallel_score import CubeCalculator
import yaml
import rerun as rr
from earthnet.plot_cube import gallery, colorize

def timeUpsample(x, factor):
    """ 
    Args:
        x: (B, C, T, H, W)
    """
    _, _, T, H, W = x.shape
    return F.interpolate(input=x, size=(T*factor, H, W), mode='trilinear')


class WeatherDataAutoEncoder(nn.Module):
    def __init__(self, config, logger):
        super().__init__()
        self.config = config

        self.enConv1 = nn.Conv3d(in_channels=5, out_channels=48, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.enConv2 = nn.Conv3d(in_channels=48, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.avgpool1 = nn.AvgPool3d(kernel_size=(5, 2, 2), stride=(5, 2, 2), padding=(0, 0, 0))
        self.enConv3 = nn.Conv3d(in_channels=96, out_channels=192, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True)
        self.avgpool2 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        
        self.upsampe1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.deConv1 = nn.Conv3d(in_channels=192, out_channels=96, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True)
        self.upsampe2 = nn.Upsample(scale_factor=(5, 2, 2), mode='nearest')
        self.deConv2 = nn.Conv3d(in_channels=96, out_channels=48, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True)
        self.deConv3 = nn.Conv3d(in_channels=48, out_channels=5, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=True)
        

    def forward(self, x):

        # Input shape B,C,T,H,W
        x = self.enConv1(x)
        # print("Encoder conv 1 output shape {}".format(x.shape))

        x = self.enConv2(x)
        # print("Encoder conv 2 output shape {}".format(x.shape))

        x = self.avgpool1(x)
        # print("Encoder avg pooling 1 output shape {}".format(x.shape))

        x = self.enConv3(x)
        # print("Encoder conv 3 output shape {}".format(x.shape))

        x = self.avgpool2(x)
        # print("Encoder avg pooling 2 output shape {}".format(x.shape))

        x = self.upsampe1(x)
        # print("Decoder upsampling 1 output shape {}".format(x.shape))

        x = self.deConv1(x)
        # print("Dencoder conv 1 output shape {}".format(x.shape))

        x = self.upsampe2(x)
        # print("Decoder upsampling 2 output shape {}".format(x.shape))

        x = self.deConv2(x)
        # print("Dencoder conv 2 output shape {}".format(x.shape))

        x = self.deConv3(x)
        # print("Dencoder conv 3 output shape {}".format(x.shape))

        return x


if __name__ == "__main__":

    def load_config(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config('/home/nikoskot/earthnetThesis/experiments/configWeatherDataAE.yml')

    startTime = time.time()
    model = WeatherDataAutoEncoder(config, logger=None).to(torch.device('cuda'))
    endTime = time.time()
    print("Model creation time {}".format(endTime - startTime))
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    model.train()

    totalParams = sum(p.numel() for p in model.parameters())

    startTime = time.time()
    contextWeather = torch.rand(1, 5, 50, 80, 80).to(torch.device('cuda')) # B, C, T, H, W
    endTime = time.time()
    print("Dummy data creation time {}".format(endTime - startTime))

    # torch.cuda.memory._record_memory_history(max_entries=100000)
    
    for i in range(20):
        _ = torch.randn(1).cuda()        
        startTime = time.time()
        y = model(contextWeather)
        endTime = time.time()
        print("Pass time {}".format(endTime - startTime))
        optimizer.zero_grad(set_to_none=True)

    # torch.cuda.memory._dump_snapshot("videoSwinUnetV1.pickle")
    # torch.cuda.memory._record_memory_history(enabled=None)

    y = model(contextWeather)

    print("Output shape {}".format(y.shape))

    print("Number of model parameters {}".format(totalParams))


def train_loop(dataloader, 
               model, 
               l1Loss, 
               ssimLoss, 
               mseLoss,
               vggLoss,
               optimizer, 
               config, 
               logger, 
               trainVisualizationCubenames,
               epoch):
    
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    l1LossSum = 0
    mseLossSum = 0
    SSIMLossSum = 0
    vggLossSum = 0

    batchNumber = 0

    maxGradPre = torch.tensor(0)
    minGradPre = torch.tensor(0)

    for data in tqdm.tqdm(dataloader):
        batchNumber += 1

        optimizer.zero_grad()

        # Move data to GPU
        
        contextWeather = data['contextWeather'].to(torch.device('cuda'))
        y = data['contextWeather'].to(torch.device('cuda'))
        mask = torch.ones(size=(y.shape[0], 1, y.shape[2], y.shape[3], y.shape[3])).to(torch.device('cuda'))

        # Compute prediction and loss
        pred = model(contextWeather)
        # pred = torch.clamp(pred, min=0.0, max=1.0)

        l1LossValue = l1Loss(pred, y, mask)
        if config['useMSE']:
            mseLossValue = mseLoss(pred, y, mask)
        if config['useSSIM']:
            ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, mask)
        if config['useVGG']:
            vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), mask)

        # Backpropagation
        totalLoss = config['l1Weight'] * l1LossValue
        if config['useMSE']:
            totalLoss = totalLoss + (config['mseWeight'] * mseLossValue)
        if config['useSSIM']:
            totalLoss = totalLoss + (config['ssimWeight'] * ssimLossValue)
        if config['useVGG']:
            totalLoss = totalLoss + (config['vggWeight'] * vggLossValue)

        totalLoss.backward()

        for param in model.parameters():
            maxGradPre = torch.maximum(maxGradPre, torch.max(param.grad.view(-1)))
            minGradPre = torch.minimum(minGradPre, torch.min(param.grad.view(-1)))

        if config['gradientClipping']:
            nn.utils.clip_grad_value_(model.parameters(), config['gradientClipValue'])

        optimizer.step()

        # Add the loss to the total loss 
        l1LossSum += l1LossValue.item()
        if config['useSSIM']:
            SSIMLossSum += ssimLossValue.item()
        if config['useMSE']:
            mseLossSum += mseLossValue.item()
        if config['useVGG']:
            vggLossSum += vggLossValue.item()

        # # Log visualizations if available
        if epoch % config['visualizationFreq'] == 0 or epoch == 1 or epoch == config['epochs']:
            rr.set_time_sequence('epoch', epoch)
            for i in range(len(data['cubename'])):
                if data['cubename'][i] in trainVisualizationCubenames:

        #             # # Log ground truth
        #             # targ = data['y'][i, ...].numpy()
        #             # targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
        #             # targ[targ < 0] = 0
        #             # targ[targ > 0.5] = 0.5
        #             # targ = 2 * targ
        #             # targ[np.isnan(targ)] = 0
        #             # grid = gallery(targ)
        #             # rr.log('/train/groundTruth/{}'.format(data['cubename'][i]), rr.Image(grid))

                    # Log prediction
                    # variable rr
                    targ = pred[i, ...].detach().cpu().numpy()[0, :, :, :]#.transpose(1, 2 ,3 ,0)
                    targ = colorize(targ, colormap="Blues", mask_red=np.isnan(targ))
                    grid = gallery(targ)
                    rr.log('/train/prediction/precipitaion/{}'.format(data['cubename'][i]), rr.Image(grid))
                    # variable pp
                    targ = pred[i, ...].detach().cpu().numpy()[1, :, :, :]#.transpose(1, 2 ,3 ,0)
                    targ = colorize(targ, colormap="rainbow", mask_red=np.isnan(targ))
                    grid = gallery(targ)
                    rr.log('/train/prediction/pressure/{}'.format(data['cubename'][i]), rr.Image(grid))
                    # variable tg
                    targ = pred[i, ...].detach().cpu().numpy()[2, :, :, :]#.transpose(1, 2 ,3 ,0)
                    targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
                    grid = gallery(targ)
                    rr.log('/train/prediction/meanTmp/{}'.format(data['cubename'][i]), rr.Image(grid))
                    # variable tn
                    targ = pred[i, ...].detach().cpu().numpy()[3, :, :, :]#.transpose(1, 2 ,3 ,0)
                    targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
                    grid = gallery(targ)
                    rr.log('/train/prediction/minimumTmp/{}'.format(data['cubename'][i]), rr.Image(grid))
                    # variable tx
                    targ = pred[i, ...].detach().cpu().numpy()[4, :, :, :]#.transpose(1, 2 ,3 ,0)
                    targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
                    grid = gallery(targ)
                    rr.log('/train/prediction/maximumTmp/{}'.format(data['cubename'][i]), rr.Image(grid))
    
        logger.info("Maximum gradient before clipping: {}".format(maxGradPre))
        logger.info("Minimum gradient before clipping: {}".format(minGradPre))

    return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber

def validation_loop(dataloader, 
                    model, 
                    l1Loss, 
                    ssimLoss, 
                    mseLoss,
                    vggLoss,
                    config, 
                    validationVisualizationCubenames,
                    epoch,
                    ensCalculator,
                    logger):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    l1LossSum = 0
    SSIMLossSum = 0
    mseLossSum = 0
    vggLossSum = 0
    earthnetScore = 0

    batchNumber = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            batchNumber += 1
            
            # Move data to GPU
            contextWeather = data['contextWeather'].to(torch.device('cuda'))
            y = data['contextWeather'].to(torch.device('cuda'))
            mask = torch.ones(size=(y.shape[0], 1, y.shape[2], y.shape[3], y.shape[3])).to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(contextWeather)
            # pred = torch.clamp(pred, min=0.0, max=1.0)

            l1LossValue = l1Loss(pred, y, mask)
            if config['useMSE']:
                mseLossValue = mseLoss(pred, y, mask)
            if config['useSSIM']:
                ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, mask)
            if config['useVGG']:
                vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), mask)

            # Add the loss to the total loss
            l1LossSum += l1LossValue.item()
            if config['useSSIM']:
                SSIMLossSum += ssimLossValue.item()
            if config['useMSE']:
                mseLossSum += mseLossValue.item()
            if config['useVGG']:
                vggLossSum += vggLossValue.item()

            # Log visualizations if available
            if epoch % config['visualizationFreq'] == 0 or epoch == 1 or epoch == config['epochs']:
                rr.set_time_sequence('epoch', epoch)
                for i in range(len(data['cubename'])):
                    if data['cubename'][i] in validationVisualizationCubenames:

            #             # # Log ground truth
            #             # targ = data['y'][i, ...].numpy()
            #             # targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
            #             # targ[targ < 0] = 0
            #             # targ[targ > 0.5] = 0.5
            #             # targ = 2 * targ
            #             # targ[np.isnan(targ)] = 0
            #             # grid = gallery(targ)
            #             # rr.log('/train/groundTruth/{}'.format(data['cubename'][i]), rr.Image(grid))

                        # Log prediction
                        # variable rr
                        targ = pred[i, ...].detach().cpu().numpy()[0, :, :, :]#.transpose(1, 2 ,3 ,0)
                        targ = colorize(targ, colormap="Blues", mask_red=np.isnan(targ))
                        grid = gallery(targ)
                        rr.log('/validation/prediction/precipitaion/{}'.format(data['cubename'][i]), rr.Image(grid))
                        # variable pp
                        targ = pred[i, ...].detach().cpu().numpy()[1, :, :, :]#.transpose(1, 2 ,3 ,0)
                        targ = colorize(targ, colormap="rainbow", mask_red=np.isnan(targ))
                        grid = gallery(targ)
                        rr.log('/validation/prediction/pressure/{}'.format(data['cubename'][i]), rr.Image(grid))
                        # variable tg
                        targ = pred[i, ...].detach().cpu().numpy()[2, :, :, :]#.transpose(1, 2 ,3 ,0)
                        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
                        grid = gallery(targ)
                        rr.log('/validation/prediction/meanTmp/{}'.format(data['cubename'][i]), rr.Image(grid))
                        # variable tn
                        targ = pred[i, ...].detach().cpu().numpy()[3, :, :, :]#.transpose(1, 2 ,3 ,0)
                        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
                        grid = gallery(targ)
                        rr.log('/validation/prediction/minimumTmp/{}'.format(data['cubename'][i]), rr.Image(grid))
                        # variable tx
                        targ = pred[i, ...].detach().cpu().numpy()[4, :, :, :]#.transpose(1, 2 ,3 ,0)
                        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
                        grid = gallery(targ)
                        rr.log('/validation/prediction/maximumTmp/{}'.format(data['cubename'][i]), rr.Image(grid))

    return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber, earthnetScore