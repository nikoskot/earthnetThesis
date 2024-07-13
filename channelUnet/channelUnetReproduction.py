import segmentation_models_pytorch as smp
import numpy as np
import torch
from torch import nn
import datetime
import argparse
import yaml
import os
import logging
import shutil
import tqdm
from torch.utils.tensorboard import SummaryWriter
from channelUnetDataset import EarthNet2021Dataset
from torch.utils.data import DataLoader, random_split, Subset
from maskedLoss import BaseLoss


model = smp.Unet(encoder_name='densenet161', encoder_weights='imagenet', in_channels=191, classes=80, activation='sigmoid')
upsample = nn.Upsample(size =(128,128))

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='The path to the configuration file.')
    parser.add_argument('--resumeTraining', default=False, action='store_true', help='If we want to resume the training from a checkpoint.')
    parser.add_argument('--resumeCheckpoint', help='The path of the checkpoint to resume training from.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Training loop
def train_loop(dataloader, model, lossFunction, optimizer, device):

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    lossSum = 0
    # numberOfSamples = 0

    for data in tqdm.tqdm(dataloader):

        satimgs = data["dynamic"][0][:,:10,...]

        b, t, c, h, w = satimgs.shape

        satimgs = satimgs.reshape(b, t*c, h, w)

        dem = data["static"][0]
        clim = data["dynamic"][1][:,:,:5,...]
        b, t, c, h2, w2 = clim.shape
        clim = clim.reshape(b, t//5, 5, c, h2, w2).mean(2)[:,:,:,39:41,39:41].reshape(b, t//5 * c, 2, 2)

        inputs = torch.cat((satimgs, dem, upsample(clim)), dim = 1)

        # Move data to GPU
        inputs = inputs.to(device)

        # Compute prediction and loss
        pred = model(inputs).reshape(b, 20, 4, h, w)
        loss, lossLogs = lossFunction(pred, inputs, None, None)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Add the loss to the total loss of the batch and keep track of the number of samples
        lossSum += loss.item()
        # numberOfSamples += len(X)
        # numberOfSamples += torch.numel(X)

        # Isolate the mask from the ground truth and update the Earthnet Score Calculator
        # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
        # trainENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

    # Calculate Earthnet Score and reset the calculator
    # trainENS = trainENSCalculator.compute()
    # trainENSCalculator.reset()

    return lossSum #/ numberOfSamples#, trainENS
        

def validation_loop(dataloader, model, lossFunction, device):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSum = 0
    # numberOfSamples = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):

            satimgs = data["dynamic"][0][:,:10,...]

            b, t, c, h, w = satimgs.shape

            satimgs = satimgs.reshape(b, t*c, h, w)

            dem = data["static"][0]
            clim = data["dynamic"][1][:,:,:5,...]
            b, t, c, h2, w2 = clim.shape
            clim = clim.reshape(b, t//5, 5, c, h2, w2).mean(2)[:,:,:,39:41,39:41].reshape(b, t//5 * c, 2, 2)

            inputs = torch.cat((satimgs, dem, upsample(clim)), dim = 1)
            
            # Move data to GPU
            inputs = inputs.to(device)

            # Compute prediction and loss
            pred = model(inputs).reshape(b, 20, 4, h, w)
            loss, lossLogs = lossFunction(pred, inputs, None, None)

            # Add the loss to the total loss of the batch and keep track of the number of samples
            lossSum += loss.item()
            # numberOfSamples += len(X)
            # numberOfSamples += torch.numel(X)

            # Isolate the mask from the ground truth and update the Earthnet Score Calculator
            # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
            # validENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

    # Calculate Earthnet Score and reset the calculator
    # validationENS = validENSCalculator.compute()
    # validENSCalculator.reset()

    return lossSum #/ numberOfSamples#, validationENS


def main():

    # Get the date and time when the execution started
    runDateTime = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse the input arguments and load the configuration file
    args   = parseArgs()
    config = load_config(args.config)

    # Setup the output folder structure
    outputFolder = os.path.join(config['experimentsFolder'], 'ChannelUnet_' + runDateTime) # .../experiments/modelType_trainDatetime/
    os.makedirs(outputFolder, exist_ok=True)

    # Make a copy of the configuration file on the output folder
    shutil.copy(args.config, outputFolder)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(outputFolder, 'training.log'), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    tbWriter = SummaryWriter(os.path.join(config['experimentsFolder'], 'tensorboardLogs', 'ChannelUnet_' + runDateTime))

    # Intiialize Video Swin Unet model and move to GPU
    model = smp.Unet(encoder_name='densenet161', encoder_weights='imagenet', in_channels=191, classes=80, activation='sigmoid').to(device)

    # Setup Loss Function and Optimizer
    if config['trainLossFunction'] == "mse":
        lossFunction = nn.MSELoss(reduction='sum')
    elif config['trainLossFunction'] == "maskedL1":
        lossFunction = BaseLoss()
    else:
        raise NotImplementedError
    
    if config['trainingOptimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    elif config['trainingOptimizer'] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[40, 70, 90], gamma=0.1)
    else:
        raise NotImplementedError
    
    if args.resumeTraining:
        checkpoint = torch.load(args.resumeCheckpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        startEpoch = checkpoint['epoch'] + 1
        bestValLoss = checkpoint['valLoss']
        logger.info("Resuming training from checkpoint {} from epoch {}".format(args.resumeCheckpoint, startEpoch))
    else:
        startEpoch = 0
        bestValLoss = np.inf
        logger.info("Starting training from from epoch 0")

    # Set Preprocessing for Earthnet data
    preprocessingStage = None

    # Create dataset of training part of Earthnet dataset
    trainDataset = EarthNet2021Dataset(folder=config['trainDataDir'], dtype=config['dataDtype'], noisy_masked_pixels = False, use_meso_static_as_dynamic = False, fp16 = False)
    trainDataset = Subset(trainDataset, range(3))

    # Split to training and validation dataset
    # trainDataset, valDataset = random_split(trainDataset, [config['trainSplit'], config['validationSplit']])

    # Create training and validation Dataloaders
    trainDataloader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)
    valDataloader   = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)

    # Create objects for calculation of Earthnet Score during training, validation and testing
    # trainENSCalculator = EarthNet2021ScoreUpdateWithoutCompute(layout="NHWCT", eps=1E-4)
    # validENSCalculator = EarthNet2021ScoreUpdateWithoutCompute(layout="NHWCT", eps=1E-4)
    # testENSCalculator  = EarthNet2021ScoreUpdateWithoutCompute(layout="NHWCT", eps=1E-4)

    for i in range(startEpoch, config['epochs']):

        logger.info("Epoch {}\n-------------------------------".format(i))
        trainLoss = train_loop(trainDataloader, model, lossFunction, optimizer)
        logger.info("Mean training loss: {}".format(trainLoss))
        # print("Training ENS metrics:")
        # print(trainENS)

        valLoss = validation_loop(valDataloader, model, lossFunction)
        logger.info("Mean validation loss: {}".format(valLoss))
        # print("Validation ENS metrics:")
        # print(valENS)

        # Early stopping based on validation loss
        if valLoss < bestValLoss:

            bestValLoss = valLoss
            checkpoint  = {'state_dict' : model.state_dict(),
                           'modelType'  : config['modelType'],
                           'epoch'      : i,
                        #    'bestValENS' : bestValENS['EarthNetScore'],
                         #   'valMAD'     : bestValENS['MAD'],
                         #   'valOLS'     : bestValENS['OLS'],
                         #   'valEMD'     : bestValENS['EMD'],
                         #   'valSSIM'    : bestValENS['SSIM'],
                           'valLoss'    : valLoss,
                           'trainLoss'  : trainLoss,
                           'optimizer'  : optimizer.state_dict(),
                           'scheduler'  : scheduler.state_dict()}
            torch.save(checkpoint, os.path.join(outputFolder, 'checkpoint.pth'))
            logger.info("New best validation Loss {}, at epoch {}".format(bestValLoss, i))
        
        scheduler.step()

        tbWriter.add_scalar('Loss/Train', trainLoss, i)
        tbWriter.add_scalar('Loss/Val', valLoss, i)

    logger.info("Training Finished")
    tbWriter.flush()

if __name__ == "__main__":
    main()