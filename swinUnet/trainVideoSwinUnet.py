import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, random_split, Subset
import torchmetrics
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
from ..channelUnet.maskedLoss import BaseLoss
import tqdm
import time
import datetime
from videoSwinUnetMMAction import VideoSwinUNet
import argparse
import yaml
import os
import logging
import shutil


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='The path to the configuration file.')
    parser.add_argument('--resumeTraining', default=False, action='store_true', help='If we want to resume the training from a checkpoint.')
    parser.add_argument('--resumeCheckpoint', help='The path of the checkpoint to resume training from.')
    parser.add_argument('--note', help='Note to write at beginning of log file.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Training loop
def train_loop(dataloader, model, lossFunction, optimizer):

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    lossSum = 0
    # numberOfSamples = 0

    for X, y, _ in tqdm.tqdm(dataloader):

        # Move data to GPU
        X = X.to(torch.device('cuda'))
        y = y.to(torch.device('cuda'))

        # Compute prediction and loss
        pred = model(X)
        loss = lossFunction(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Add the loss to the total loss of the batch and keep track of the number of samples
        lossSum         += loss.item()
        # numberOfSamples += len(X)

        # Isolate the mask from the ground truth and update the Earthnet Score Calculator
        # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
        # trainENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

    # Calculate Earthnet Score and reset the calculator
    # trainENS = trainENSCalculator.compute()
    # trainENSCalculator.reset()

    return lossSum #/ numberOfSamples#, trainENS
        

def validation_loop(dataloader, model, lossFunction):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSum = 0
    # numberOfSamples = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y, _ in tqdm.tqdm(dataloader):
            
            # Move data to GPU
            X = X.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(X)
            loss = lossFunction(pred, y)

            # Add the loss to the total loss of the batch and keep track of the number of samples
            lossSum         += loss.item()
            # numberOfSamples += len(X)

            # Isolate the mask from the ground truth and update the Earthnet Score Calculator
            # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
            # validENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

    # Calculate Earthnet Score and reset the calculator
    # validationENS = validENSCalculator.compute()
    # validENSCalculator.reset()

    return lossSum #/ numberOfSamples#, validationENS


def main():

    # Get the date and time when the execution started
    runDateTime = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    device = torch.device('cuda')

    # Parse the input arguments and load the configuration file
    args   = parseArgs()
    config = load_config(args.config)

    # Setup the output folder structure
    outputFolder = os.path.join(config['experimentsFolder'], config['modelType'] + '_' + runDateTime) # .../experiments/modelType_trainDatetime/
    os.makedirs(outputFolder, exist_ok=True)

    # Make a copy of the configuration file on the output folder
    shutil.copy(args.config, outputFolder)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(outputFolder, 'training.log'), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger.info("NOTE: {}".format(args.note))

    # Intiialize Video Swin Unet model and move to GPU
    model = VideoSwinUNet(inputChannels=config['modelInputCh'], C=config['C']).to(torch.device('cuda'))

    # Setup Loss Function and Optimizer
    if config['trainLossFunction'] == "mse":
        lossFunction = nn.MSELoss(reduction='mean')
    elif config['trainLossFunction'] == "maskedL1":
        lossFunction = BaseLoss({}, device=device)
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
    preprocessingStage = Preprocessing()

    # Create dataset of training part of Earthnet dataset
    trainDataset = EarthnetTrainDataset(dataDir=config['trainDataDir'], dtype=config['dataDtype'], transform=preprocessingStage)
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

    logger.info("Training Finished")

if __name__ == "__main__":
    main()