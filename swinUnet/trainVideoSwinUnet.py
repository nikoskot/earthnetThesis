import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
import random
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing, PreprocessingStack
from torch.utils.data import DataLoader, random_split, Subset
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
import losses
import tqdm
import datetime
from videoSwinUnetMMActionV1 import VideoSwinUNet
import argparse
import yaml
import os
import logging
import shutil
import rerun as rr
from torch.utils.tensorboard import SummaryWriter
from piqa import SSIM
from focal_frequency_loss import FocalFrequencyLoss as FFL


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/nikoskot/earthnetThesis/experiments/config.yml', help='The path to the configuration file.')
    parser.add_argument('--resumeTraining', default=False, action='store_true', help='If we want to resume the training from a checkpoint.')
    parser.add_argument('--resumeCheckpoint', help='The path of the checkpoint to resume training from.')
    parser.add_argument('--note', help='Note to write at beginning of log file.')
    parser.add_argument('--seed', default=88, type=int)
    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Training loop
def train_loop(dataloader, model, lossFunction, optimizer, config, logger):

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    lossSum = 0

    maxGradPre = torch.tensor(0)
    minGradPre = torch.tensor(0)

    for data in tqdm.tqdm(dataloader):

        optimizer.zero_grad()

        # Move data to GPU
        x = data['x'].to(torch.device('cuda'))
        y = data['y'].to(torch.device('cuda'))
        masks = data['targetMask'].to(torch.device('cuda'))

        # Compute prediction and loss
        pred = model(x)
        # loss = lossFunction(pred, y, masks)
        loss = lossFunction(pred, y)
        # print("Train loss: {}".format(loss))

        # Backpropagation
        loss.backward()

        for param in model.parameters():
            maxGradPre = torch.maximum(maxGradPre, torch.max(param.grad.view(-1)))
            minGradPre = torch.minimum(minGradPre, torch.min(param.grad.view(-1)))

        if config['gradientClipping']:
            nn.utils.clip_grad_value_(model.parameters(), config['gradientClipValue'])

        optimizer.step()

        # Add the loss to the total loss 
        lossSum += loss.item()
        # print("Train loss: {}".format(loss.item()))
    
    logger.info("Maximum gradient before clipping: {}".format(maxGradPre))
    logger.info("Minimum gradient before clipping: {}".format(minGradPre))

    return lossSum
        

def validation_loop(dataloader, model, lossFunction, config):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSum = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            
            # Move data to GPU
            x = data['x'].to(torch.device('cuda'))
            y = data['y'].to(torch.device('cuda'))
            masks = data['targetMask'].to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(x)
            # loss = lossFunction(pred, y, masks)
            loss = lossFunction(pred, y)
            # print("Val loss: {}".format(loss))

            # Add the loss to the total loss
            lossSum += loss.item()
            # print("Val loss: {}".format(loss.item()))

    return lossSum

def main():

    # Get the date and time when the execution started
    runDateTime = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    # Set device
    device = torch.device('cuda')

    # Parse the input arguments and load the configuration file
    args   = parseArgs()
    config = load_config(args)

    # Setup the output folder structure
    outputFolder = os.path.join(config['experimentsFolder'], config['modelType'] + '_' + runDateTime) # .../experiments/modelType_trainDatetime/
    os.makedirs(outputFolder, exist_ok=True)

    # Make a copy of the configuration file on the output folder
    shutil.copy(args.config, outputFolder)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(outputFolder, 'training.log'), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger.info("NOTE: {}".format(args.note))
    tbWriter = SummaryWriter(outputFolder)

    # Setup seeds
    logger.info("Torch, random, numpy seed: {}".format(args.seed))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Intiialize Video Swin Unet model and move to GPU
    model = VideoSwinUNet(config, logger).to(device)

    # Setup Loss Function and Optimizer
    # lossFunction = losses.MaskedLoss(lossType='l1', lossFunction=nn.L1Loss(reduction='sum'), maskFlag=True)
    lossFunction = nn.L1Loss(reduce='mean')

    if config['trainingOptimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    elif config['trainingOptimizer'] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=config['lrScaleFactor'], patience=config['schedulerPatience'], min_lr=0.000001)
    else:
        logger.info('No scheduler used')
    
    # Load model in case we want to resume training
    if args.resumeTraining:
        checkpoint = torch.load(args.resumeCheckpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['lr']
        if config['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
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
    trainDataset = EarthnetTrainDataset(dataDir=config['trainDataDir'], dtype=config['dataDtype'], transform=preprocessingStage, cropMesodynamic=config['cropMesodynamic'])
    if isinstance(config['trainDataSubset'], int):
        trainDataset = Subset(trainDataset, range(config['trainDataSubset']))

    # Create training and validation Dataloaders
    if config['overfitTraining']:
        # If we want to overfit the model to a subset of the training data
        trainDataloader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)
        valDataloader   = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)
    else:
        # Normal training
        trainDataset, valDataset = random_split(trainDataset, [config['trainSplit'], config['validationSplit']])
        # Split to training and validation dataset
        trainDataloader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)
        valDataloader   = DataLoader(valDataset,   batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)


    for i in range(startEpoch, config['epochs']):

        logger.info("Epoch {}\n-------------------------------".format(i))
        for j, param_group in enumerate(optimizer.param_groups):
            logger.info("Learning Rate of group {}: {}".format(j, param_group['lr']))

        trainLoss = train_loop(trainDataloader, model, lossFunction, optimizer, config, logger)
        logger.info("Mean training loss: {}".format(trainLoss))

        valLoss = validation_loop(valDataloader, model, lossFunction, config)
        logger.info("Mean validation loss: {}".format(valLoss))

        # Save checkpoint based on validation loss
        if valLoss < bestValLoss:

            bestValLoss = valLoss
            checkpoint  = {'state_dict' : model.state_dict(),
                           'modelType'  : config['modelType'],
                           'epoch'      : i,
                           'valLoss'    : valLoss,
                           'optimizer'  : optimizer.state_dict(),
                           'scheduler'  : scheduler.state_dict() if config['scheduler'] is not None else None
                           }
            
            torch.save(checkpoint, os.path.join(outputFolder, 'checkpoint.pth'))
            logger.info("New best validation Loss {}, at epoch {}".format(bestValLoss, i))

        if config['scheduler'] is not None:
            scheduler.step(valLoss)
        
        tbWriter.add_scalar('Loss/Train', trainLoss, i)
        tbWriter.add_scalar('Loss/Val', valLoss, i)

    logger.info("Training Finished")
    tbWriter.flush()

if __name__ == "__main__":
    main()