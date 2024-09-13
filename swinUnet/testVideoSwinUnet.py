import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing, PreprocessingStack
from torch.utils.data import DataLoader, random_split, Subset
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
import tqdm
from videoSwinUnetMMActionV1 import VideoSwinUNet
import os
import argparse
import yaml
import logging
from losses import MaskedLoss, maskedSSIMLoss
import random
from piqa import SSIM
from focal_frequency_loss import FocalFrequencyLoss as FFL

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainingFolder', help='The path to the folder of the training run whose testing we want to execute.')
    parser.add_argument('--testSplit', default='iid_test_split', choices=['iid_test_split', 'ood_test_split', 'seasonal_test_split', 'extreme_test_split'], help='The split of the testing dataset to use.')
    parser.add_argument('--note', help='Note to write at beginning of log file.')
    parser.add_argument('--seed', default=88, type=int)
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def testing_loop(dataloader, 
                model, 
                l1Loss, 
                ssimLoss, 
                mseLoss,
                predsFolder):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    l1LossSum = 0
    SSIMLossSum = 0
    mseLossSum = 0

    batchNumber = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            
            batchNumber += 1
            
            # Move data to GPU
            x = data['x'].to(torch.device('cuda'))
            y = data['y'].to(torch.device('cuda'))
            masks = data['targetMask'].to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(x)
            # pred = torch.clamp(pred, min=0.0, max=1.0)

            l1LossValue = l1Loss(pred, y, masks)
            mseLossValue = mseLoss(pred, y, masks)
            ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, masks)

            # Add the loss to the total loss
            l1LossSum += l1LossValue.item()
            SSIMLossSum += ssimLossValue.item()
            mseLossSum += mseLossValue.item()

            tiles = data['tile']
            cubenames = data['cubename']

            # Save predictions
            for i in range(len(tiles)):
                path = os.path.join(predsFolder, tiles[i])
                os.makedirs(path, exist_ok=True)
                np.savez_compressed(os.path.join(path, cubenames[i]), highresdynamic=pred[i].permute(2, 3, 0, 1).detach().cpu().numpy().astype(np.float16))

    return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber


def main():
    # Parse the input arguments and load the corresponding configuration file
    args   = parseArgs()
    config = load_config(os.path.join(args.trainingFolder, 'savedConfig.yml'))

    device = torch.device('cuda')

    # Setup the predictions output folder
    predsFolder = os.path.join(args.trainingFolder, 'predictions', args.testSplit) # .../experiments/modelType_trainDatetime/predictions/split/
    os.makedirs(predsFolder, exist_ok=True)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(args.trainingFolder, 'testing_{}.log'.format(args.testSplit)), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger.info("NOTE: {}".format(args.note))

    # Setup seeds
    logger.info("Torch, random, numpy seed: {}".format(args.seed))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load training checkpoint
    checkpoint = torch.load(os.path.join(args.trainingFolder, 'checkpoint.pth'))

    # Intiialize Video Swin Unet model and move to GPU
    model = VideoSwinUNet(config, logger).to(torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])

    # Setup Loss Function
    l1Loss = MaskedLoss(lossType='l1', lossFunction=nn.L1Loss(reduction='sum'))
    mseLoss = MaskedLoss(lossType='mse', lossFunction=nn.MSELoss(reduction='sum'))
    ssimLoss = maskedSSIMLoss

    # Set Preprocessing for Earthnet data
    if config['modelInputCh'] == 11:
        preprocessingStage = Preprocessing()
    elif config['modelInputCh'] == 81:
        preprocessingStage = PreprocessingStack()

    # Create dataset for testing part of Earthnet dataset
    if config['overfitTraining']:
        testDataset = EarthnetTrainDataset(dataDir=config['trainDataDir'], dtype=config['dataDtype'], transform=preprocessingStage)
        testDataset = Subset(testDataset, np.random.randint(0, len(testDataset), size=config['trainDataSubset']))
    else:
        testDataset = EarthnetTestDataset(dataDir=os.path.join(config['testDataDir'], args.testSplit), dtype=config['dataDtype'], transform=preprocessingStage)

    # Create testing Dataloader
    testDataloader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)  

    testL1Loss, testSSIMLoss, testMSELoss = testing_loop(testDataloader,
                                                         model,
                                                         l1Loss,
                                                         ssimLoss,
                                                         mseLoss,
                                                         predsFolder)

    logger.info("Mean testing L1 loss: {}".format(testL1Loss))
    logger.info("Mean testing SSIM loss: {}".format(testSSIMLoss))
    logger.info("Mean testing MSE loss: {}".format(testMSELoss))

    logger.info("Testing Finished")


if __name__ == "__main__":
    main()