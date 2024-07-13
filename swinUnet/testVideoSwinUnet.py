import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, random_split, Subset
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
import tqdm
from videoSwinUnetMMAction import VideoSwinUNet
import os
import argparse
import yaml
import logging

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainingFolder', help='The path to the folder of the training run whose testing we want to execute.')
    parser.add_argument('--testSplit', default='iid_test_split', choices=['iid_test_split', 'ood_test_split', 'seasonal_test_split', 'extreme_test_split'], help='The split of the testing dataset to use.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def testing_loop(dataloader, model, lossFunction, predsFolder):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSum = 0
    numberOfSamples = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y, tiles, cubenames, _ in tqdm.tqdm(dataloader):
            
            # Move data to GPU
            X = X.to(torch.device('cuda'))
            y = y.to(torch.device('cuda'))

            # Compute prediction and loss
            pred = model(X)
            loss = lossFunction(pred, y)

            # Add the loss to the total loss of the batch and keep track of the number of samples
            lossSum         += loss.item()
            numberOfSamples += len(X)

            # Isolate the mask from the ground truth and update the Earthnet Score Calculator
            # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
            # validENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

            # Save predictions
            for i in range(len(tiles)):
                path = os.path.join(predsFolder, tiles[i])
                os.makedirs(path, exist_ok=True)
                np.savez_compressed(os.path.join(path, cubenames[i]), highresdynamic=pred[0].permute(2, 3, 0, 1).detach().cpu().numpy().astype(np.float16))

    # Calculate Earthnet Score and reset the calculator
    # validationENS = validENSCalculator.compute()
    # validENSCalculator.reset()

    return lossSum / numberOfSamples#, validationENS


def main():
    # Parse the input arguments and load the corresponding configuration file
    args   = parseArgs()
    config = load_config(os.path.join(args.trainingFolder, 'config.yml'))

    # Setup the predictions output folder
    predsFolder = os.path.join(args.trainingFolder, 'predictions', args.testSplit) # .../experiments/modelType_trainDatetime/predictions/split/
    os.makedirs(predsFolder, exist_ok=True)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(args.trainingFolder, 'testing_{}.log'.format(args.testSplit)), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    # Load training checkpoint
    checkpoint = torch.load(os.path.join(args.trainingFolder, 'checkpoint.pth'))

    # Intiialize Video Swin Unet model and move to GPU
    model = VideoSwinUNet(inputChannels=config['modelInputCh'], C=config['C']).to(torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])

    # Setup Loss Function
    if config['trainLossFunction'] == "mse":
        lossFunction = nn.MSELoss(reduction='sum')
    else:
        raise NotImplementedError

    # Set Preprocessing for Earthnet data
    preprocessingStage = Preprocessing()

    # Create dataset for testing part of Earthnet dataset
    testDataset = EarthnetTestDataset(dataDir=os.path.join(config['testDataDir'], args.testSplit), dtype=config['dataDtype'], transform=preprocessingStage)
    # testDataset = Subset(testDataset, range(450))

    # Create testing Dataloader
    testDataloader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)  

    testLoss = testing_loop(testDataloader, model, lossFunction, predsFolder)
    logger.info("Mean Testing loss: {}".format(testLoss))

    logger.info("Testing Finished")


if __name__ == "__main__":
    main()