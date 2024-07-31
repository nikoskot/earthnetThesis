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
import losses
import random
from piqa import SSIM
from focal_frequency_loss import FocalFrequencyLoss as FFL

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainingFolder', help='The path to the folder of the training run whose testing we want to execute.')
    parser.add_argument('--testSplit', default='iid_test_split', choices=['iid_test_split', 'ood_test_split', 'seasonal_test_split', 'extreme_test_split'], help='The split of the testing dataset to use.')
    parser.add_argument('--note', help='Note to write at beginning of log file.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def testing_loop(dataloader, model, lossFunctions, predsFolder, config):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSums = {l:0 for l in config['trainLossFunctions']}

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
            losses = {l:lossFunctions[l](pred, y, masks) for l in config['trainLossFunctions']}

            # Add the loss to the total loss of the batch and keep track of the number of samples
            for l in config['trainLossFunctions']:
                lossSums[l] += losses[l]

            tiles = data['tile']
            cubenames = data['cubename']

            # Save predictions
            for i in range(len(tiles)):
                path = os.path.join(predsFolder, tiles[i])
                os.makedirs(path, exist_ok=True)
                np.savez_compressed(os.path.join(path, cubenames[i]), highresdynamic=pred[i].permute(2, 3, 0, 1).detach().cpu().numpy().astype(np.float16))

    return lossSums


def main():
    # Parse the input arguments and load the corresponding configuration file
    args   = parseArgs()
    config = load_config(os.path.join(args.trainingFolder, 'config.yml'))

    device = torch.device('cuda')

    # Setup seeds
    torch.manual_seed(config['torchSeed'])
    random.seed(config['pythonSeed'])
    np.random.seed(config['numpySeed'])

    # Setup the predictions output folder
    predsFolder = os.path.join(args.trainingFolder, 'predictions', args.testSplit) # .../experiments/modelType_trainDatetime/predictions/split/
    os.makedirs(predsFolder, exist_ok=True)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(args.trainingFolder, 'testing_{}.log'.format(args.testSplit)), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logger.info("NOTE: {}".format(args.note))
    
    # Load training checkpoint
    checkpoint = torch.load(os.path.join(args.trainingFolder, 'checkpoint.pth'))

    # Intiialize Video Swin Unet model and move to GPU
    model = VideoSwinUNet(config, logger).to(torch.device('cuda'))
    model.load_state_dict(checkpoint['state_dict'])

    # Setup Loss Function
    lossFunctions = losses.setupLossFunctions(config, device)

    # Set Preprocessing for Earthnet data
    preprocessingStage = Preprocessing()

    # Create dataset for testing part of Earthnet dataset
    if config['overfitTraining']:
        testDataset = EarthnetTrainDataset(dataDir=config['trainDataDir'], dtype=config['dataDtype'], transform=preprocessingStage)
        testDataset = Subset(testDataset, range(config['trainDataSubset']))
    else:
        testDataset = EarthnetTestDataset(dataDir=os.path.join(config['testDataDir'], args.testSplit), dtype=config['dataDtype'], transform=preprocessingStage)

    # Create testing Dataloader
    testDataloader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)  

    testLosses = testing_loop(testDataloader, model, lossFunctions, predsFolder, config)
    for l in config['trainLossFunctions']:
        logger.info("Mean validation loss {}: {}".format(l, testLosses[l]))

    logger.info("Testing Finished")


if __name__ == "__main__":
    main()