import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, PreprocessingV2
from torch.utils.data import DataLoader, random_split, Subset
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
import tqdm
from earthnetThesis.swinUnet.videoSwinUnetMMActionV2 import VideoSwinUNet
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


def testing_loop(dataloader, model, lossFunction, predsFolder, device):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSum = 0
    numberOfSamples = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y, tiles, cubenames, xMesodynamic in tqdm.tqdm(dataloader):
            
            # Move data to GPU
            X            = X.to(device)
            y            = y.to(device)
            xMesodynamic = xMesodynamic.to(device)

            # Compute prediction and loss
            pred = model(X, xMesodynamic)
            loss = lossFunction(pred, y)

            # Add the loss to the total loss of the batch and keep track of the number of samples
            lossSum         += loss.item()
            # numberOfSamples += len(X)
            numberOfSamples += torch.numel(X)

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

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup the predictions output folder
    predsFolder = os.path.join(args.trainingFolder, 'predictions', args.testSplit) # .../experiments/modelType_trainDatetime/predictions/split/
    os.makedirs(predsFolder, exist_ok=True)

    # Initialize logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=os.path.join(args.trainingFolder, 'testing_{}.log'.format(args.testSplit)), encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    logger.info("Testing on {}.".format(device.type))

    # Load training checkpoint
    checkpoint = torch.load(os.path.join(args.trainingFolder, 'checkpoint.pth'), map_location=device)

    # Intiialize Video Swin Unet model and move to GPU
    model = VideoSwinUNet(mainInputChannels=config['modelInputCh'], 
                            extraInputChannels=config['extraInputCh'], 
                            mainInputTimeDimension=config['mainInputTime'], 
                            inputHW=config['inputHeightWidth'], 
                            C=config['C'], 
                            window_size=(config['windowSizeT'], config['windowSizeH'], config['windowSizeW'])).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    # Setup Loss Function
    if config['trainLossFunction'] == "mse":
        lossFunction = nn.MSELoss(reduction='sum')
    else:
        raise NotImplementedError

    # Set Preprocessing for Earthnet data
    preprocessingStage = PreprocessingV2()

    # Create dataset for testing part of Earthnet dataset
    # testDataset = EarthnetTestDataset(dataDir=os.path.join(config['testDataDir'], args.testSplit), dtype=config['dataDtype'], transform=preprocessingStage)
    # testDataset = Subset(testDataset, range(450))
    testDataset = EarthnetTrainDataset(dataDir=config['trainDataDir'], dtype=config['dataDtype'], transform=preprocessingStage)
    testDataset = Subset(testDataset, range(320))

    # Create testing Dataloader
    testDataloader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)  

    testLoss = testing_loop(testDataloader, model, lossFunction, predsFolder, device)
    logger.info("Mean Testing loss: {}".format(testLoss))

    logger.info("Testing Finished")


if __name__ == "__main__":
    main()