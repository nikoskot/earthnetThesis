import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from channelUnetDataset import EarthNet2021Dataset
from torch.utils.data import DataLoader, random_split, Subset
import tqdm
import os
import argparse
import yaml
import logging
import segmentation_models_pytorch as smp
from maskedLoss import BaseLoss


upsample = nn.Upsample(size =(128,128))

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
            # loss, lossLogs = lossFunction(pred, data, None, None)

            # Add the loss to the total loss of the batch and keep track of the number of samples
            # lossSum         += loss.item()
            # numberOfSamples += len(X)
            # numberOfSamples += torch.numel(X)

            # Isolate the mask from the ground truth and update the Earthnet Score Calculator
            # mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
            # validENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

            # Save predictions
            for i in range(pred.shape[0]):
                path = predsFolder
                os.makedirs(path, exist_ok=True)
                np.savez_compressed(os.path.join(path, data['cubename'][i]), highresdynamic=pred[i].permute(2, 3, 1, 0).detach().cpu().numpy().astype(np.float16))

    # Calculate Earthnet Score and reset the calculator
    # validationENS = validENSCalculator.compute()
    # validENSCalculator.reset()

    return lossSum# / numberOfSamples#, validationENS


def main():
    # Parse the input arguments and load the corresponding configuration file
    args   = parseArgs()
    config = load_config(os.path.join(args.trainingFolder, 'config.yml'))

    # Setup device
    device = torch.device('cuda')

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
    model = smp.Unet(encoder_name='densenet161', encoder_weights='imagenet', in_channels=191, classes=80, activation='sigmoid').to(device)
    model.load_state_dict(checkpoint['state_dict'])

    # Setup Loss Function and Optimizer
    if config['trainLossFunction'] == "mse":
        lossFunction = nn.MSELoss(reduction='sum')
    elif config['trainLossFunction'] == "maskedL1":
        lossFunction = BaseLoss({}, device=device)
    else:
        raise NotImplementedError

    # Set Preprocessing for Earthnet data
    preprocessingStage = None

    # Create dataset for testing part of Earthnet dataset
    # testDataset = EarthnetTestDataset(dataDir=os.path.join(config['testDataDir'], args.testSplit), dtype=config['dataDtype'], transform=preprocessingStage)
    # testDataset = Subset(testDataset, range(450))
    testDataset = EarthNet2021Dataset(folder=os.path.join(config['testDataDir'], args.testSplit, 'context'), noisy_masked_pixels = False, use_meso_static_as_dynamic = False, fp16 = False)
    # testDataset = Subset(testDataset, range(3))

    # Create testing Dataloader
    testDataloader = DataLoader(testDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)  

    testLoss = testing_loop(testDataloader, model, lossFunction, predsFolder, device)
    # logger.info("Mean Testing loss: {}".format(testLoss))

    logger.info("Testing Finished")


if __name__ == "__main__":
    main()