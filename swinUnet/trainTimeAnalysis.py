import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing
from torch.utils.data import DataLoader, random_split, Subset
import torchmetrics
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
import tqdm
import time
import datetime
from videoSwinUnetMMAction import VideoSwinUNet

# Set paremeters
C                    = 96 # C of Video Swin Transformer (number of channels after patch embedding)
BATCH_SIZE           = 16 # Bactch size
MODEL_INPUT_CHANNELS = 12 # Number of channels of input data
NUM_WORKERS          = 8  # Number of workers for Dataloaders
EPOCHS               = 4  # Number of epochs to run the training for
RUN_DATE_TIME        = datetime.datetime.now() # The date and time that the training started


# Intiialize Video Swin Unet model and move to GPU
model = VideoSwinUNet(inputChannels=MODEL_INPUT_CHANNELS, C=C).to(torch.device('cuda'))

# Setup Loss Function and Optimizer
lossFunction = nn.MSELoss(reduction='sum')
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

# Set Preprocessing for Earthnet data
preprocessingStage = Preprocessing()

# Create dataset of training part of Earthnet dataset
trainDataset = EarthnetTrainDataset(dataDir='/home/nikoskot/EarthnetDataset/train', dtype=np.float32, transform=preprocessingStage)
trainDataset = Subset(trainDataset, range(160))

# Split to training and validation dataset
trainDataset, valDataset = random_split(trainDataset, [0.8, 0.2])

# Create training and validation Dataloaders
trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
valDataloader   = DataLoader(valDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Create dataset for testing part of Earthnet dataset
testDataset = EarthnetTestDataset(dataDir='/home/nikoskot/EarthnetDataset/iid_test_split', dtype=np.float32, transform=preprocessingStage)
# testDataset = Subset(testDataset, range(320))

# Create testing Dataloader
testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Create objects for calculation of Earthnet Score during training, validation and testing
trainENSCalculator = EarthNet2021ScoreUpdateWithoutCompute(layout="NHWCT", eps=1E-4)
validENSCalculator = EarthNet2021ScoreUpdateWithoutCompute(layout="NHWCT", eps=1E-4)
testENSCalculator  = EarthNet2021ScoreUpdateWithoutCompute(layout="NHWCT", eps=1E-4)

# Training loop
def train_loop(dataloader, model, lossFunction, optimizer):

    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    lossSum = 0
    numberOfSamples = 0
    loadTimes = []
    moveTimes = []
    passTimes = []
    lossTimes = []
    backTimes = []
    optimTimes = []
    maskTimes = []
    ensTimes = []

    ts = time.time()
    for X, y in tqdm.tqdm(dataloader):

        loadTimes.append(time.time() - ts)
        
        ts = time.time()
        # Move data to GPU
        X = X.to(torch.device('cuda'))
        y = y.to(torch.device('cuda'))
        moveTimes.append(time.time() - ts)

        ts = time.time()
        # Compute prediction and loss
        pred = model(X)
        passTimes.append(time.time() - ts)

        ts = time.time()
        loss = lossFunction(pred, y)
        lossTimes.append(time.time() - ts)

        ts = time.time()
        # Backpropagation
        loss.backward()
        backTimes.append(time.time() - ts)

        ts = time.time()
        optimizer.step()
        optimizer.zero_grad()
        optimTimes.append(time.time() - ts)

        # Add the loss to the total loss of the batch and keep track of the number of samples
        lossSum         += loss.item()
        numberOfSamples += len(X)

        ts = time.time()
        # Isolate the mask from the ground truth and update the Earthnet Score Calculator
        mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
        maskTimes.append(time.time() - ts)

        ts = time.time()
        trainENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)
        ensTimes.append(time.time() - ts)

        ts = time.time()

    # Calculate Earthnet Score and reset the calculator
    trainENS = trainENSCalculator.compute()
    # trainENS = 0
    trainENSCalculator.reset()

    print("Data loading times: {}".format(loadTimes))
    print("Data moving times: {}".format(moveTimes))
    print("Pass times: {}".format(passTimes))
    print("Loss calc times: {}".format(lossTimes))
    print("Backprop times: {}".format(backTimes))
    print("Optimizer step times: {}".format(optimTimes))
    print("Mask times times: {}".format(maskTimes))
    print("ENS update times: {}".format(ensTimes))

    return lossSum / numberOfSamples, trainENS
        

def validation_loop(dataloader, model, lossFunction):

    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    lossSum = 0
    numberOfSamples = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader):
            
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
            mask = y[:, 4, :, :, :].unsqueeze(1).permute(0, 3, 4, 1, 2)
            validENSCalculator.update(pred.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], y.permute(0, 3, 4, 1, 2)[:, :, :, :4, :], mask=mask)

    # Calculate Earthnet Score and reset the calculator
    validationENS = validENSCalculator.compute()
    validENSCalculator.reset()

    return lossSum / numberOfSamples, validationENS

# Initilize current best validation earthnet score
bestValENS = {'EarthNetScore':-1}

for i in range(EPOCHS):

    print("Epoch {}\n-------------------------------".format(i))
    trainLoss, trainENS = train_loop(trainDataloader, model, lossFunction, optimizer)
    print("Training ENS metrics:")
    print(trainENS)

    valLoss, valENS = validation_loop(valDataloader, model, lossFunction)
    print("Validation ENS metrics:")
    print(valENS)

    # Early stopping based on validation EarthNet score
    if valENS['EarthNetScore'] > bestValENS['EarthNetScore']:
        bestValENS = valENS
        checkpoint = {'state_dict' : model.state_dict(),
                        'epoch'      : i,
                        'bestValENS' : bestValENS['EarthNetScore'],
                        'valMAD'     : bestValENS['MAD'],
                        'valOLS'     : bestValENS['OLS'],
                        'valEMD'     : bestValENS['EMD'],
                        'valSSIM'    : bestValENS['SSIM'],
                        'valLoss'    : valLoss,
                        'trainLoss'  : trainLoss,
                        'optimizer'  : optimizer.state_dict()}
        torch.save(checkpoint, 'checkpoint_{}.pth'.format(RUN_DATE_TIME))
        print("New best validation EarthNet score {}, at epoch {}".format(bestValENS['EarthNetScore'], i))

print("Training Finished")
