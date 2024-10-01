import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import numpy as np
import random
from earthnetDataloader import EarthnetTestDataset, EarthnetTrainDataset, Preprocessing, PreprocessingStack, PreprocessingSeparate, PreprocessingV2, PreprocessingWeather
from torch.utils.data import DataLoader, random_split, Subset
from earthnet_scores import EarthNet2021ScoreUpdateWithoutCompute
from losses import MaskedLoss, maskedSSIMLoss, MaskedVGGLoss
import tqdm
import datetime
import argparse
import yaml
import os
import logging
import shutil
from torch.utils.tensorboard import SummaryWriter
from piqa import SSIM
import copy
import rerun as rr
import rerun.blueprint as rrb
from earthnet.plot_cube import gallery, colorize
import yaml
from ray import train, tune
from ray.tune import Trainable


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/nikoskot/earthnetThesis/experiments/config.yml', help='The path to the configuration file.')
    parser.add_argument('--resumeTraining', default=False, action='store_true', help='If we want to resume the training from a checkpoint.')
    parser.add_argument('--resumeCheckpoint', help='The path of the checkpoint to resume training from.')
    parser.add_argument('--optimizeHyperparams', default=False, action='store_true', help='If this is a hyperparameter optimization run.')
    parser.add_argument('--note', help='Note to write at beginning of log file.')
    parser.add_argument('--seed', default=88, type=int)
    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Training loop
# def train_loop(dataloader, 
#                model, 
#                l1Loss, 
#                ssimLoss, 
#                mseLoss,
#                vggLoss,
#                optimizer, 
#                config, 
#                logger, 
#                trainVisualizationCubenames,
#                epoch):

#     # Set the model to training mode - important for batch normalization and dropout layers
#     model.train()
#     l1LossSum = 0
#     mseLossSum = 0
#     SSIMLossSum = 0
#     vggLossSum = 0

#     batchNumber = 0

#     maxGradPre = torch.tensor(0)
#     minGradPre = torch.tensor(0)

#     for data in tqdm.tqdm(dataloader):
#         batchNumber += 1

#         optimizer.zero_grad()

#         # Move data to GPU
#         x = data['x'].to(torch.device('cuda'))
#         y = data['y'].to(torch.device('cuda'))
#         masks = data['targetMask'].to(torch.device('cuda'))

#         # Compute prediction and loss
#         pred = model(x)
#         # pred = torch.clamp(pred, min=0.0, max=1.0)

#         l1LossValue = l1Loss(pred, y, masks)
#         if config['useMSE']:
#             mseLossValue = mseLoss(pred, y, masks)
#         if config['useSSIM']:
#             ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, masks)
#         if config['useVGG']:
#             vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), masks)

#         # Backpropagation
#         totalLoss = config['l1Weight'] * l1LossValue
#         if config['useMSE']:
#             totalLoss = totalLoss + (config['mseWeight'] * mseLossValue)
#         if config['useSSIM']:
#             totalLoss = totalLoss + (config['ssimWeight'] * ssimLossValue)
#         if config['useVGG']:
#             totalLoss = totalLoss + (config['vggWeight'] * vggLossValue)

#         totalLoss.backward()

#         for param in model.parameters():
#             maxGradPre = torch.maximum(maxGradPre, torch.max(param.grad.view(-1)))
#             minGradPre = torch.minimum(minGradPre, torch.min(param.grad.view(-1)))

#         if config['gradientClipping']:
#             nn.utils.clip_grad_value_(model.parameters(), config['gradientClipValue'])

#         optimizer.step()

#         # Add the loss to the total loss 
#         l1LossSum += l1LossValue.item()
#         if config['useSSIM']:
#             SSIMLossSum += ssimLossValue.item()
#         if config['useMSE']:
#             mseLossSum += mseLossValue.item()
#         if config['useVGG']:
#             vggLossSum += vggLossValue.item()

#         # Log visualizations if available
#         if epoch % config['visualizationFreq'] == 0 or epoch == 1 or epoch == config['epochs']:
#             rr.set_time_sequence('epoch', epoch)
#             for i in range(len(data['cubename'])):
#                 if data['cubename'][i] in trainVisualizationCubenames:

#                     # # Log ground truth
#                     # targ = data['y'][i, ...].numpy()
#                     # targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
#                     # targ[targ < 0] = 0
#                     # targ[targ > 0.5] = 0.5
#                     # targ = 2 * targ
#                     # targ[np.isnan(targ)] = 0
#                     # grid = gallery(targ)
#                     # rr.log('/train/groundTruth/{}'.format(data['cubename'][i]), rr.Image(grid))

#                     # Log prediction
#                     targ = pred[i, ...].detach().cpu().numpy()
#                     targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
#                     targ[targ < 0] = 0
#                     targ[targ > 0.5] = 0.5
#                     targ = 2 * targ
#                     targ[np.isnan(targ)] = 0
#                     grid = gallery(targ)
#                     rr.log('/train/prediction/{}'.format(data['cubename'][i]), rr.Image(grid))
    
#     logger.info("Maximum gradient before clipping: {}".format(maxGradPre))
#     logger.info("Minimum gradient before clipping: {}".format(minGradPre))

#     return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber
        

# def validation_loop(dataloader, 
#                     model, 
#                     l1Loss, 
#                     ssimLoss, 
#                     mseLoss,
#                     vggLoss,
#                     config, 
#                     validationVisualizationCubenames,
#                     epoch,
#                     ensCalculator,
#                     logger):

#     # Set the model to evaluation mode - important for batch normalization and dropout layers
#     model.eval()
#     l1LossSum = 0
#     SSIMLossSum = 0
#     mseLossSum = 0
#     vggLossSum = 0
#     earthnetScore = 0

#     batchNumber = 0

#     # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
#     # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#     with torch.no_grad():
#         for data in tqdm.tqdm(dataloader):
#             batchNumber += 1
            
#             # Move data to GPU
#             x = data['x'].to(torch.device('cuda'))
#             y = data['y'].to(torch.device('cuda'))
#             masks = data['targetMask'].to(torch.device('cuda'))

#             # Compute prediction and loss
#             pred = model(x)
#             # pred = torch.clamp(pred, min=0.0, max=1.0)

#             l1LossValue = l1Loss(pred, y, masks)
#             if config['useMSE']:
#                 mseLossValue = mseLoss(pred, y, masks)
#             if config['useSSIM']:
#                 ssimLossValue = ssimLoss(torch.clamp(pred, min=0, max=1), y, masks)
#             if config['useVGG']:
#                 vggLossValue = vggLoss(torch.clamp(pred.clone(), min=0, max=1), y.clone(), masks)

#             # Add the loss to the total loss
#             l1LossSum += l1LossValue.item()
#             if config['useSSIM']:
#                 SSIMLossSum += ssimLossValue.item()
#             if config['useMSE']:
#                 mseLossSum += mseLossValue.item()
#             if config['useVGG']:
#                 vggLossSum += vggLossValue.item()

#             # Log visualizations if available
#             if epoch % config['visualizationFreq'] == 0 or epoch == 1 or epoch == config['epochs']:
#                 rr.set_time_sequence('epoch', epoch)
#                 for i in range(len(data['cubename'])):
#                     if data['cubename'][i] in validationVisualizationCubenames:

#                         # # Log ground truth
#                         # targ = data['y'][i, ...].numpy()
#                         # targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
#                         # targ[targ < 0] = 0
#                         # targ[targ > 0.5] = 0.5
#                         # targ = 2 * targ
#                         # targ[np.isnan(targ)] = 0
#                         # grid = gallery(targ)
#                         # rr.log('/train/groundTruth/{}'.format(data['cubename'][i]), rr.Image(grid))

#                         # Log prediction
#                         targ = pred[i, ...].detach().cpu().numpy()
#                         targ = np.stack([targ[2, :, :, :], targ[1, :, :, :], targ[0, :, :, :]], axis=0).transpose(1, 2 ,3 ,0)
#                         targ[targ < 0] = 0
#                         targ[targ > 0.5] = 0.5
#                         targ = 2 * targ
#                         targ[np.isnan(targ)] = 0
#                         grid = gallery(targ)
#                         rr.log('/validation/prediction/{}'.format(data['cubename'][i]), rr.Image(grid))
            
#             if epoch % config['calculateENSonValidationFreq'] == 0 or epoch == config['epochs']:
#                 ensCalculator(pred.permute(0, 2, 3, 4, 1), y.permute(0, 2, 3, 4, 1), 1-masks.permute(0, 2, 3, 4, 1))

#         if epoch % config['calculateENSonValidationFreq'] == 0 or epoch == config['epochs']:
#             logger.info("Validation split Earthnet Score on epoch {}".format(epoch))
#             ens = ensCalculator.compute()
#             logger.info(ens)
#             earthnetScore = ens['EarthNetScore']
#             ensCalculator.reset()

#     return l1LossSum / batchNumber, SSIMLossSum / batchNumber, mseLossSum / batchNumber, vggLossSum / batchNumber, earthnetScore

def trainWeatherDataAutoEncoder(config, args):

    # Get the date and time when the execution started
    runDateTime = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')

    # Set device
    device = torch.device('cuda')
    print(device)

    # Parse the input arguments and load the configuration file
    # args   = parseArgs()
    # config = load_config(args)

    # Setup the output folder structure
    if args.resumeTraining:
        outputFolder = os.path.dirname(args.resumeCheckpoint) if args.resumeCheckpoint.endswith('.pth') else args.resumeCheckpoint # .../experiments/modelType_trainDatetime/
        resumePostfix = '_' + runDateTime
        if not args.resumeCheckpoint.endswith('.pth'):
            checkpointsList = [f for f in os.listdir(outputFolder) if (os.path.isfile(os.path.join(outputFolder, f)) and f.startswith(('checkpoint.', 'checkpoint_')))]
            print(checkpointsList)
            if (len(checkpointsList) == 1 and 'checkpoint.pth' in checkpointsList):
                lastCheckpointName = 'checkpoint.pth'
            else:
                checkpointsList.remove('checkpoint.pth')
                print(checkpointsList)
                checkpointsList.sort(key=lambda name: datetime.datetime.strptime(name.removeprefix('checkpoint_').removesuffix('.pth'), "%d-%m-%Y_%H-%M-%S"))
                lastCheckpointName = checkpointsList[-1]
            print(lastCheckpointName)
    else:
        outputFolder = os.path.join(config['experimentsFolder'], config['modelType'] + '_' + runDateTime) # .../experiments/modelType_trainDatetime/
        os.makedirs(outputFolder, exist_ok=True)
        resumePostfix = ''
    config['runFolder'] = config['modelType'] + '_' + runDateTime

    # Make a copy of the configuration file on the output folder
    shutil.copy(args.config, outputFolder)

    with open(os.path.join(outputFolder, 'savedConfig.yml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=None, sort_keys=False)

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
    from WeatherDataAutoEncoder import WeatherDataAutoEncoder, train_loop, validation_loop
    model = WeatherDataAutoEncoder(config, logger).to(device)

    # Setup Loss Function and Optimizer
    l1Loss = MaskedLoss(lossType='l1', lossFunction=nn.L1Loss(reduction='sum'))
    mseLoss = MaskedLoss(lossType='mse', lossFunction=nn.MSELoss(reduction='sum'))
    ssimLoss = maskedSSIMLoss
    vggLoss = MaskedVGGLoss()

    if config['trainingOptimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
    elif config['trainingOptimizer'] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weightDecay'])
    else:
        raise NotImplementedError
    
    # Define the lambda function for linear warmup followed by constant LR
    def lr_lambda(current_step):
        if current_step < config['warmupEpochs']:
            return float(current_step) / float(max(1, config['warmupEpochs']))  # Linear warmup
        return 1.0  # Constant learning rate after warmup

    if config['scheduler'] == 'ReduceLROnPlateau':
        warmupScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=config['lrScaleFactor'], patience=config['schedulerPatience'], min_lr=0.00001)
    elif config['scheduler'] == 'CosineAnnealing':
        warmupScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=config['epochs'], eta_min=0.000001)
    elif config['scheduler'] == 'CosineAnnealingWarmRestarts':
        warmupScheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=config['cosineAnnealingRestartsInterval'], eta_min=0.000001)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
        logger.info('Scheduler: Warmup + constant LR')
    
    # Load model in case we want to resume training
    if args.resumeTraining:
        checkpoint = torch.load(args.resumeCheckpoint) if args.resumeCheckpoint.endswith('.pth') else torch.load(os.path.join(args.resumeCheckpoint, lastCheckpointName))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = checkpoint['lr']
        if config['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint['warmupScheduler'] is not None:
            warmupScheduler.load_state_dict(checkpoint['warmupScheduler'])
        startEpoch = checkpoint['epoch'] + 1
        bestValLoss = checkpoint['bestValLoss']
        bestEns = checkpoint['ens']
        logger.info("Resuming training from checkpoint {} from epoch {}".format(args.resumeCheckpoint, startEpoch))
    else:
        startEpoch = 1
        bestValLoss = np.inf
        bestEns = 0
        logger.info("Starting training from from epoch {}".format(startEpoch))

    # Set Preprocessing for Earthnet data
    preprocessingStage = PreprocessingWeather(reduceTime=not config['autoencoderReduceTime'])

    # Create dataset of training part of Earthnet dataset
    trainDataset = EarthnetTrainDataset(dataDir=config['trainDataDir'], dtype=config['dataDtype'], transform=preprocessingStage, cropMesodynamic=config['cropMesodynamic'])
    if isinstance(config['trainDataSubset'], int):
        trainDataset = Subset(trainDataset, np.random.randint(0, len(trainDataset), size=config['trainDataSubset']))

    # Create training and validation Dataloaders
    if config['overfitTraining']:
        # If we want to overfit the model to a subset of the training data
        trainDataloader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)
        valDataset      = copy.deepcopy(trainDataset)
        valDataloader   = DataLoader(valDataset, batch_size=config['batchSize'], shuffle=False, num_workers=config['numWorkers'], pin_memory=True)
    else:
        # Normal training
        trainDataset, valDataset = random_split(trainDataset, [config['trainSplit'], config['validationSplit']])
        # Split to training and validation dataset
        trainDataloader = DataLoader(trainDataset, batch_size=config['batchSize'], shuffle=True, num_workers=config['numWorkers'], pin_memory=True)
        valDataloader   = DataLoader(valDataset,   batch_size=config['batchSize'], shuffle=True, num_workers=config['numWorkers'], pin_memory=True)

    # Setup visualization
    # Choose 10 random cubes from the training and validation datasets ato visualize through the training
    rr.init(config['modelType'] + '_' + runDateTime, spawn=False)
    rr.save(path=os.path.join(outputFolder, 'trainingVisualizations.rrd'))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="Training prediction", origin="/train/prediction/"),
                rrb.Spatial2DView(name="Training groundTruth", origin="/train/groundTruth/"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="Validation prediction", origin="/validation/prediction/"),
                rrb.Spatial2DView(name="Validation groundTruth", origin="/validation/groundTruth/"),
            ),
        )
    )
    rr.send_blueprint(blueprint)
    rr.set_time_sequence('epoch', startEpoch)
    cubeIds = np.random.choice(range(len(trainDataset)), size=min(5, len(trainDataset)), replace=False)
    trainVisualizationCubenames = [trainDataset.__getitem__(id)['cubename'] for id in cubeIds]

    # Log training ground truths
    for id in cubeIds:
        # variable rr
        targ = trainDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[0, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="Blues", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/train/groundTruth/precipitaion/{}'.format(trainDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable pp
        targ = trainDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[1, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="rainbow", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/train/groundTruth/pressure/{}'.format(trainDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable tg
        targ = trainDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[2, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/train/groundTruth/meanTmp/{}'.format(trainDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable tn
        targ = trainDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[3, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/train/groundTruth/minimumTmp/{}'.format(trainDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable tx
        targ = trainDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[4, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/train/groundTruth/maximumTmp/{}'.format(trainDataset.__getitem__(id)['cubename']), rr.Image(grid))
        

    cubeIds = np.random.choice(range(len(valDataset)), size=min(5, len(valDataset)), replace=False)
    validationVisualizationCubenames = [valDataset.__getitem__(id)['cubename'] for id in cubeIds]

    for id in cubeIds:
        # variable rr
        targ = valDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[0, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="Blues", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/validation/groundTruth/precipitaion/{}'.format(valDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable pp
        targ = valDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[1, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="rainbow", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/validation/groundTruth/pressure/{}'.format(valDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable tg
        targ = valDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[2, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/validation/groundTruth/meanTmp/{}'.format(valDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable tn
        targ = valDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[3, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/validation/groundTruth/minimumTmp/{}'.format(valDataset.__getitem__(id)['cubename']), rr.Image(grid))
        # variable tx
        targ = valDataset.__getitem__(id)[config['weatherAEImagesPart'] + 'Weather'].numpy()[4, :, :, :]#.transpose(1, 2 ,3 ,0)
        targ = colorize(targ, colormap="coolwarm", mask_red=np.isnan(targ))
        grid = gallery(targ)
        rr.log('/validation/groundTruth/maximumTmp/{}'.format(valDataset.__getitem__(id)['cubename']), rr.Image(grid))

    # # Initialize Earthnet Score calculator based on Earthformer implementation
    # ensCalculator = EarthNet2021ScoreUpdateWithoutCompute(layout='NTHWC', eps=1E-4)

    for i in range(startEpoch, config['epochs'] + 1):

        logger.info("Epoch {}\n-------------------------------".format(i))
        for j, param_group in enumerate(optimizer.param_groups):
            logger.info("Learning Rate of group {}: {}".format(j, param_group['lr']))

        trainL1Loss, trainSSIMLoss, trainMSELoss, trainVGGLoss = train_loop(trainDataloader, 
                                                            model, 
                                                            l1Loss, 
                                                            ssimLoss, 
                                                            mseLoss,
                                                            vggLoss, 
                                                            optimizer, 
                                                            config, 
                                                            logger, 
                                                            trainVisualizationCubenames, 
                                                            epoch=i)
        
        logger.info("Mean training L1 loss: {}".format(trainL1Loss))
        logger.info("Mean training SSIM loss: {}".format(trainSSIMLoss))
        logger.info("Mean training MSE loss: {}".format(trainMSELoss))
        logger.info("Mean training VGG loss: {}".format(trainVGGLoss))

        valL1Loss, valSSIMLoss, valMSELoss, valVGGLoss, ens = validation_loop(valDataloader, 
                                                           model, 
                                                           l1Loss, 
                                                           ssimLoss, 
                                                           mseLoss,
                                                           vggLoss, 
                                                           config, 
                                                           validationVisualizationCubenames, 
                                                           epoch=i,
                                                           ensCalculator=None,
                                                           logger=logger)
        
        logger.info("Mean validation L1 loss: {}".format(valL1Loss))
        logger.info("Mean validation SSIM loss: {}".format(valSSIMLoss))
        logger.info("Mean validation MSE loss: {}".format(valMSELoss))
        logger.info("Mean validation VGG loss: {}".format(valVGGLoss))

        # Save checkpoint based on validation loss
        if valL1Loss < bestValLoss:

            bestValLoss = valL1Loss
            checkpoint  = {'state_dict' : model.state_dict(),
                           'modelType'  : config['modelType'],
                           'epoch'      : i,
                           'valL1Loss'  : valL1Loss,
                           'valSSIMLoss': valSSIMLoss,
                           'valMSELoss' : valMSELoss,
                           'valVGGLoss' : valVGGLoss,
                           'bestValLoss': bestValLoss,
                           'optimizer'  : optimizer.state_dict(),
                           'scheduler'  : scheduler.state_dict() if config['scheduler'] is not None else None,
                           'warmupScheduler': warmupScheduler.state_dict() if warmupScheduler is not None else None,
                           'lr'         : scheduler.get_last_lr()[0],
                           'ens': bestEns,
                           }
            
            torch.save(checkpoint, os.path.join(outputFolder, 'checkpoint{}.pth'.format(resumePostfix)))
            logger.info("New best validation Loss {}, at epoch {}".format(bestValLoss, i))

        checkpoint  = {'state_dict' : model.state_dict(),
                       'modelType'  : config['modelType'],
                       'epoch'      : i,
                       'valL1Loss'  : valL1Loss,
                       'valSSIMLoss': valSSIMLoss,
                       'valMSELoss' : valMSELoss,
                       'valVGGLoss' : valVGGLoss,
                       'bestValLoss': bestValLoss,
                       'optimizer'  : optimizer.state_dict(),
                       'scheduler'  : scheduler.state_dict() if config['scheduler'] is not None else None,
                       'warmupScheduler': warmupScheduler.state_dict() if warmupScheduler is not None else None,
                       'lr'         : scheduler.get_last_lr()[0],
                       'ens': bestEns,
                        }
        torch.save(checkpoint, os.path.join(outputFolder, 'checkpointLast.pth'))

        if config['scheduler'] == 'ReduceLROnPlateau':
            if i <= config['warmupEpochs']:
                warmupScheduler.step()
            else:
                scheduler.step(valL1Loss)
        elif config['scheduler'] == 'CosineAnnealing' or config['scheduler'] == 'CosineAnnealingWarmRestarts':
            if i <= config['warmupEpochs']:
                warmupScheduler.step()
            else:
                scheduler.step()
        else:
            scheduler.step()

        if args.optimizeHyperparams:
            train.report({'valL1Loss': valL1Loss,
                          'valSSIMLoss': valSSIMLoss,
                          'valMSELoss' : valMSELoss,
                          'valVGGLoss' : valVGGLoss,
                          'trainL1Loss': trainL1Loss,
                          'trainSSIMLoss': trainSSIMLoss,
                          'trainMSELoss' : trainMSELoss,
                          'trainVGGLoss' : trainVGGLoss,
                          'bestValLoss' : bestValLoss,
                          })
        
        tbWriter.add_scalar('Loss/Train', trainL1Loss, i)
        tbWriter.add_scalar('Loss/Val', valL1Loss, i)
        tbWriter.add_scalar('SSIM/Train', trainSSIMLoss, i)
        tbWriter.add_scalar('SSIM/Val', valSSIMLoss, i)
        tbWriter.add_scalar('MSE/Train', trainMSELoss, i)
        tbWriter.add_scalar('MSE/Val', valMSELoss, i)
        tbWriter.add_scalar('VGG/Train', trainVGGLoss, i)
        tbWriter.add_scalar('VGG/Val', valVGGLoss, i)

    logger.info("Training Finished")
    tbWriter.flush()

if __name__ == "__main__":

    # Parse the input arguments and load the configuration file
    args   = parseArgs()
    config = load_config(args)

    trainWeatherDataAutoEncoder(config, args)