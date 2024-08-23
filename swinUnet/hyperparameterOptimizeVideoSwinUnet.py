from trainVideoSwinUnet import trainVideoSwinUnet, parseArgs, load_config
import argparse
import yaml
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler



def main():

    # Parse the input arguments and load the configuration file
    args   = parseArgs()
    config = load_config(args)

    # Modify config to include search spaces
    config['lr'] = tune.loguniform(0.0001, 0.01)
    config['weightDecay'] = tune.quniform(0.005, 0.05, 0.005)
    config['warmupEpochs'] = tune.choice([5, 10, 15])
    config['lrScaleFactor'] = tune.choice([0.1, 0.2, 0.5])
    config['schedulerPatience'] = tune.choice([5, 7, 10])
    config['windowSize'] = tune.choice([[1, 7, 7], [3, 7, 7], [8, 7, 7], [16, 7, 7], [1, 14, 14], [3, 14, 14], [8, 14, 14], [16, 14, 14], [1, 3, 3], [3, 3, 3], [8, 3, 3], [16, 3, 3]])
    config['patchSize'] = tune.choice([[1, 4, 4], [2, 4, 4], [4, 4, 4]])
    config['patchEmbedding'] ['norm'] = tune.choice([True, False])
    config['encoder']['layerDepths'] = tune.choice([[2, 2, 2], [2, 2]])
    config['encoder']['downsample'] = tune.sample_from(lambda spec: [False, True, True] if spec.config.encoder.layerDepths == [2, 2, 2] else [False, True])
    config['encoder']['numChannels'] = tune.sample_from(lambda spec: [96, 192, 384] if spec.config.encoder.layerDepths == [2, 2, 2] else [96, 192])
    config['encoder']['norm'] = tune.choice([True, False])
    config['encoder']['dropPath'] = tune.choice([0.1, 0.2, 0.3, 0.4])
    config['decoder']['layerDepths'] = tune.sample_from(lambda spec: [2, 2, 2] if spec.config.encoder.layerDepths == [2, 2, 2] else [2, 2])
    config['decoder']['upsample'] = tune.sample_from(lambda spec: [True, True, False] if spec.config.encoder.layerDepths == [2, 2, 2] else [True, False])
    config['decoder']['numChannels'] = tune.sample_from(lambda spec: [384, 384, 288] if spec.config.encoder.layerDepths == [2, 2, 2] else [192, 192])
    config['decoder']['norm'] = tune.choice([True, False])
    config['decoder']['dropPath'] = tune.choice([0.1, 0.2, 0.3, 0.4])
    config['l1Weight'] = tune.quniform(0.1, 10, 0.1)
    config['mseWeight'] = tune.quniform(0.1, 10, 0.1)
    config['ssimWeight'] = tune.quniform(0.1, 10, 0.1)

    scheduler = ASHAScheduler(time_attr='training_iteration',
                              metric='valL1Loss',
                              mode='min',
                              stop_last_trials=False)
    
    tuner = tune.Tuner(
        tune.with_resources(tune.with_parameters(trainVideoSwinUnet, args=args), 
                            resources={"cpu": 16, "gpu": 1}
                            ),
        tune_config=tune.TuneConfig(scheduler=scheduler,
                                    num_samples=-1,
                                    time_budget_s=22*60*60,  #22*60*60 # hours * minutes * seconds
                                    max_concurrent_trials=2
                                    ),
        # run_config=train.RunConfig(log_to_file=True),
        param_space=config,
        )
    
    results = tuner.fit()
    
    print(results)

if __name__ == '__main__':
    main()