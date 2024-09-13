from typing import List
from ray.tune.experiment import Trial
from trainVideoSwinUnetV2TestLossCombination import trainVideoSwinUnet, parseArgs, load_config
import argparse
import yaml
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
import shutil
import os
import pandas as pd

# class AddRunFolderPathToConfiguration(tune.Callback):
#     """
#     This callback will delete a trial experiment folder if it was not the best performing trial.
#     """
#     def __init__(self):
#         pass

#     def on_trial_start(self, iteration, trials, trial, **info):
#         trial.config['']
def emptyFolder(folder):
        if not os.path.exists(folder):
            return
        
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

        os.rmdir()

class DeleteWorseTrialExperimentFolder(tune.Callback):
    """
    This callback will delete a trial experiment folder if it was not the best performing trial.
    """
    def __init__(self, metric: str):
        self.metric = metric
        self.metricOfBestTrialSoFar = None
        self.nameOfBestTrialSoFar = None
        self.foldersToEmpty = []
        self.cleanFrequency = 2
        self.bestMetricsOfTrials = {}

    def on_trial_result(self, iteration, trials, trial, result, **info):

        trial.config['runFolder'] = result['config']['runFolder']

        # metricOfCurrentTrial = result[self.metric]
        self.bestMetricsOfTrials[trial] = result[self.metric]

        # if len(trials) % self.cleanFrequency == 0:
        #     for fd in self.foldersToEmpty:
        #         emptyFolder(fd)

        # if (self.metricOfBestTrialSoFar and metricOfCurrentTrial < self.metricOfBestTrialSoFar):
        #     self.foldersToEmpty.append(os.path.join(trial.config['experimentsFolder'], self.nameOfBestTrialSoFar))

            # emptyFolder(os.path.join(trial.config['experimentsFolder'], self.nameOfBestTrialSoFar))

            # if os.path.exists(os.path.join(trial.config['experimentsFolder'], self.nameOfBestTrialSoFar)):
            #     shutil.rmtree(os.path.join(trial.config['experimentsFolder'], self.nameOfBestTrialSoFar))

        #     self.metricOfBestTrialSoFar = metricOfCurrentTrial
        #     self.nameOfBestTrialSoFar = trial.config['runFolder']

        # elif (self.metricOfBestTrialSoFar):
        #     self.foldersToEmpty.append(os.path.join(trial.config['experimentsFolder'], trial.config['runFolder']))

            # emptyFolder(os.path.join(trial.config['experimentsFolder'], trial.config['runFolder']))

            # if os.path.exists(os.path.join(trial.config['experimentsFolder'], trial.config['runFolder'])):
            #     shutil.rmtree(os.path.join(trial.config['experimentsFolder'], trial.config['runFolder']))

        # else:
        #     self.metricOfBestTrialSoFar = metricOfCurrentTrial
        #     self.nameOfBestTrialSoFar = trial.config['runFolder']
        
    def on_trial_complete(self, iteration: int, trials: List[Trial], trial: Trial, **info):

        completedTrials = [t for t in trials if ((t.status == 'TERMINATED') and t in list(self.bestMetricsOfTrials.keys()))]
        # erroredTrials = [t for t in trials if t.status == 'ERROR']

        if len(completedTrials) >= self.cleanFrequency:

            sortedCompletedTrialsByMetric = sorted(completedTrials, key=lambda x : self.bestMetricsOfTrials[x])

            for t in sortedCompletedTrialsByMetric[1:]:
                shutil.rmtree(os.path.join(t.config['experimentsFolder'], t.config['runFolder']))
                self.bestMetricsOfTrials.pop(t)
                print("Removed folder {} of trial {}".format(t.config['runFolder'], t))

        # if len(erroredTrials) >= self.cleanFrequency:

        #     print("Errored trials")
        #     print(erroredTrials)

        #     for t in erroredTrials:
        #         print("Config of errored Errored trial")
        #         print(t.config)
        #         shutil.rmtree(os.path.join(t.config['experimentsFolder'], t.config['runFolder']))
        #         # self.bestMetricsOfTrials.pop(t)
        #         print("Removed folder {} of trial {}".format(t.config['runFolder'], t))
            
    

def main():

    # Parse the input arguments and load the configuration file
    args   = parseArgs()
    config = load_config(args)

    # Modify config to include search spaces
    # config['lr'] = tune.loguniform(0.0001, 0.01)
    # config['weightDecay'] = tune.quniform(0.005, 0.05, 0.005)
    # config['warmupEpochs'] = tune.choice([5, 10, 15])
    # config['lrScaleFactor'] = tune.choice([0.1, 0.2, 0.5])
    # config['schedulerPatience'] = tune.choice([5, 7, 10])
    # config['cosineAnnealingRestartsInterval'] = tune.choice([5, 10, 15])
    # config['C'] = tune.choice([96, 144, 192])
    # config['windowSize'] = tune.choice([[1, 7, 7], [3, 7, 7], [8, 7, 7], [1, 14, 14], [3, 14, 14], [8, 14, 14], [1, 3, 3], [3, 3, 3], [8, 3, 3]])
    # config['patchSize'] = tune.grid_search([[1, 4, 4], [2, 4, 4], [5, 4, 4], [1, 2, 2], [2, 2, 2], [5, 2, 2]])
    # config['patchEmbedding'] ['norm'] = tune.choice([True, False])
    # config['encoder']['downsample'] = tune.sample_from(lambda spec: [False, True, True] if spec.config.encoder.layerDepths == [2, 2, 2] else [False, True])
    # config['encoder']['dropPath'] = tune.choice([0.1, 0.2, 0.3, 0.4])
    # config['encoder']['norm'] = tune.choice([True, False])
    # config['decoder']['upsample'] = tune.sample_from(lambda spec: [True, True, False] if spec.config.encoder.layerDepths == [2, 2, 2] else [True, False])
    # config['decoder']['norm'] = tune.choice([True, False])
    # config['decoder']['dropPath'] = tune.choice([0.1, 0.2, 0.3, 0.4])
    # config['encoder']['layerDepths'] = tune.grid_search([[2, 2, 2], [2, 2]])
    # config['encoder']['numChannels'] = tune.sample_from(lambda spec: [spec.config.C, spec.config.C*2, spec.config.C*4] if spec.config.encoder.layerDepths == [2, 2, 2] else [spec.config.C, spec.config.C*2])
    # config['decoder']['layerDepths'] = tune.sample_from(lambda spec: [2, 2, 2] if spec.config.encoder.layerDepths == [2, 2, 2] else [2, 2])

    # V1
    # config['decoder']['numChannels'] = tune.sample_from(lambda spec: [spec.config.C*4, spec.config.C*4, spec.config.C*3] if spec.config.encoder.layerDepths == [2, 2, 2] else [spec.config.C*2, spec.config.C*2])
    
    # V2
    # config['decoder']['numChannels'] = tune.sample_from(lambda spec: [480, 528, 456] if spec.config.encoder.layerDepths == [2, 2, 2] else [288, 336])

    # config['l1Weight'] = tune.quniform(1, 10, 0.2)
    # config['mseWeight'] = tune.quniform(1, 10, 0.2)
    # config['ssimWeight'] = tune.quniform(1, 10, 0.2)
    # config['vggWeight'] = tune.quniform(1, 10, 0.2)
    config['useMSE'] = tune.grid_search([True, False])
    config['useSSIM'] = tune.grid_search([True, False])
    config['useVGG'] = tune.grid_search([True, False])
    # config['scheduler'] = tune.choice(['ReduceLROnPlateau', 'CosineAnnealing'])
    # config['ape'] = tune.choice([True, False])

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
                                    time_budget_s=8*50*60,  #22*60*60 # hours * minutes * seconds
                                    max_concurrent_trials=2
                                    ),
        run_config=train.RunConfig(callbacks=[DeleteWorseTrialExperimentFolder(metric='bestValLoss')]),
        param_space=config,
        )
    
    results = tuner.fit()
    
    # Save results dataframe to run folder
    results.get_dataframe().to_csv(os.path.join(results.experiment_path, 'results.csv'))

if __name__ == '__main__':
    main()