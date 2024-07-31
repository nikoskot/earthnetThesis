import pandas as pd
import os
import argparse
import yaml
import torch
from collections.abc import MutableMapping

def flatten_dict(d: MutableMapping, sep: str= '_') -> MutableMapping:
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experimentsFolder', default='/home/nikoskot/earthnetThesis/experiments', help='The path to the experiments folder.')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':

    args = parseArgs()

    # Get the list of subfolders with full paths
    subfolders = [os.path.join(args.experimentsFolder, name) for name in os.listdir(args.experimentsFolder) if os.path.isdir(os.path.join(args.experimentsFolder, name))]

    data = []

    for folder in subfolders:

        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        if 'checkpoint.pth' not in files:
            continue
        if 'config.yml' not in files:
            continue

        # Read config file
        config = load_config(os.path.join(folder, 'config.yml'))

        # Read best loss and epoch
        checkpoint = torch.load(os.path.join(folder, 'checkpoint.pth'), map_location='cpu')
        if 'trainLossFunctions' in list(config.keys()):
            for l in config['trainLossFunctions']:
                if ('train_' + l) in  list(checkpoint.keys()):
                    config['train_' + l] = float(checkpoint['train_' + l])
        elif 'trainLossFunction' in list(config.keys()):
            for l in config['trainLossFunction']:
                if ('train_' + l) in  list(checkpoint.keys()):
                    config['train_' + l] = float(checkpoint['train_' + l])
        else:
            print("No training losses")
        
        # config['valLoss'] = checkpoint['valLoss'].item() if torch.is_tensor(checkpoint['valLoss']) else checkpoint['valLoss']
        config['valLoss'] = checkpoint['valLoss']
        config['epoch'] = checkpoint['epoch']
        config['folder'] = folder.split('/')[-1]

        data.append(flatten_dict(config))


    df = pd.DataFrame(data)
    print(data)
        
    df.to_csv(os.path.join(args.experimentsFolder, 'configurations.csv'), index=False)

