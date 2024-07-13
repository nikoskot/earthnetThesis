import earthnet as en
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictionsFolder', help='The path to the predictions folder.')
    parser.add_argument('--targetsFolder', help='The path to the tergets folder.')
    parser.add_argument('--dataOutputFilePath', help='The path to the data output file.')
    parser.add_argument('--scoreOutputFilePath', help='The path to the score output file.')
    return parser.parse_args()

def main():
    args = parseArgs()
    en.EarthNetScore.get_ENS(args.predictionsFolder, args.targetsFolder, data_output_file=args.dataOutputFilePath, ens_output_file=args.scoreOutputFilePath)

if __name__ == "__main__":
    main()