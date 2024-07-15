import earthnet as en
import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt

variables = []

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictionsFolder', help='The path to the folder with the predictions .npz files (path up until the tile folders).')
    parser.add_argument('--groundTruthFolder', help='The path to the folder the correspondingground truth .npz files (path up until the tile folders).')
    parser.add_argument('--variablesToPlot', nargs='+', help='The variable to plot')
    return parser.parse_args()

def main():
    # Parse the input arguments
    args = parseArgs()

    predictionsFolderPath = Path(args.predictionsFolder)

    cubesRelativePathList = sorted([str(path.relative_to(predictionsFolderPath)) for path in predictionsFolderPath.glob("**/*.npz")])
    
    imagesFolderPath = os.path.join(str(predictionsFolderPath.parent), 'images', args.predictionsFolder.split('/')[-1])
    os.makedirs(imagesFolderPath, exist_ok=True)

    for cubeRelativePath in cubesRelativePathList:

        for var in args.variablesToPlot:

            fig1 = en.cube_gallery(os.path.join(str(predictionsFolderPath), cubeRelativePath), variable=var, save_path=os.path.join(imagesFolderPath, cubeRelativePath.removesuffix('.npz') + var))
            # fig2 = en.cube_gallery(os.path.join(args.groundTruthFolder, cubeRelativePath), variable=var, save_path=os.path.join(imagesFolderPath, cubeRelativePath.removesuffix('.npz') + var))
            
            # ax1 = fig1.gca()

            # # Access the axes from the second figure
            # ax2 = fig2.gca()

            # # Create a new figure to combine the plots
            # combined_fig, (combined_ax1, combined_ax2) = plt.subplots(2, 1, figsize=(8, 10))

            # # Plot the data from the first figure on the first subplot
            # combined_ax1.plot(x1, y1, label='Sine')
            # combined_ax1.set_title('Sine Wave')
            # combined_ax1.legend()

            # # Plot the data from the second figure on the second subplot
            # combined_ax2.plot(x2, y2, label='Cosine')
            # combined_ax2.set_title('Cosine Wave')
            # combined_ax2.legend()

            # # Adjust layout for better appearance
            # plt.tight_layout()

            # # Save the combined figure
            # combined_fig.savefig('combined_figure.png')
            plt.close(fig1)



if __name__ == "__main__":
    main()