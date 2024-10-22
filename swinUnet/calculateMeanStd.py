# Calculate the mean and std of unpreproccessed data.
from earthnetDataloader import EarthnetTrainDataset, EarthnetTestDataset
from torch.utils.data import DataLoader
import numpy as np
import tqdm

BATCH_SIZE = 8
NUM_INPUT_CHANNELS = 11
NUM_OUTPUT_CHANNELS = 4
INPUT_TIME = 10
OUTPUT_TIME = 20
IMG_HW = 128

dataset = EarthnetTrainDataset(dataDir='/home/nikoskot/earthnetThesis/EarthnetDataset/train', dtype=np.float32, transform=None)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

ImgSum = 0
ImgSumSquares = 0
ImgCount = 0

WeatherSum = 0
WeatherSumSquares = 0
WeatherCount = 0

demHighSum = 0
demHighSumSquares = 0
demHighCount = 0

demMesoSum = 0
demMesoSumSquares = 0
demMesoCount = 0

for data in tqdm.tqdm(dataloader):

    contextImages = data['context']['images'].numpy()
    contextWeather = data['context']['weather'].numpy()
    targetImages = data['target']['images'].numpy()
    targetWeather = data['target']['weather'].numpy()
    demHigh = data['demHigh'].numpy()
    demMeso = data['demMeso'].numpy()

    ImgSum += np.sum(contextImages, axis=(0, 2, 3, 4))
    ImgSumSquares += np.sum(contextImages ** 2, axis=(0, 2, 3, 4))
    ImgCount += contextImages.shape[0] * contextImages.shape[2] * contextImages.shape[3] * contextImages.shape[4]

    WeatherSum += np.sum(contextWeather, axis=(0, 2, 3, 4))
    WeatherSumSquares += np.sum(contextWeather ** 2, axis=(0, 2, 3, 4))
    WeatherCount += contextWeather.shape[0] * contextWeather.shape[2] * contextWeather.shape[3] * contextWeather.shape[4]

    ImgSum += np.sum(targetImages, axis=(0, 2, 3, 4))
    ImgSumSquares += np.sum(targetImages ** 2, axis=(0, 2, 3, 4))
    ImgCount += targetImages.shape[0] * targetImages.shape[2] * targetImages.shape[3] * targetImages.shape[4]

    WeatherSum += np.sum(targetWeather, axis=(0, 2, 3, 4))
    WeatherSumSquares += np.sum(targetWeather ** 2, axis=(0, 2, 3, 4))
    WeatherCount += targetWeather.shape[0] * targetWeather.shape[2] * targetWeather.shape[3] * targetWeather.shape[4]

    demHighSum += np.sum(demHigh, axis=(0, 2, 3))
    demHighSumSquares += np.sum(demHigh ** 2, axis=(0, 2, 3))
    demHighCount += demHigh.shape[0] * demHigh.shape[2] * demHigh.shape[3]

    demMesoSum += np.sum(demMeso, axis=(0, 2, 3))
    demMesoSumSquares += np.sum(demMeso ** 2, axis=(0, 2, 3))
    demMesoCount += demMeso.shape[0] * demMeso.shape[2] * demMeso.shape[3]

imagesMeanPerChannel = ImgSum / ImgCount
imagesStdPerChannel = np.sqrt((ImgSumSquares / ImgCount) - imagesMeanPerChannel ** 2)

weatherDataMeanPerChannel = WeatherSum / WeatherCount
weatherDataStdPerChannel = np.sqrt((WeatherSumSquares / WeatherCount) - weatherDataMeanPerChannel ** 2)

demHighMeanPerChannel = demHighSum / demHighCount
demHighStdPerChannel = np.sqrt((demHighSumSquares / demHighCount) - demHighMeanPerChannel ** 2)

demMesoMeanPerChannel = demMesoSum / demMesoCount
demMesoStdPerChannel = np.sqrt((demMesoSumSquares / demMesoCount) - demMesoMeanPerChannel ** 2)

print(f"Satellite images mean value per channel: {imagesMeanPerChannel}")
print(f"Satellite images std value per channel: {imagesStdPerChannel}")

print(f"Weather data images mean value per channel: {weatherDataMeanPerChannel}")
print(f"Weather data images std value per channel: {weatherDataStdPerChannel}")

print(f"DEM High images mean value per channel: {demHighMeanPerChannel}")
print(f"DEM High images std value per channel: {demHighStdPerChannel}")

print(f"DEM Meso images mean value per channel: {demMesoMeanPerChannel}")
print(f"DEM Meso images std value per channel: {demMesoStdPerChannel}")