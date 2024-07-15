import earthnet as en
import matplotlib.pyplot as plt
import numpy as np

predCubePath   = '/home/nikoskot/experiments/videoSwinUNetV2_09-07-2024_20:41:27/predictions/limited_train_data/29SND/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz'
targetCubePath = '/home/nikoskot/earthnetThesis/EarthnetDataset/train/29SND/29SND_2017-06-20_2017-11-16_1849_1977_3641_3769_28_108_56_136.npz'

fig = en.cube_gallery(targetCubePath, variable='rgb', save_path='/home/nikoskot/3')