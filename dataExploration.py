import earthnet as en
import matplotlib.pyplot as plt
import numpy as np

predCubePath   = '/home/nikoskot/experiments/videoSwinUNetV2_09-07-2024_20:41:27/predictions/limited_train_data/29SND/29SND_2017-06-10_2017-11-06_2105_2233_2873_3001_32_112_44_124.npz'
targetCubePath = '/home/nikoskot/EarthnetDataset/iid_test_split/target/29SND/target_29SND_2017-06-20_2017-11-16_953_1081_3641_3769_14_94_56_136.npz'

fig = en.cube_gallery(predCubePath, variable='rgb', save_path='/home/nikoskot/1')