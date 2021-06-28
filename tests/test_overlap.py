import sys
import time
import numpy as np
import pyvips as pv
import warnings
from PIL import Image
import imageio
import glob
#from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import argparse
sys.path.append('/Users/gehrun01/Desktop/pathml')
from pathml import slide

pathSlide = slide.Slide("/Users/gehrun01/Desktop/BEST2_CAM_0654_HE_1.svs",level=0)
print(pathSlide.slideProperties)
quit()
pathSlide.setTileProperties(tileSize=400)
pathSlide.detectForeground(threshold=95) # scale between 0 100







foregroundImage = np.zeros((pathSlide.numTilesInY,pathSlide.numTilesInX))
for key, val in pathSlide.tileMetadata.items():
    foregroundImage[key[1], key[0]] = int(val['foreground'] if 'foreground' in val else 0)

print(pathSlide.getTileCount())
print(pathSlide.getTileCount(foregroundOnly=True))

print(pathSlide.ind2sub(0))
print(pathSlide.ind2sub(0, foregroundOnly=True))

#plt.figure()
#plt.imshow(pathSlide.lowMagSlideRGB)
#plt.figure()
#plt.imshow(foregroundImage)
#plt.show()
