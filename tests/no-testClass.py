import sys
sys.path.append("/home/gehrun01/Desktop/pathml")
from multiprocessing import Pool

from pathml import slide
import pyvips as pv
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


left = 19000
top = 20000
width = 512
height = 512


demoSlidePath = '/home/gehrun01/Desktop/normal_081.tif'

demoImage = slide.Slide(demoSlidePath,verbose=True)
demoImage.setTileProperties(tileSize=400)
#demoImage.detectForeground()
#print(demoImage.getTileCount())
#print(demoImage.tileMetadata.keys())
#
for tile in tqdm(demoImage.iterateTiles()):
    pass

def runtester(address):
    np.mean(demoImage.getTile(address,writeToNumpy=True))
    #print(address)
    return address[0]**address[1]

numbers = demoImage.tileMetadata.keys()
pool = Pool(processes=16)

for _ in tqdm(pool.imap_unordered(runtester, numbers), total=len(numbers)):
    pass

#print(demoImage.tileMetadata[(0,2)])


#pvImage = pv.Image.new_from_file(demoSlidePath)
#print(demoImage.slide.width)
#img = demoImage.slide.extract_area(left,top,width,height)

#np_3d = np.ndarray(buffer=img.write_to_memory(),
#                   dtype=format_to_dtype[img.format],
#                   shape=[img.height, img.width, img.bands])
#plt.imshow(np_3d)
#plt.show()
