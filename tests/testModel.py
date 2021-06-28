import sys
sys.path.append('/home/gehrun01/Desktop/pathml')
from pathml.slide import Slide
from pathml.analysis import Analysis
from pathml.processor import Processor
from pathml.models.tissuedetector import tissueDetector
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


pathmlSlide = Slide('/media/gehrun01/Cytosponge-Store/miRNA-1/Biopsies/OCCAMS_AH_183_OAC_Biopsy.svs').setTileProperties(tileSize=224)
tissueForegroundSlide = Slide('/media/gehrun01/Cytosponge-Store/miRNA-1/Biopsies/OCCAMS_AH_183_OAC_Biopsy.svs', level=1).setTileProperties(tileSize=448, tileOverlap=0.5)
tmpProcessor = Processor(tissueForegroundSlide)
tissueForegroundTmp = tmpProcessor.applyModel(tissueDetector(), batch_size=20, predictionKey='tissue_detector').adoptKeyFromTileDictionary(upsampleFactor=4)

predictionMap = np.zeros([tissueForegroundTmp.numTilesInY, tissueForegroundTmp.numTilesInX,3])
for address in tissueForegroundTmp.iterateTiles():
    if 'tissue_detector' in tissueForegroundTmp.tileDictionary[address]:
        predictionMap[address[1], address[0], :] = tissueForegroundTmp.tileDictionary[address]['tissue_detector']

predictionMap2 = np.zeros([pathmlSlide.numTilesInY, pathmlSlide.numTilesInX])
predictionMap1res = resize(predictionMap, predictionMap2.shape, order=0, anti_aliasing=False)
for address in pathmlSlide.iterateTiles():
    pathmlSlide.tileDictionary[address].update({'tissueLevel': predictionMap1res[address[1], address[0]][2]})

plt.figure()
plt.imshow(predictionMap)
plt.show(block=False)

plt.figure()
plt.imshow(predictionMap1res)
plt.show()

print(pathmlSlide.tileDictionary)
quit()
testAnalysis = Analysis(pathmlSlide.tileDictionary)
mapClass0=testAnalysis.generateInferenceMap(predictionSelector=0,predictionKey='foreground')
mapClass1=testAnalysis.generateInferenceMap(predictionSelector=1,predictionKey='foreground')
mapClass2=testAnalysis.generateInferenceMap(predictionSelector=2,predictionKey='foreground')

plt.figure()
plt.imshow(mapClass0,vmin=0, vmax=1)
plt.colorbar()
plt.title('Class 0')
plt.show(block=False)
plt.figure()
plt.imshow(mapClass1,vmin=0, vmax=1)
plt.colorbar()
plt.title('Class 1')
plt.show(block=False)
plt.figure()
plt.imshow(mapClass2,vmin=0, vmax=1)
plt.colorbar()
plt.title('Class 2')
plt.show()
