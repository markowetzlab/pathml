import sys
sys.path.append('/Users/gehrun01/Desktop/pathml')
from pathml.analysis import Analysis
import matplotlib.pyplot as plt

demoPath = '/Users/gehrun01/Downloads/tumor_001.tif.pml'

testAnalysis = Analysis(demoPath)
mapClass0=testAnalysis.generateInferenceMap(predictionSelector=0)
mapClass1=testAnalysis.generateInferenceMap(predictionSelector=1)
foreground=testAnalysis.generateForegroundMap()

plt.figure()
plt.imshow(foreground,vmin=0, vmax=1,cmap='gray')
plt.title('Foreground')
plt.show(block=False)
plt.figure()
plt.imshow(mapClass0,vmin=0, vmax=1)
plt.title('Class 0')
plt.show(block=False)
plt.figure()
plt.imshow(mapClass1,vmin=0, vmax=1)
plt.title('Class 1')
plt.show()
