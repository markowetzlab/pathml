import sys
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean

sys.path.append('/home/gehrun01/Desktop/pathml')
from pathml import slide

pathmlSlide = slide.Slide('/home/gehrun01/Desktop/test.tif')
pathmlSlide.setTileProperties(tileSize=224)
pathmlSlide.run(num = range(10000))

quit()

#print(pathmlSlide.slideProperties)
#pathmlSlide.detectForeground(threshold=95)
pathmlSlide.detectForeground(threshold='triangle', level=5)

for tileAddress in pathmlSlide.iterateTiles():
    if pathmlSlide.tileMetadata[tileAddress]['foreground']:
        print(pathmlSlide.tileMetadata[tileAddress])




plt.figure()
plt.imshow(pathmlSlide.lowMagSlideRGB)
#plt.imshow(resize(pathmlSlide.foregroundMask(), pathmlSlide.lowMagSlideRGB.shape, anti_aliasing=False, order=0), alpha=0.5)
plt.figure()
plt.imshow(pathmlSlide.foregroundMask())

plt.show()
