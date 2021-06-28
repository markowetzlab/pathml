import pyvips as pv
import matplotlib.pyplot as plt
import numpy as np

demoSlide = pv.Image.new_from_file('/Users/gehrun01/Desktop/BEST2_CAM_0654_HE_1.svs',level=0)
regionFetcher = pv.Region.new(demoSlide)

entireImage = regionFetcher.fetch(40000,40000,2147,1015)

entireActualImage = np.ndarray(buffer=entireImage, dtype=np.uint8, shape=[1015,2147,4])

plt.figure()
plt.imshow(entireActualImage)
plt.show()

slideProperties = {x: demoSlide.get(x) for x in demoSlide.get_fields()}




#print(slideProperties)
