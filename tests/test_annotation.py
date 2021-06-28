import sys
sys.path.append('/home/gehrun01/Desktop/pathml')
from pathml.slide import Slide
from pathml.annotation import Annotation

heSlide = Slide('/media/gehrun01/archive-storage/BEST2/BEST2_CAM_0012/BEST2_CAM_0012_HE_1.svs')

tissueTypeAnnotations = Annotation(heSlide)
tissueTypeAnnotations.loadAnnotationFile('/home/gehrun01/Desktop/cytosponge-decision-support/data/annotations/best2-he-qc-annotations/BEST2_CAM_0012_HE_1.xml', fileType='asap')
print(tissueTypeAnnotations.annotations)
