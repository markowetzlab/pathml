import pyvips as pv
import xml.etree.ElementTree as ET

class Annotation:

    def __init__(self, parentSlide, maskingMode='inverse', verbose=False):
        # maskingMode: 'inverse' sets all areas to zero / ''
        self.__verbose = verbose
        self.__parentSlide = parentSlide
        pass

    def loadAnnotationFile(self, annotationFile, fileType):
        if fileType == 'asap':
            tree = ET.parse(annotationFile)  # Open .xml file
            root = tree.getroot()  # Get root of .xml tree
            if root.tag == "ASAP_Annotations":  # Check whether we actually deal with an ASAP .xml file
                if self.__verbose: print('.xml file identified as ASAP annotation collection')  # Display number of found annotations
            else:
                raise Warning('Not an ASAP .xml file')
            self.annotations = root.find('Annotations')  # Find all annotations for this slide
            if self.__verbose: print('XML file valid - ' + str(len(allHeAnnotations)) + ' annotations found.') # Display number of found annotations
        elif fileType == 'qupath':
            pass
        else:
            raise ValueError('Annotation file type was not specifiedd')

        pass

    # packing mode / and other modes for tiling
