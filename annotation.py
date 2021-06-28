import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

class SlideAnnotation:
    def __init__(self, annotationFilePath, verbose=False):

    def loadASAPAnnotations(file, imgSize, tileSize):
        tree = ET.parse(file)  # Open .xml file
        root = tree.getroot()  # Get root of .xml tree
        if root.tag != "ASAP_Annotations":  # Check whether we actually deal with an ASAP .xml file
            raise Warning('Not an ASAP .xml file')
        allAnnotations = root.find('Annotations')  # Find all annotations for this slide
        print('XML file valid - ' + str(
            len(allAnnotations)) + ' annotations found.')  # Display number of found annotations
        tilePositions = []  # Initialize output argument which contains tile positions
        tilePositionClass = []  # Initialize output argument which contains class of tile positions
        for className in classNames:
            # Extract width and height of images
            width = imgSize[0]
            height = imgSize[1]
            # Create image and draw filled polygons for each annotation of this class
            img = Image.new('L', (width, height), 0)
            for annotation in allAnnotations:
                if annotation.attrib['PartOfGroup'] == className:
                    annotationTree = annotation.find('Coordinates')
                    x = []
                    y = []
                    polygon = ()
                    for coordinate in annotationTree:
                        info = coordinate.attrib
                        polygon = polygon + (float(info['X']), float(info['Y']))
                    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=255)
            # Crop the image to get rid of non-square tiles
            img = img.crop((0, 0, (width // tileSize) * tileSize, (height // tileSize) * tileSize))
            # Resize image to comply with tile size for tile positions
            maskImg = img.resize((width // tileSize, height // tileSize), resample=PIL.Image.NEAREST)
            maskImg = np.asarray(maskImg)
            ii = np.where(maskImg == 255)
            tilePositionClass.append(className)
            tilePositions.append(tuple(zip(*ii)))
        return tilePositions, tilePositionClass