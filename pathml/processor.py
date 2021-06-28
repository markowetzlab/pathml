import numpy as np
from tqdm import tqdm
import os
import torch
import pickle
from .utils.torch.WholeSlideImageDataset import WholeSlideImageDataset

class Processor:

    __verbosePrefix = '[PathML] '

    def __init__(self, slideObject, verbose=False):
        self.__verbose = verbose
        self.__slideObject = slideObject

    def applyModel(self, modelZip, batch_size, predictionKey = 'prediction'):
        device, model, data_transforms = modelZip
        pathSlideDataset = WholeSlideImageDataset(self.__slideObject, transform=data_transforms)
        pathSlideDataloader = torch.utils.data.DataLoader(pathSlideDataset, batch_size=batch_size, shuffle=False, num_workers=16)
        for inputs in tqdm(pathSlideDataloader):
            inputTile = inputs['image'].to(device)
            output = model(inputTile)
            output = output.to(device)

            batch_prediction = torch.nn.functional.softmax(
                output, dim=1).cpu().data.numpy()

            # Reshape it is a Todo - instead of for looping
            for index in range(len(inputTile)):
                tileAddress = (inputs['tileAddress'][0][index].item(),
                               inputs['tileAddress'][1][index].item())
                self.__slideObject.appendTag(tileAddress, predictionKey, batch_prediction[index, ...])
        return self

    def adoptKeyFromTileDictionary(self, upsampleFactor=1):
        for orphanTileAddress in self.__slideObject.iterateTiles():
            self.__slideObject.tileDictionary[orphanTileAddress].update({'x': self.__slideObject.tileDictionary[orphanTileAddress]['x']*upsampleFactor,
                                                            'y': self.__slideObject.tileDictionary[orphanTileAddress]['y']*upsampleFactor,
                                                            'width': self.__slideObject.tileDictionary[orphanTileAddress]['width']*upsampleFactor,
                                                            'height': self.__slideObject.tileDictionary[orphanTileAddress]['height']*upsampleFactor,})
        return self.__slideObject
