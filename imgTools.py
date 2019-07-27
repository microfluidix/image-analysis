import cv2
import numpy as np


def cv2Bandpass(myImage, filterLargeDia, filterSmallDia, satPercent=0):
    #get the gaussian kernels
    gaussianLarge = cv2.getGaussianKernel(filterLargeDia*3+1, filterLargeDia/2)
    gaussianSmall = cv2.getGaussianKernel(filterSmallDia*3+1, filterSmallDia/2)
    #filter the images
    imgFilteredLarge = cv2.sepFilter2D(myImage,-1,gaussianLarge, gaussianLarge)
    imgFilteredSmall = cv2.sepFilter2D(myImage,-1,gaussianSmall, gaussianSmall)
    bpResult = imgFilteredSmall.astype(float)-imgFilteredLarge.astype(float)
    
    #saturate or just normalize the result
    if (satPercent>0 and satPercent<=1):
        bpResultSat = saturateImage(bpResult, satPercent)
    else:
        bpResultSat = np.uint8(imStretchNorm(bpResult)*255)
        
    return bpResultSat



def imStretchNorm(filteredImage):
    filteredImage = filteredImage.astype(float)
    sn = (filteredImage-np.min(filteredImage))/(np.max(filteredImage) - np.min(filteredImage))
    return sn



def saturateImage(filteredImage, satPercent):
    
    filteredImageSN = np.uint8(imStretchNorm(filteredImage)*255)
    
    hist = cv2.calcHist([filteredImageSN],[0],None,[256],[0,256])
    cumhist = np.cumsum(hist)/np.size(filteredImageSN)
    xmin = np.nonzero(cumhist<satPercent/2)[0][-1]
    xmax = np.nonzero(cumhist>(1-satPercent/2))[0][0]
    filteredImageSat = filteredImageSN
    filteredImageSat[filteredImageSat<xmin] = xmin
    filteredImageSat[filteredImageSat>xmax] = xmax
    filteredImageSat = np.uint8(imStretchNorm(filteredImageSat)*255)
    
    return filteredImageSat



def averageAllImages(allImagesPath):
    a = 0
    nbImages = len(allImagesPath)
    for currImPath in allImagesPath:
        myImage = cv2.imread(currImPath, cv2.IMREAD_GRAYSCALE)
        myImage = myImage.astype(float)
        a = a+myImage
    a = a/nbImages
    return a
