import cv2
import numpy as np


def cv2Bandpass(myImage, filterLargeDia, filterSmallDia, satPercent=0):
    #apply a bandpass filter on an image
    
    #myImage: image to bandpass (opened e.g. with cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE) )
    #filterLargeDia: EVEN integer, diameter of largest pattern of interest in myImage
    #filterSmallDia: EVEN integer, diameter of smallest pattern of interest in myImage
    #satPercent: float between 0 and 1, setting the contrast of the resulting image
    #If satPercent = 0: the resulting bandpassed image has a histogram stretched between 0 and 255 uniformly
    #If satPercent > 0: the resulting bandpassed image has a histogram stretched between 0 and 255 and a fraction satPercent of the pixels are saturated
    
    #Output: grayscale image (uint8 2d array) with pixel values stretching between 0 and 255
    
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
    #takes a grayscale image, stretches its histogram between 0 and 1 to improve contrast
    #filteredImage : input image, 2d array of whatever type
    #output: grayscale image (2d array) with pixel values between 0 and 1
    filteredImage = filteredImage.astype(float)
    sn = (filteredImage-np.min(filteredImage))/(np.max(filteredImage) - np.min(filteredImage))
    return sn



def saturateImage(filteredImage, satPercent):
    #takes a grayscale image, stretches its histogram between 0 and 255 and saturates part of the pixels
    #filteredImage: image whose contrast we want to improve
    #satPercent: float between 0 and 1 (not including 0 and 1), fraction of pixels we want to saturate 
    #(careful, may not work with satPercent=0 or satPercent=1)
    
    #output: grayscale image of type uint8, with pixel values between 0 and 255
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
    #averages all the images in a given path
    #allImagesPath: path with all the images, obtained e.g. with sorted(glob.glob(os.path.join(folderPathWithAllImages, '*.png')))
    #output: average of all images -- float type 2d array
    a = 0
    nbImages = len(allImagesPath)
    for currImPath in allImagesPath:
        myImage = cv2.imread(currImPath, cv2.IMREAD_GRAYSCALE)
        myImage = myImage.astype(float)
        a = a+myImage
    a = a/nbImages
    return a
