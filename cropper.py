import numpy as np
import pims
from numpy import unravel_index
import cv2
import os
from tqdm import tqdm_notebook as tqdm
import skimage


def _crop(imgToCrop,imgMask,maskSize,wellSize,aspectRatio):

    """Crop function. Works only on 2D images.
    """

    (xc, yc) = _getCenter(imgMask,maskSize,wellSize,aspectRatio)

    cropDist = maskSize*aspectRatio
    startx = max(xc-(cropDist//2), 0)
    starty = max(yc-(cropDist//2), 0)

    return imgToCrop[int(starty):int(starty+cropDist),int(startx):int(startx+cropDist)]

def _makeCircMask(maskSize, wellSize, aspectRatio):

    cropDist = maskSize*aspectRatio

    X = np.arange(0, cropDist)
    Y = np.arange(0, cropDist)
    X, Y = np.meshgrid(X, Y)

    mask = ((np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) > (wellSize*aspectRatio)//2 - 20*aspectRatio) &
            (np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) < (wellSize*aspectRatio)//2 + 20*aspectRatio))

    return mask.astype(np.int)

def _getCenter(imgMask,maskSize,wellSize,aspectRatio):

    mask = _makeCircMask(maskSize,wellSize,aspectRatio)

    conv = cv2.filter2D(imgMask, cv2.CV_32F, mask)

    return unravel_index(conv.argmin(), conv.shape)

def _cropByWell(PATH,maskSize,wellSize,aspectRatio):

    if not os.path.exists(PATH + r'\cropped'):
        os.mkdir(PATH + r'\\' + 'cropped')

    img = pims.ImageSequence(PATH + '\\*.tif', as_grey=True)
    i = 0

    for im in tqdm(img):

        skimage.external.tifffile.imsave(PATH + r'\\cropped\\' + 'crop_%d.tif' %i,
            _crop(im, im,maskSize,wellSize,aspectRatio))
        i += 1

    return

""" ====== SPHEROID CROPPING ======= """

def _verifDim(im):

    if not im.ndim == 4:

        return False

    return True

def _loadImage(path):

    """
    ====== COMMENT ======

    The function needs to be improved so as to add new channels without
    requiring to manually add new channels by hand.

    """

    image_list = []
    for filename in tqdm(sorted(os.listdir(path))): #assuming tif

        if '.tif' in filename:

            im = io.imread(path + '/' + filename)

            image_list.append(im[:,:,:])

    return np.asarray(image_list)

def _getCenterBary(im,livePosition):

    """ IDs barycenter of image.

    """

    value = np.percentile(img[:,:,:,livePosition], 99.9)
    temp = img[:,:,:,livePosition] > value
    z, x, y = np.nonzero(temp)

    return np.mean(z), np.mean(x), np.mean(y)


def _crop3D(imgToCrop,livePosition,maskSize,aspectRatio):

    """Crop function. Works only on 3D images. Hypothesis that image arranged
    along 'z, x, y' dimensions.

    ====== Variable ======

     - aspectRatio: mu-to-px conversion rate

    """

    if not _verifDim(imgToCrop):

        return print("Image dimension not equal to 4")

    zc, yc, xc = _getCenterBary(imgToCrop,livePosition)

    cropDist = maskSize*aspectRatio
    dz,dx,dy,nChannels = np.shape(imgToCrop)

    startx = int(max(xc-(cropDist//2), 0))
    starty = int(max(yc-(cropDist//2), 0))
    endx = int(min(xc+(cropDist//2), dx))
    endy = int(min(yc+(cropDist//2), dy))

    return imgToCrop[:, starty:endy,startx:endx,:]