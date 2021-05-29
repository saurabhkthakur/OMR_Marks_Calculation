import numpy as np
import cv2

def translate(image, x, y):
    M = np.float32([[1,0,x], [0,1,y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0] ))

    return shifted

def rotate(image, angle, center=None, scale=1.0):
    (h,w) = image.shape[:2]
    center = (h//2, w//2)

    M = cv2.getRotationMatrix2D(center, 45, 1)
    rotated = cv2.warpAffine(image, M, (w,h))

    return rotated

def resize(image, width=None, height=None, inter= cv2.INTER_AREA):

    if width is None and height is None:
        return image

    if width == None:
        r = height/image.shape[0]
        dim = (int(image.shape[1]*r), height)

    else:

        r = width/ image.shape[1]
        dim = (width, int(image.shape[0]*r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
