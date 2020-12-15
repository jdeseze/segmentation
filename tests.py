# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 10:10:27 2020

@author: Jean
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from common_functions import *
from skimage import data
from skimage import filters
from sklearn.cluster import KMeans
import skimage.segmentation as seg
from skimage.transform import resize
from skimage.util import img_as_float
from skimage import measure
import cv2 as cv
from scipy import ndimage
from skimage.morphology import disk
# =============================================================================
# file_dir="E:/optorhoa/201028_RPE1_optoRhoA_LA_niceones/"
# filename="cell5.nd"
# 
# exp=get_exp(file_dir+filename)
# 
# if exp.nbpos>1:
#     pos='_s1'
# else:
#     pos=''
# img1=exp.get_image(0,pos)
# 
# plt.imshow(img1,cmap='gray')
# =============================================================================

exp=get_exp("E:/optorhoa/201028_RPE1_optoRhoA_LA_niceones/cell5.nd")

img=np.array(exp.get_first_image(0))

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def segment_threshold(img):
    #img1=np.array(img)
    #img=img.astype(np.uint8)
    #img=filters.median(img,disk(5))
    thresh = filters.threshold_otsu(img)
    binary =  (img > thresh)
    dil=ndimage.binary_dilation(binary, structure=None,iterations=1)
    filled=ndimage.binary_fill_holes(dil).astype(binary.dtype)
    plt.imshow(filled, cmap=plt.cm.gray)
    
segment_threshold(img)