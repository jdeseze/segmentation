# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:36:33 2020

@author: Jean
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data
from sklearn.cluster import KMeans
import skimage.segmentation as seg
from skimage.transform import resize
from skimage.util import img_as_float
from skimage import measure
import cv2 as cv
from scipy import ndimage
from skimage import filters

class WL:
    def __init__(self,name,step=1):
        self.name=name
        self.step=step

class Exp:
    def __init__(self,expname,wl=[],nbpos=1,nbtime=1):
        self.name=expname
        self.nbpos=nbpos
        self.nbtime=nbtime
        self.wl=wl
        self.nbwl=len(wl)
    
    def get_image(self,wl_ind,pos=''):
        return Image.open(self.name+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+pos+'_t1.tif')
    
    def get_first_image(self,wl_ind):
        if self.nbpos>1:
            pos='_s1'
        else:
            pos=''
        return self.get_image(wl_ind,pos)

def get_exp(filename):
    nb_pos=1
    nb_wl=1
    with open(filename,'r') as file:
        for i in range(4):
            file.readline()
        line=file.readline()
        
        #get number of timepoints
        nb_tp=int(line.rstrip().split(', ')[1])
        line=file.readline()
        
        #get positions if exist
        if line.split(', ')[1].rstrip('\n')=='TRUE':
            line=file.readline()
            nb_pos=int(line.split(', ')[1].rstrip('\n'))
            for i in range(nb_pos):
                file.readline()            
        file.readline()
        
        #get number of wavelengths
        line=file.readline()
        nb_wl=int(line.rstrip().split(', ')[1])
    
        #create all new wavelengths
        wl=[]
        for i in range (nb_wl):
            line=file.readline()
            wl.append(WL(line.rstrip().split(', ')[1].strip('\"')))
            file.readline()
    
        #change the time steps
        line=file.readline()
        while line.split(', ')[0].strip('\"')=='WavePointsCollected':
            sep=line.rstrip().split(', ')
            if len(sep)>3:
                wl[int(sep[1])-1].step=int(sep[3])-int(sep[2])
            line=file.readline()
        
        expname=filename.rstrip('.nd')
        
        return Exp(expname,wl,nb_pos,nb_tp)

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def segment_threshold(img,thresh):
    img=img.astype(np.uint8)
    binary = img > thresh
    dil=ndimage.binary_dilation(binary,iterations=2)
    filled=ndimage.binary_fill_holes(dil).astype(int)
    label_img, cc_num = ndimage.label(filled)
    CC = ndimage.find_objects(label_img)
    cc_areas = ndimage.sum(filled, label_img, range(cc_num+1))
    area_mask = (cc_areas < 10000)
    label_img[area_mask[label_img]] = 0
    contours = measure.find_contours(label_img, 0.8)
    return label_img, contours
    
def segment_kmean(image):
    pic=image/255
    pic_n = pic.reshape(pic.shape[0]*pic.shape[1],1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1],1)
    return plt.imshow(cluster_pic,cmap=plt.cm.gray)

def segment_active_contour(image):
    img = filters.gaussian(resize(image, (200, 200)),1)/255
    points = circle_points(500, [100, 100], 90)[:-1]
    #snake = seg.active_contour(image, points,coordinates='rc',alpha=0.001,beta=1, gamma=0.001, w_line=-20, w_edge=-1,max_iterations=25000)
    #snake=seg.chan_vese(img)
    contours = measure.find_contours(img, 0.8)
    fig, ax = image_show(img)
    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
    for snake in contours:    
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3);
    #return snake

def circle_points(resolution, center, radius):   

    s = np.linspace(0, 2*np.pi, resolution)
    r = center[0] + radius*np.sin(s)
    c = center[1] + radius*np.cos(s)

    return np.array([c, r]).T


