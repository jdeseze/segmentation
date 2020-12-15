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
from scipy import ndimage
from skimage import filters
import cv2

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
    
    def get_image(self,wl_ind,pos='',timepoint=1):
        return Image.open(self.name+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+pos+'_t'+str(timepoint)+'.tif')
    
    def get_first_image(self,wl_ind,pos='',timepoint=1):
        timepoint=1
        return self.get_image(wl_ind,pos,timepoint)
    
    def get_last_image(self,wl_ind,pos='',timepoint=1):
        last_ind=int(self.nbtime/self.wl[wl_ind].step-1)*self.wl[wl_ind].step+1
        return self.get_image(wl_ind,pos,last_ind)

class Result:
    def __init__(self, exp,ill=[],noill=[],whole=[]):
        self.exp=exp
        self.ill=ill
        self.noill=noill
        self.whole=whole

def image_with_seg(img1,contours):
    fig1 = plt.figure()
    
    scalefactor=1
    #original image
    a=plt.imshow(resize(img1, (img1.shape[0] // scalefactor, img1.shape[1] // scalefactor)),cmap='gray')
    #find mask and contour        
    for contour in contours:
        a.axes.plot(list(map(int,contour[:, 1]/scalefactor)), list(map(int,contour[:, 0]/scalefactor)), linewidth=2)
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    return fig1
    

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

# =============================================================================
# # import the necessary packages
# import cv2
# import argparse
# 
# # now let's initialize the list of reference point
# 
# 
# def shape_selection(event, x, y, flags, param):
#     # grab references to the global variables
#     global ref_point, crop
# 
#     # if the left mouse button was clicked, record the starting
#     # (x, y) coordinates and indicate that cropping is being performed
#     if event == cv2.EVENT_LBUTTONDOWN:
#         ref_point = [(x, y)]
# 
#     # check to see if the left mouse button was released
#     elif event == cv2.EVENT_LBUTTONUP:
#         # record the ending (x, y) coordinates and indicate that
#         # the cropping operation is finished
#         ref_point.append((x, y))
# 
#         # draw a rectangle around the region of interest
#         cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
#         cv2.imshow("image", image)
# 
# 
# def input_rectangle(img):
#     global image
#     image=img
#     clone = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     cv2.namedWindow("image")
#     cv2.setMouseCallback("image", shape_selection)
#        
#     # keep looping until the 'q' key is pressed
#     while True:
#         # display the image and wait for a keypress
#         cv2.imshow("image", image)
#         key = cv2.waitKey(1) & 0xFF
#     
#         # press 'r' to reset the window
#         if key == ord("r"):
#             image = clone.copy()
#     
#         # if the 'c' key is pressed, break from the loop
#         elif key == ord("c"):
#             break
#     
#     # close all open windows
#     cv2.destroyAllWindows() 
# 
# image=np.array(Image.open('test2.tif'))
# clone = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imshow("image", image)
# windowName = 'image'
# rectI = selectinwindow.dragRect
# selectinwindow.init(rectI, image, windowName, 500, 500)
# cv2.setMouseCallback(windowName, selectinwindow.dragrect, rectI)
# rectI.outRect
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# =============================================================================


# Making The Blank Image
# =============================================================================
# image = np.zeros((512,512,3))
# drawing = False
# ref_point=[]
# =============================================================================
# Adding Function Attached To Mouse Callback
# =============================================================================
# def draw(event,x,y,flags,params):
#     global ix,iy,drawing
#     # Left Mouse Button Down Pressed
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ref_point.append((x,y))
#     if event == cv2.EVENT_LBUTTONUP:
#         if(drawing==True):
#             #For Drawing Line
#             #cv2.line(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
#             ref_point.append((x,y))
#             # For Drawing Rectangle
#             cv2.rectangle(image,pt1=ref_point[0],pt2=ref_point[1],color=(255,255,255),thickness=3)
#     if(event==4):
#         drawing = False
# =============================================================================



# =============================================================================
# # Making Window For The Image
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# cv2.imshow("output", image) 
# #cv2.waitKey(0)
# 
# # Adding Mouse CallBack Event
# cv2.setMouseCallback("output",draw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #cv2.waitKey(1)
# print(ref_point[0])
# 
# plt.imshow(image)
# 
# print(np.max(image))
# =============================================================================
