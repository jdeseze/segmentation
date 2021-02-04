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

# =============================================================================
# exp=get_exp("E:/optorhoa/201028_RPE1_optoRhoA_LA_niceones/cell5.nd")
# 
# img=np.array(exp.get_first_image(0))
# 
# def image_show(image, nrows=1, ncols=1, cmap='gray'):
#     fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
#     ax.imshow(image, cmap='gray')
#     ax.axis('off')
#     return fig, ax
# 
# def segment_threshold(img):
#     #img1=np.array(img)
#     #img=img.astype(np.uint8)
#     #img=filters.median(img,disk(5))
#     thresh = filters.threshold_otsu(img)
#     binary =  (img > thresh)
#     dil=ndimage.binary_dilation(binary, structure=None,iterations=1)
#     filled=ndimage.binary_fill_holes(dil).astype(binary.dtype)
#     plt.imshow(filled, cmap=plt.cm.gray)
#     
# segment_threshold(img)
# =============================================================================
# =============================================================================
# 
# =============================================================================
# import tkinter as tk
# from PIL import ImageTk
# 
# class GUI_SelectArea(tk.Toplevel):
#     def __init__(self,image):
#         super().__init__()
#         self.withdraw()
#         #self.attributes('-fullscreen', True)
# 
#         self.canvas = tk.Canvas(self)
#         self.canvas.pack(fill="both",expand=True)
# 
#         self.image = ImageTk.PhotoImage(image)
#         self.photo = self.canvas.create_image(0,0,image=self.image,anchor="nw")
# 
#         self.x, self.y = 0, 0
#         self.rect, self.start_x, self.start_y = None, None, None
#         self.deiconify()
# 
#         self.canvas.tag_bind(self.photo,"<ButtonPress-1>", self.on_button_press)
#         self.canvas.tag_bind(self.photo,"<B1-Motion>", self.on_move_press)
#         self.canvas.tag_bind(self.photo,"<ButtonRelease-1>", self.on_button_release)
# 
#     def on_button_press(self, event):
#         self.start_x = event.x
#         self.start_y = event.y
#         self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')
# 
#     def on_move_press(self, event):
#         curX, curY = (event.x, event.y)
#         self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
# 
#     def on_button_release(self, event):
#         bbox = self.canvas.bbox(self.rect)
#         
#         self.withdraw()
#         #self.attributes('-fullscreen', False)
#         #self.title("Image grabbed")
#         self.canvas.destroy()
#         self.deiconify()
#         self.quit()
#         return bbox
#         #tk.Label(self,image=self.image).pack()
#         #self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, curX, curY, outline='red')
# 
# #root = GUI_SelectArea(img1)
# 
# #root.mainloop()
# =============================================================================

# =============================================================================
#                 if st.button("Define activation area"):
#                     image=img1
#                     drawing = False
#                     ref_point=[]
#         # =============================================================================
#         #             for i,j in zip(range(min(ref_point[0][0],ref_point[1][0]),max(ref_point[0][0],ref_point[1][0])),range(min(ref_point[0][1],ref_point[1][1]),max(ref_point[0][1],ref_point[1][1]))):
#         #                 act_area[i][j]=1
#         # =============================================================================
#                     def draw(event,x,y,flags,params):
#                         global ix,iy,drawing
#                         # Left Mouse Button Down Pressed
#                         if event == cv2.EVENT_LBUTTONDOWN:
#                             drawing = True
#                             ref_point.append((x,y))
#                         if event == cv2.EVENT_LBUTTONUP:
#                             if(drawing==True):
#                                 #For Drawing Line
#                                 #cv2.line(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
#                                 ref_point.append((x,y))
#                                 # For Drawing Rectangle
#                                 cv2.rectangle(image,pt1=ref_point[0],pt2=ref_point[1],color=(255,255,255),thickness=-1)
#                         if(event==4):
#                             drawing = False
#                     
#                     cv2.namedWindow("Draw rectangle", cv2.WINDOW_NORMAL)
#                     cv2.setWindowProperty("Draw rectangle",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#                     cv2.setWindowProperty("Draw rectangle",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
#                     cv2.imshow("Draw rectangle", image) 
#                     cv2.setMouseCallback("Draw rectangle",draw)
#                     cv2.waitKey(0)
#                     cv2.destroyAllWindows()
#                     act_area=image>0
# =============================================================================