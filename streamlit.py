# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:59:11 2020

@author: Jean
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from common_functions import *
from skimage.transform import resize

def main():
    
    
    c1,c2=st.beta_columns(2)
    
    with c1:
        file_dir=st.text_input('File directory',"E:/optorhoa/201028_RPE1_optoRhoA_LA_niceones/")
        filename=st.text_input('File name',"cell5.nd")
    
        exp=get_exp(file_dir+filename)
        st.write("Number of positions : "+str(exp.nbpos))
        st.write("Number of time steps : "+str(exp.nbtime))
        
        ind=range(len(exp.wl))
        wl=st.selectbox('Wavelengths',ind,format_func=lambda i: exp.wl[i].name )
        
        coeff=st.slider('Threshold',0.8,1.2,1.0,0.01)
        
    with c2:
        fig1 = plt.figure()
        img1=np.array(exp.get_first_image(wl))
        
        scalefactor=5
        #original image
        a=plt.imshow(resize(img1, (img1.shape[0] // scalefactor, img1.shape[1] // scalefactor)),cmap='gray')
        #threshold
        thresh = filters.threshold_otsu(img1)
        
        #find mask and contour        
        mask, contours=segment_threshold(img1,coeff*thresh)
        for contour in contours:
            a.axes.plot(list(map(int,contour[:, 1]/scalefactor)), list(map(int,contour[:, 0]/scalefactor)), linewidth=2)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        st.pyplot(fig1)

if __name__ == "__main__":
   main()
   
   