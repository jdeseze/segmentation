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
import os
from SessionState import _get_state

def main():
    
    state = _get_state()
        
    c1,c2=st.beta_columns(2)
    
    with c1:
        file_dir=st.text_input('File directory',"C:/Users/Jean/Documents/testpython/PAKiRFP")
        try:
            filename=file_selector(file_dir)
        except:
            "No such directory"
        
        if filename:
            exp=get_exp(filename)
            st.write("Number of positions : "+str(exp.nbpos))
            st.write("Number of time steps : "+str(exp.nbtime))
            
            ind=range(len(exp.wl))
            wl_ind=st.selectbox('Wavelengths',ind,format_func=lambda i: exp.wl[i].name )
            
            coeff=st.slider('Threshold',0.8,1.2,1.0,0.01)
            
        if st.button("Save results"):
            result = Result() 
            filehandler = open("results.obj", 'w') 
            pickle.dump(object, filehandler)
    
        
    with c2:
        img1=np.array(exp.get_first_image(wl_ind))
        #threshold
        thresh = filters.threshold_otsu(img1)
        #find mask and contour        
        mask, contours=segment_threshold(img1,coeff*thresh)
        
        fig1=image_with_seg(img1,contours)
        st.pyplot(fig1)

        
        if st.checkbox("Show last frame segmentation"):

            img2=np.array(exp.get_last_image(wl_ind))
            #threshold
            thresh = filters.threshold_otsu(img2)
            #find mask and contour        
            mask, contours=segment_threshold(img2,coeff*thresh)
            
            fig2=image_with_seg(img2,contours)
            st.pyplot(fig2)
        
        values=['Choose what to do with the area','Define as activation area',"Define as segmentation","Define as not activated area"]
        defarea=st.selectbox('Define area',values,index=0)
        if defarea=='Choose what to do with the area':
            pass
        if defarea=="Define as activation area":
            state.act_area=contours
        if defarea=="Define as segmentation":
            state.masks=calculate_segmentation(exp,coeff,wl_ind)

        if defarea=="Define as not activated area":
            state.notact_area=contours
            defarea='Choose what to do with the area'
        
        if st.button("Plot results for this wavelength"):
            fig=plt.figure()
            whole=[np.mean(np.array(exp.get_image(wl_ind,'',i*exp.wl[wl_ind].step+1))[state.masks[i]]) for i in range(int(exp.nbtime/exp.wl[wl_ind].step))]
            plt.plot(whole)
            st.pyplot(fig)

@st.cache()
def calculate_segmentation(exp,coeff,wl_ind):
    masks=[]
    for frame in range(int(exp.nbtime/exp.wl[wl_ind].step)):
        img=np.array(exp.get_image(wl_ind,'',frame*exp.wl[wl_ind].step+1))
        thresh = filters.threshold_otsu(img)
        mask, contours=segment_threshold(img,coeff*thresh)
        
        masks.append(mask)
    
    return masks
        
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)            


if __name__ == "__main__":
   main()
   
   