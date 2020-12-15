# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:59:11 2020

@author: Jean
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import pandas as pd
from common_functions import *
from skimage.transform import resize
import os
from SessionState import _get_state
from skimage import filters
import cv2

def main():
    state = _get_state()
        
    c1,c2=st.beta_columns(2)
    
    with c1:
        file_dir=st.text_input('File directory',"E:/optorhoa/201208_RPE_optoRhoA_PAKiRFP")
        try:
            filename=file_selector(file_dir)
        except:
            filename=None
            "No such directory or no .nd file in this directory"
        
        if filename:
            exp=get_exp(filename)
            st.write("Number of positions : "+str(exp.nbpos))
            st.write("Number of time steps : "+str(exp.nbtime))
            if exp.nbpos==1:
                state.pos=''
            else:
                state.pos='_s'+str(st.selectbox('Position',range(1,exp.nbpos+1)))
            
            ind=range(len(exp.wl))
            wl_ind=st.selectbox('Wavelengths',ind,format_func=lambda i: exp.wl[i].name )
            
            coeff=st.slider('Threshold',0.8,1.2,1.0,0.01)
            
        if st.button("Save results"):
            result = Result() 
            filehandler = open("results.obj", 'w') 
            pickle.dump(object, filehandler)
    
        
    with c2:
        if filename:
            img1=np.array(exp.get_first_image(wl_ind,state.pos))
            #threshold
            filtered=filters.median(img1.astype(np.uint8))
            thresh = filters.threshold_otsu(filtered)
            #find mask and contour        
            mask, contours=segment_threshold(filtered,coeff*thresh)
            
            fig1=image_with_seg(img1,contours)
            st.pyplot(fig1)

        
        if st.checkbox("Check last frame segmentation"):

            img2=np.array(exp.get_last_image(wl_ind,state.pos))
            #threshold
            thresh = filters.threshold_otsu(img2)
            #find mask and contour        
            mask, contours=segment_threshold(img2,coeff*thresh)
            
            fig2=image_with_seg(img2,contours)
            st.pyplot(fig2)
        
        values=['Choose what to do with the area',"Define activation area","Define as segmentation","Define not activated area"]
        defarea=st.selectbox('Define area',values,index=0)
        if defarea==values[0]:
            pass
        if st.button("Define activation area"):
            image=img1
            drawing = False
            ref_point=[]
# =============================================================================
#             for i,j in zip(range(min(ref_point[0][0],ref_point[1][0]),max(ref_point[0][0],ref_point[1][0])),range(min(ref_point[0][1],ref_point[1][1]),max(ref_point[0][1],ref_point[1][1]))):
#                 state.act_area[i][j]=1
# =============================================================================
            def draw(event,x,y,flags,params):
                global ix,iy,drawing
                # Left Mouse Button Down Pressed
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    ref_point.append((x,y))
                if event == cv2.EVENT_LBUTTONUP:
                    if(drawing==True):
                        #For Drawing Line
                        #cv2.line(image,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
                        ref_point.append((x,y))
                        # For Drawing Rectangle
                        cv2.rectangle(image,pt1=ref_point[0],pt2=ref_point[1],color=(255,255,255),thickness=-1)
                if(event==4):
                    drawing = False
            
            cv2.namedWindow("Draw rectangle", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Draw rectangle",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.setWindowProperty("Draw rectangle",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
            cv2.imshow("Draw rectangle", image) 
            cv2.setMouseCallback("Draw rectangle",draw)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            state.act_area=image>0
                                                        
        if defarea==values[3]:
            ref_point = []
            input_rectangle(image)
            state.notact_area=contours
            
        if defarea==values[2]:
            state.masks=calculate_segmentation(exp,coeff,wl_ind,state.pos)
        
        if st.button("Plot results for this wavelength"):
            whole=[]
            activated=[]
            for i in range(int(exp.nbtime/exp.wl[wl_ind].step)):
                img=np.array(exp.get_image(wl_ind,state.pos,i*exp.wl[wl_ind].step+1))
                whole.append(np.mean(img[state.masks[i]]))
                activated.append(np.mean(img[state.act_area*state.masks[i]]))
            fig=plt.figure()
            plt.plot(whole)
            plt.plot(activated)
            st.pyplot(fig)
            
        if st.checkbox("Show activation area"):
            fig,ax=image_show(state.act_area*255)
            st.pyplot(fig)

@st.cache
def calculate_segmentation(exp,coeff,wl_ind,pos):
    masks=[]
    for frame in range(int(exp.nbtime/exp.wl[wl_ind].step)):
        img=np.array(exp.get_image(wl_ind,pos,frame*exp.wl[wl_ind].step+1))
        thresh = filters.threshold_otsu(img)
        mask, contours=segment_threshold(img,coeff*thresh)
        
        masks.append(mask)
    
    return masks
        
def file_selector(folder_path='.'):
    filenames = [f for f in os.listdir(folder_path) if f.endswith('.nd')]
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)            


if __name__ == "__main__":
   main()
   
   