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
from skimage import filters
import cv2
from streamlit_drawable_canvas import st_canvas
from SessionState import _get_state
import threading
import pickle
import copy
import plotly.express as px

def main():
    st.set_page_config(page_title="Segmentation", page_icon=":microscope:",layout="wide")
    
    state = _get_state()
    pages = {
        "Make measures": page_measures,
        "Look at results": page_results,
    }
    
    with st.sidebar:
        
        page=st.selectbox('Choose page',tuple(pages.keys()))
        
        with st.beta_expander('Choose experiment'):
            state.file_dir=st.text_input('File directory',"E:/optorhoa/201210_RPE1_optoRhoa_RBDiRFP/")
            try:
                state.filename=file_selector(state.file_dir)
            except:
                state.filename=None
                st.write("No such directory or no .nd file in this directory")
            
            if st.button('Clear state'):
                state.clear()            
    
    if state.resultsfile==None:
        state.resultsfile='./pdresults.pkl'
        resultspd=pd.DataFrame()
        resultspd.to_pickle(state.resultsfile)
        with open('./results.pkl', 'wb') as output:
            results=[]
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
    
    if not state.filename==None:
        pages[page](state)
    

    state.new_exp=False
    state.sync()
               

def page_measures(state):
    with st.sidebar:
        exp=get_exp(state.filename)
        state.exp=exp
        
        with st.beta_expander('Experiment'):
            st.write("Number of positions : "+str(exp.nbpos))
            st.write("Number of time steps : "+str(exp.nbtime))
            st.write("Time step : "+str(exp.timestep)+' sec')
            state.pos=st.selectbox('Position',range(1,exp.nbpos+1))
        
        with st.beta_expander('Segmentation and activation'):
            inds=range(exp.nbwl)
            
            state.coeff_seg=st.slider('Threshold',0.7,1.3,1.0,0.01,key='seg')
            state.wl_act=st.selectbox('Activation channel',inds,format_func=lambda i: state.exp.wl[i].name,key='act')
            
            seg_options=['Import region','Draw rectangle','Segment channel']
            state.def_rgn=st.selectbox('Segmentation',range(3),format_func=lambda i: seg_options[i])
            if state.def_rgn==2:
                state.draw=0
                state.isrgn=0
                state.wl_seg=st.selectbox('Segmentation channel',inds,format_func=lambda i: state.exp.wl[i].name,key='seg')
                state.coeff_act=st.slider('Threshold',0.7,1.3,1.0,0.01,key='act')
            if state.def_rgn==0:
                state.draw=0
                state.isrgn=1
                try:
                    rgn_file=file_selector(state.file_dir,extension='.rgn')
                    if st.button('Load region'):
                        with open(rgn_file) as file:
                            line=file.readline().rstrip().split(', ')
                            x,y=int(line[2].split(' ')[1]),int(line[2].split(' ')[2])
                            w,l=int(line[6].split(' ')[2]),int(line[6].split(' ')[3])
                            mask=np.zeros((1024,1024))
                            mask[y:y+l,x:x+w]=1
                            state.rgn=mask
                            contour=np.zeros((1024,1024))
                            contour[y:y+l,x]=1
                            contour[y:y+l,x+w]=1
                            contour[y,x:x+w]=1
                            contour[y+l,x:x+w]=1
                            state.rgn_contour=contour
                            state.isrgn=1
                except:
                    pass
            if state.def_rgn==1:
                state.draw=1
                state.isrgn=0
        
        with st.beta_expander('Measures and movie'):
            inds=range(len(state.exp.wl))
            state.wls_meas=st.multiselect('Measurement channel',inds,format_func=lambda i: state.exp.wl[i].name,key='meas')
            state.prot=st.checkbox('Makes protrusions?')
            #st.write(state.wl_meas)
            if st.button("Compute and save results"):
                #st.write(int(state.exp.nbtime/state.exp.wl[wl].step)*state.exp.wl[wl].step)
                threading.Thread(target=save_results,args=[state]).start()    
            
            state.crop=st.checkbox('Crop')
            if st.button('Make movies'):
                th=threading.Thread(target=make_all_movies,args=[state]) 
                th.start()
            
        
        
# =============================================================================
#         debug=st.sidebar.beta_expander("Debug",expanded=False)
#         with debug:
#             state.frame=st.selectbox('Time Frame', range(1,exp.nbtime+1),key='selectbox')
# =============================================================================
            
    if (not state.temppos==state.pos) or (not state.exp.name==state.tempexpname):
        state.new_exp=True
        state.temppos=state.pos
        state.tempexpname=state.exp.name
        state.rgn=None
    
    #decide how many columns should be done       
    try:
        col=list(st.beta_columns(int((len(exp.wl)+1)/2)))
        state(wlthres=[None]*4)
        state(fig=[None]*4) 
        state(img=[None]*4)
    except:
        st.write('Please load a valid experiment first')
        col=None
    
    #Create columns only if experiment is loaded
    if col is not None:
        
        for i in range(len(exp.wl)):
                
            #checking if I should update the temporary image
            
            if state.new_exp:
                create_image(state,i)
                #storing the image to be opened faster afterwards
                state.img[i]=Image.open('temp_wl_'+str(i)+'.png')
            else:
                if (not state.tempcoeff_seg==state.coeff_seg) and (i==state.wl_seg):
                    create_image(state,i)
                    state.tempcoeff_seg=state.coeff_seg
                    #storing the image to be opened faster afterwards
                    state.img[i]=Image.open('temp_wl_'+str(i)+'.png')
                if (not state.tempcoeff_act==state.coeff_act) and (i==state.wl_act) and (not state.draw):
                    create_image(state,i)
                    state.tempcoeff_act=state.coeff_act
                    #storing the image to be opened faster afterwards
                    state.img[i]=Image.open('temp_wl_'+str(i)+'.png')         
            
            
            #displaying intermediate columns
            with col[int(i/2)]:
                if state.draw and (i==state.wl_act):
                    make_canvas(state,i)
                else:
                    st.write(state.exp.wl[i].name)
                    st.image(state.img[i], use_column_width=True)
                    
# =============================================================================
#         #last column
#         with col[-1]:
#                last_col(state)
#         if st.checkbox('Show table'):     
#             st.write(pd.read_pickle('./pdresults.pkl'))    
# =============================================================================


def create_image(state,i):


    img1=np.array(state.exp.get_first_image(i,state.pos))
    img2=np.array(state.exp.get_last_image(i,state.pos))
        
    if (state.wl_seg==i) or (state.wl_act==i):
        coeff=(state.wl_seg==i)*state.coeff_seg+(state.wl_act==i)*state.coeff_act
        #threshold
        filtered1=filters.median(img1.astype(np.uint8))
        filtered2=filters.median(img2.astype(np.uint8))
        thresh1 = filters.threshold_otsu(filtered1)
        thresh2 = filters.threshold_otsu(filtered2)
        #if not st.checkbox("Last frame",key=i):
        #find mask and contour        
        mask1, contour1=segment_threshold(filtered1,coeff*thresh1)
        mask2, contour2=segment_threshold(filtered2,coeff*thresh2)
        fig1=images_with_seg(img1,img2,contour1,contour2)
        plt.tight_layout()

    else:
        fig1 = plt.figure()
        plt.subplot(1,2,1)
        if state.isrgn:
            a=plt.imshow(np.multiply(img1,1-state.rgn_contour),cmap='gray')
        else:
            a=plt.imshow(img1,cmap='gray')
        a.axes.axis('off')
        plt.subplot(1,2,2)
        if state.isrgn:
            b=plt.imshow(np.multiply(img2,1-state.rgn_contour),cmap='gray')
        else:
            b=plt.imshow(img2,cmap='gray')
        b.axes.axis('off')
        plt.tight_layout()
    fig1.savefig('temp_wl_'+str(i)+'.png',bbox_inches="tight")

def page_results(state):
    plot_values(state)        
    
    co=list(st.beta_columns(2))
    with open('./results.pkl', 'rb') as output:
        results=pickle.load(output)
    results_string=[result.exp.name+' : pos '+str(result.pos) for result in results]
    exp_to_modif=co[0].selectbox('Experiment to delete',range(len(results_string)),format_func=lambda i:results_string[i])
    if co[1].button('Delete this experiment'):
        with open('./results.pkl', 'wb') as output:
            results.pop(exp_to_modif)
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)                    

    filenametosave=st.text_input('Filename')
    if st.button('Save datas'):
        with open('./results.pkl', 'rb') as output:
            results=pickle.load(output)
        with open(state.file_dir+filenametosave+'_obj.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)    
    
    co=list(st.beta_columns(2))
    try:
        file_to_upload=file_selector(state.file_dir,extension='.pkl')
    except:
        file_to_upload=None
        "No such directory or no .nd file in this directory"
    if file_to_upload is not None:
        if st.button('Load these results'):
            with open(file_to_upload, 'rb') as output:
                results=pickle.load(output)           
            with open('results.pkl', 'wb') as output:
                pickle.dump(results, output, pickle.HIGHEST_PROTOCOL) 

# =============================================================================
# def last_col(state):
#     inds=range(len(state.exp.wl))
#     state.wls_meas=st.multiselect('Measurement channel',inds,format_func=lambda i: state.exp.wl[i].name,key='meas')
#     state.prot=st.checkbox('Makes protrusions?')
#     #st.write(state.wl_meas)
#     if st.button("Compute and save results"):
#         #st.write(int(state.exp.nbtime/state.exp.wl[wl].step)*state.exp.wl[wl].step)
#         threading.Thread(target=save_results,args=[state]).start()    
#     
#     state.crop=st.checkbox('Crop')
#     if st.button('Make movies'):
#         th=threading.Thread(target=make_all_movies,args=[state]) 
#         th.start()
# =============================================================================

    
def save_results(state):
    exp=copy.deepcopy(state.exp)
    if state.draw:
        mask_act=state.mask_act
    wls_meas=state.wls_meas
    prot=state.prot
    pos=state.pos
    stepseg=exp.wl[state.wl_seg].step
    stepact=exp.wl[state.wl_act].step
    pos, coeff_seg, coeff_act, wl_seg, wl_act, resultsfile=state.pos, state.coeff_seg, state.coeff_act, state.wl_seg, state.wl_act,state.resultsfile
    for wl_meas in wls_meas:
        whole=[]
        act=[]
        notact=[]
        result=Result(exp,prot,wl_meas,pos)
        stepmeas=exp.wl[wl_meas].step
        nb_img=int(exp.nbtime/stepmeas)
        
        for i in range(nb_img):
            #define the good frame to take for each segmentation or activation
            timg=i*stepmeas+1
            tseg=int(i*stepmeas/stepseg)*stepseg+1
            tact=int(i*stepmeas/stepact)*stepact+1
            try:
                img=np.array(Image.open(exp.get_image_name(wl_meas,pos,timg)))
                mask_seg=calculate_segmentation(exp,coeff_seg,wl_seg,pos,tseg)
                if state.draw:
                    mask_act=np.array(Image.fromarray(exp.mask_act).resize((img.shape[0],img.shape[1])))>0
                elif state.isrgn:
                    mask_act=state.rgn
                else:
                    mask_act=calculate_segmentation(exp,coeff_act,wl_act,pos,tact)
                whole_int=np.sum(img[mask_seg>0])
                whole_surf=np.sum(mask_seg>0)
                act_int=np.sum(img[(mask_act>0)*(mask_seg>0)])
                act_surf=np.sum((mask_act>0)*(mask_seg>0))
                whole.append(whole_int/whole_surf)
                act.append(act_int/act_surf)
                notact.append((whole_int-act_int)/(whole_surf-act_surf))
            except:
                whole.append(0)
                act.append(0)
                notact.append(0)
        result.whole, result.act, result.notact=whole,act,notact    
        npwhole=np.array(whole)/whole[0]
        npact=np.array(act)/act[0]
        npnotact=np.array(notact)/notact[0]
        #put into pd dataframe
        current_resultpd=pd.DataFrame(np.array([npwhole,npact,npnotact]).transpose())
        resultspd=pd.read_pickle('./pdresults.pkl')
        new_resultspd=pd.concat([resultspd,current_resultpd], ignore_index=True, axis=1)
        new_resultspd.to_pickle('./pdresults.pkl')
        
        #add to the list of results
        with open('./results.pkl', 'rb') as output:
            results=pickle.load(output)
        results.append(result)
        with open('./results.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)            
    return

def make_all_movies(state):
    exp=copy.deepcopy(state.exp)
    if exp.nbpos==1:
        posstring=''
    else:
        posstring='_s'+str(state.pos)
    crop=state.crop
    for i in range(exp.nbwl):
        make_movie(exp.name+'_w'+str(i+1)+exp.wl[i].name+posstring+'_t','.tif',exp.nbtime,max([wl.step for wl in exp.wl]),crop)

def make_movie(file1,file2,nbimg,step,crop):
    #initial step to find size
    img=Image.open(file1+str(1+2*step)+file2)
    size = img.size
    if crop:
        size=(1024,1024)
        start=512
        end=1537
        imgarray=np.array(img)[start:end,start:end]
    else:
        imgarray=np.array(img)
    sort=np.sort(imgarray.flatten())
    #sortcut=sort[int(len(sort)/1000):int(999*len(sort)/1000)]
    maxi,mini=np.max(sort),np.min(sort)
    
    out = cv2.VideoWriter(file1+'all.avi',0,7, size)
    for i in range(1,nbimg+1,step):
        if crop:
            img=np.array(Image.open(file1+str(i)+file2))[start:end,start:end]
        else:
            img=np.array(Image.open(file1+str(i)+file2))
        Image.fromarray(improve_contrast(img,mini,maxi)).save('./temp.png')
        img=cv2.imread('./temp.png')
        out.write(img)
    

def improve_contrast(img,mini,maxi):
    eq=255*(img.astype('float')-mini)/(maxi-mini)
    eq[eq<0]=0
    eq[eq>255]=255
    return np.uint8(eq)

def calculate_segmentation(exp,coeff,wl_ind,pos,t):
    img=np.array(Image.open(exp.get_image_name(wl_ind,pos,t)))
    filtered=filters.median(img.astype(np.uint8))
    thresh = filters.threshold_otsu(filtered)
    mask, contours=segment_threshold(filtered,coeff*thresh)
        
    return mask

def load_image(filename):
    image = cv2.imread(filename).astype(np.uint8)
    return image

def file_selector(folder_path='.',extension='.nd'):
    filenames = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)            

def plot_values(state):
    
    with open('./results.pkl', 'rb') as output:
        results=pickle.load(output)
    
    co=list(st.beta_columns(2))
    inds=range(len(state.exp.wl))
    chan=co[0].selectbox('Channels to plot',inds,format_func=lambda i: state.exp.wl[i].name,key='toplot')
    zones=co[1].multiselect('Zones to plot',['act','notact','whole'])
    pro=['Retracting','Protruding']
    prot=co[1].selectbox('Protruding or retracting',[0,1],format_func=lambda i:pro[i]) 

    results_string=[result.exp.name.split('/')[-1]+' : pos '+str(result.pos) for result in results if result.wl_ind==chan and result.prot==prot]
    results_name=[result.exp.name+str(result.pos) for result in results if result.wl_ind==chan and result.prot==prot]
    expe=co[0].multiselect('Experiments not to plot',range(len(results_string)),format_func=lambda i:results_string[i])
    expe_name=[results_name[i] for i in expe]
    
    if st.checkbox('Plot') and len(zones)>0:    
        st.write(len(results))
        res=Result_array([result for result in results if not (result.exp.name+str(result.pos) in expe_name)])
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)     
        ax.spines["bottom"].set_visible(True)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(True)
        colors=['blue','red','green']
        for i in range(len(zones)):
            plot_options={"color":colors[i]}
            res.plot(zone=zones[i],wl_ind=chan,prot=prot,plot_options=plot_options)
        co[0].pyplot(fig)
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)    
        ax.spines["bottom"].set_visible(True)    
        ax.spines["right"].set_visible(False)    
        ax.spines["left"].set_visible(True) 
        for i in range(len(zones)):
            plot_options={"color":colors[i]}
            res.plot_mean(zone=zones[i],wl_ind=chan,prot=prot,plot_options=plot_options)
        co[1].pyplot(fig)
        
        X,Y=res.xy2plot(zone=zones[i],wl_ind=chan,prot=prot,plot_options=plot_options)
        plotly_fig=px.scatter(X,Y)
        st.plotly_chart(plotly_fig)
        

def make_canvas(state,i):
    img=np.array(state.exp.get_last_image(i,state.pos))
    st.write(img.shape)
    size_fig=(img.shape[0]/5,img.shape[1]/5)
    fig1 = plt.figure(figsize=size_fig, dpi=1)
    b=plt.imshow(img,cmap='gray')
    b.axes.axis('off')
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig1.savefig('canvas.png',bbox_inches="tight",pad_inches = 0)
    bg_image = Image.open('canvas.png')
    size_img=size_fig
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=0,
        stroke_color="#000000",
        background_color="",
        background_image=bg_image,
        update_streamlit=True,
        height=size_img[1],
        width=size_img[0],
        drawing_mode="rect",
        key="canvas",
    )
    state.exp.mask_act=np.mean(canvas_result.image_data,axis=2)>0
    plt.imshow(state.exp.mask_act,cmap='gray')
    fig1.savefig('mask.png',bbox_inches="tight",pad_inches = 0)
    
if __name__ == "__main__":
   main()
   