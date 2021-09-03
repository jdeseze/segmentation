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
import threading
import pickle
import copy
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
import tkinter as tk
from tkinter import filedialog
import glob
import sys

def main():
    st.set_page_config(page_title="Segmentation", page_icon=":microscope:",layout="wide")
    
    pages = {
        "Make measures": page_measures,
        "Look at results": page_results,
        "Text file": page_text_file
    }
    
    with st.sidebar:
        
        page=st.selectbox('Choose page',tuple(pages.keys()))
        
        with st.expander('Choose experiment'):
            
            # Folder picker button
            root = tk.Tk()
            root.withdraw()
            root.wm_attributes('-topmost', 1)
            clicked = st.button('Please select a folder:')
            
            if clicked:
                st.session_state.file_dir = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
            try:
                st.session_state.filename=file_selector(st.session_state.file_dir)
            except:
                st.session_state.filename=None
                st.write("Choose a directory (or no .nd file in this directory)")
            try: 
                new_exp()
            except:
                st.write('Unable to load an experiment')
    
    if not 'resultsfile' in st.session_state:
        st.session_state.resultsfile='./pdresults.pkl'
        resultspd=pd.DataFrame()
        resultspd.to_pickle(st.session_state.resultsfile)
        with open('./results.pkl', 'wb') as output:
            results=[]
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
    
    if 'filename' in st.session_state:
        pages[page]()
    

    st.session_state.new_exp=False
               

def page_measures():
    with st.sidebar:
        
        if 'exp' in st.session_state:        
            with st.expander('Experiment'):
                st.write("Number of positions : "+str(st.session_state.exp.nbpos))
                st.write("Number of time steps : "+str(st.session_state.exp.nbtime))
                st.write("Time step : "+str(st.session_state.exp.timestep)+' sec')
                st.session_state.pos=st.selectbox('Position',range(1,st.session_state.exp.nbpos+1),on_change=new_exp)
            
            with st.expander('Segmentation and activation'):
                inds=range(st.session_state.exp.nbwl)
                
                st.session_state.wl_seg=st.selectbox('Segmentation channel',inds,format_func=lambda i: st.session_state.exp.wl[i].name,key='seg1')
                st.session_state.coeff_seg=st.slider('Threshold',0.5,1.5,1.0,0.01,key='seg2')
                
                seg_options=['Import region','Draw rectangle']
                st.session_state.def_rgn=st.selectbox('Activation region',range(2),format_func=lambda i: seg_options[i])
                if st.session_state.def_rgn==0:
                    st.session_state.draw=0
                    try:
                        rgn_file=file_selector(st.session_state.file_dir,extension='.rgn')
                    except:
                        pass
                    if st.button('Load region'):
                        with open(rgn_file) as file:
                            line=file.readline().rstrip().split(', ')
                            x,y=int(line[2].split(' ')[1]),int(line[2].split(' ')[2])
                            w,l=int(line[6].split(' ')[2]),int(line[6].split(' ')[3])
                            size_img=st.session_state.exp.get_sizeimg()
                            mask=np.zeros((size_img[1],size_img[0]))
                            mask[y:y+l,x:x+w]=1
                            st.session_state.rgn=mask
                            contour=np.zeros((size_img[1],size_img[0]))
                            contour[y:y+l,x]=1
                            contour[y:y+l,x+w]=1
                            contour[y,x:x+w]=1
                            contour[y+l,x:x+w]=1
                            st.session_state.rgn_contour=contour
                            st.session_state.isrgn=True
                            st.session_state.new_exp=True
                    else: 
                        st.session_state.isrgn=False
                if st.session_state.def_rgn==1:
                    st.session_state.draw=1
                    st.session_state.isrgn=0
            
            with st.expander('Measures and movie'):
                inds=range(len(st.session_state.exp.wl))
                st.session_state.wls_meas=st.multiselect('Measurement channel',inds,format_func=lambda i: st.session_state.exp.wl[i].name,key='meas')
                st.session_state.prot=st.checkbox('Makes protrusions?')
                #st.write(st.session_state.wl_meas)
                if st.button("Compute and save results"):
                    #st.write(int(st.session_state.exp.nbtime/st.session_state.exp.wl[wl].step)*st.session_state.exp.wl[wl].step)
                    #threading.Thread(target=compute).start() 
                    compute()
                

# =============================================================================
#         debug=st.sidebar.beta_expander("Debug",expanded=False)
#         with debug:
#             st.session_state.frame=st.selectbox('Time Frame', range(1,exp.nbtime+1),key='selectbox')
# =============================================================================
            
    if ('pos' in st.session_state) and  ((not 'temppos' in st.session_state) or (not st.session_state.temppos==st.session_state.pos) or (not st.session_state.exp.name==st.session_state.tempexpname)):
        st.session_state.new_exp=True
        st.session_state.temppos=st.session_state.pos
        st.session_state.tempexpname=st.session_state.exp.name
        st.session_state.rgn=None
    
    #decide how many columns should be done       
    try:
        col=list(st.columns(int((len(st.session_state.exp.wl)+1)/2)))
        st.session_state.wlthres=[None]*4
        st.session_state.fig=[None]*4
        st.session_state.img=[None]*4
    except:
        st.write('Please load a valid experiment first')
        col=None
    
    #Create columns only if experiment is loaded
    if col is not None:
        
        for i in range(len(st.session_state.exp.wl)):
                
            #checking if I should update the temporary image
            
            if st.session_state.new_exp:
                create_image(i)
                #storing the image to be opened faster afterwards
                st.session_state.img[i]=Image.open('temp_wl_'+str(i)+'.png')
            else:
                if (('tempcoeff_seg' not in st.session_state) or (not st.session_state.tempcoeff_seg==st.session_state.coeff_seg)) and (i==st.session_state.wl_seg):
                    create_image(i)
                    st.session_state.tempcoeff_seg=st.session_state.coeff_seg
                    #storing the image to be opened faster afterwards
                st.session_state.img[i]=Image.open('temp_wl_'+str(i)+'.png')
# =============================================================================
#                 if (not st.session_state.tempcoeff_act==st.session_state.coeff_act) and (i==st.session_state.wl_act) and (not st.session_state.draw):
#                     create_image(st.session_state,i)
#                     st.session_state.tempcoeff_act=st.session_state.coeff_act
#                     #storing the image to be opened faster afterwards
#                     st.session_state.img[i]=Image.open('temp_wl_'+str(i)+'.png')         
# =============================================================================
            
            
            #displaying intermediate columns
            with col[int(i/2)]:
                if st.session_state.draw and (i==3):
                    make_canvas(i)
                else:
                    st.write(st.session_state.exp.wl[i].name)
                    st.image(st.session_state.img[i], use_column_width=True)


def new_exp():
    try:
        st.session_state['exp']=get_exp(st.session_state.filename)
    except:
        st.write('unable to load experiment')
                
def page_results():
    
    try:
        file_to_upload=file_selector(st.session_state.file_dir,extension='.pkl')
    except:
        file_to_upload=None
        "No such directory or no .pkl file in this directory"
    if file_to_upload is not None:
        if st.button('Load these results'):
            with open(file_to_upload, 'rb') as output:
                results=pickle.load(output)           
            with open('./results.pkl', 'wb') as output:
                pickle.dump(copy.deepcopy(results), output, pickle.HIGHEST_PROTOCOL) 
            with open('./results.txt', 'w') as output:
                write_on_text_file(results,output)   
    
    plot_values()        
    
    co=list(st.columns(2))
    with open('./results.pkl', 'rb') as output:
        results=pickle.load(output)
    results_string=[result.exp.name.split('\\')[-1]+' : pos '+str(result.pos) for result in results]
    exp_to_modif=co[0].selectbox('Experiment to delete',range(len(results_string)),format_func=lambda i:results_string[i])
    if co[1].button('Delete this experiment'):
        with open('./results.pkl', 'wb') as output:
            results.pop(exp_to_modif)
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)                    

    filenametosave=st.text_input('Filename')
    st.write(st.session_state.file_dir.replace("\\","/"))
    if st.button('Save datas'):
        with open('./results.pkl', 'rb') as output:
            results=pickle.load(output)
        with open(st.session_state.file_dir.replace("\\","/")+'/'+filenametosave+'_obj.pkl', 'wb') as output:
            pickle.dump(copy.deepcopy(results), output, pickle.HIGHEST_PROTOCOL)    
        with open(st.session_state.file_dir.replace("\\","/")+'/'+filenametosave+'.txt', 'w') as output:
            write_on_text_file(results,output)   

def page_text_file():
        with open('./results.txt', 'r') as output:
            [st.write(line) for line in output.readlines()]

def create_image(i):

    img1=np.array(st.session_state.exp.get_first_image(i,st.session_state.pos))
    img2=np.array(st.session_state.exp.get_last_image(i,st.session_state.pos))
    plt.style.use('dark_background')    
    if (st.session_state.wl_seg==i):
        coeff=st.session_state.coeff_seg

        #threshold
        filtered1=filters.median(img1)
        filtered2=filters.median(img2)
        thresh1 = filters.threshold_otsu(filtered1)
        thresh2 = filters.threshold_otsu(filtered2)
        #if not st.checkbox("Last frame",key=i):
        #find mask and contour        
        mask1, contour1=segment_threshold(filtered1,coeff*thresh1)
        mask2, contour2=segment_threshold(filtered2,coeff*thresh1)
        fig1=images_with_seg(img1,img2,contour1,contour2)
        plt.tight_layout()

    else:
        fig1 = plt.figure()
        plt.subplot(1,2,1)
        if st.session_state.isrgn:
            a=plt.imshow(np.multiply(img1,1-st.session_state.rgn_contour),cmap='gray')
        else:
            a=plt.imshow(img1,cmap='gray')
        a.axes.axis('off')
        plt.subplot(1,2,2)
        if st.session_state.isrgn:
            b=plt.imshow(np.multiply(img2,1-st.session_state.rgn_contour),cmap='gray')
        else:
            b=plt.imshow(img2,cmap='gray')
        b.axes.axis('off')
        plt.tight_layout()
    
    fig1.savefig('temp_wl_'+str(i)+'.png',bbox_inches="tight")



# =============================================================================
# def last_col(st.session_state):
#     inds=range(len(st.session_state.exp.wl))
#     st.session_state.wls_meas=st.multiselect('Measurement channel',inds,format_func=lambda i: st.session_state.exp.wl[i].name,key='meas')
#     st.session_state.prot=st.checkbox('Makes protrusions?')
#     #st.write(st.session_state.wl_meas)
#     if st.button("Compute and save results"):
#         #st.write(int(st.session_state.exp.nbtime/st.session_state.exp.wl[wl].step)*st.session_state.exp.wl[wl].step)
#         threading.Thread(target=save_results,args=[st.session_state]).start()    
#     
#     st.session_state.crop=st.checkbox('Crop')
#     if st.button('Make movies'):
#         th=threading.Thread(target=make_all_movies,args=[st.session_state]) 
#         th.start()
# =============================================================================

    
def compute():
    exp=get_exp(st.session_state.filename)
    wls_meas=st.session_state.wls_meas
    prot=st.session_state.prot
    pos=st.session_state.pos
    coeff=st.session_state.coeff_seg
    stepseg=exp.wl[st.session_state.wl_seg].step
    if st.session_state.draw:
        mask_act=np.array(Image.fromarray(st.session_state.mask_act).resize((img.shape[0],img.shape[1])))>0
    else:
        mask_act=copy.deepcopy(st.session_state.rgn)
        
    pos, coeff_seg,  wl_seg=st.session_state.pos, st.session_state.coeff_seg,  st.session_state.wl_seg
    for wl_meas in wls_meas:
        whole=[]
        act=[]  
        notact=[]
        result=Result(exp,prot,wl_meas,pos)
        stepmeas=exp.wl[wl_meas].step
        nb_img=int(exp.nbtime/stepmeas)
        img4thresh=np.array(Image.open(exp.get_image_name(wl_seg,pos,1)))
        filtered4thresh=filters.median(img4thresh)
        thresh = filters.threshold_otsu(filtered4thresh)
        
        for i in range(nb_img):
            #define the good frame to take for each segmentation or activation
            timg=i*stepmeas+1
            tseg=int(i*stepmeas/stepseg)*stepseg+1
            #calculate mask of segmentation
            img2seg=np.array(Image.open(exp.get_image_name(wl_seg,pos,tseg)))
            filtered=filters.median(img2seg)
            mask_seg, contours=segment_threshold(filtered,coeff*thresh)  
            
            #take values in current image
            img=np.array(Image.open(exp.get_image_name(wl_meas,pos,timg)))
            whole_int=np.sum(img[mask_seg>0])
            whole_surf=np.sum(mask_seg>0)
            act_int=np.sum(img[(mask_act>0)*(mask_seg>0)])
            act_surf=np.sum((mask_act>0)*(mask_seg>0))
            #print(np.sum((mask_act>0)*(mask_seg>0)))
            if i==0:
                background=np.mean(img[0:20,0:20])
            whole.append(whole_int/whole_surf)
            notact.append((whole_int-act_int)/(whole_surf-act_surf))
            if act_surf==0:
                act.append(0)
            else:
                act.append(act_int/act_surf)
        result.whole, result.act, result.notact, result.background =whole,act,notact,background  
        npwhole=(np.array(whole)-background)/whole[0]
        npact=(np.array(act)-background)/act[0]
        npnotact=(np.array(notact)-background)/notact[0]
        #put into pd dataframe
        current_resultpd=pd.DataFrame(np.array([npwhole,npact,npnotact]).transpose())
        resultspd=pd.read_pickle('./pdresults.pkl')
        new_resultspd=pd.concat([resultspd,current_resultpd], ignore_index=True, axis=1)
        new_resultspd.to_pickle('./pdresults.pkl')
        print("done")
        #add to the list of results
        with open('./results.pkl', 'rb') as output:
            results=pickle.load(output)
        results.append(result)
        with open('./results.pkl', 'wb') as output:
            pickle.dump(results, output, pickle.HIGHEST_PROTOCOL)
        with open('./results.txt','w') as output:
            write_on_text_file(results,output)
    return

def write_on_text_file(results,output):
    output.write('Number of datas : '+str(len(results))+' \n')
    output.write('\n')
    for result in results:
        output.write('Experiment: '+str(result.exp.name)+' \n')
        output.write('Position: '+str(result.pos)+' \n')
        output.write('Channel: '+str(result.channel.name)+' \n')
        output.write('Background value: '+str(result.background)+' \n')
        output.write('Activated zone: '+str(result.act)+' \n')
        output.write('Not activated zone: '+str(result.act)+' \n')
        output.write('\n')
        
    

def load_image(filename):
    image = cv2.imread(filename).astype(np.uint8)
    return image

def file_selector(folder_path='.',extension='.nd'):
    filenames = [f for f in os.listdir(folder_path) if f.endswith(extension)]
    selected_filename = st.selectbox('Select a file', filenames,on_change=new_exp)
    return os.path.join(folder_path, selected_filename)            

def plot_values():
    
    with open('./results.pkl', 'rb') as output:
        results=pickle.load(output)
    
    co=list(st.columns(2))
    chans=[st.session_state.exp.wl[i].name for i in range(len(st.session_state.exp.wl))]
    chan=co[0].selectbox('Channels to plot',chans,key='toplot')
    zones=co[1].multiselect('Zones to plot',['act','notact','whole'])
# =============================================================================
#     pro=['Retracting','Protruding']
#     prot=co[1].selectbox('Protruding or retracting',[0,1],format_func=lambda i:pro[i]) 
# =============================================================================

    results_string=[result.exp.name.split('\\')[-1]+' : pos '+str(result.pos) for result in results if result.channel.name==chan]
    results_name=[result.exp.name+str(result.pos) for result in results if result.channel.name==chan]
    #expe=co[0].multiselect('Experiments not to plot',range(len(results_string)),format_func=lambda i:results_string[i])
    #expe_name=[results_name[i] for i in expe]
    
    if st.checkbox('Plot') and len(zones)>0:    
        st.write(len(results))
        res=Result_array([result for result in results])# if not (result.exp.name+str(result.pos) in expe_name)])
        plt.style.use('dark_background')
        for prot in [0,1]:
            fig, ax = plt.subplots()
            ax.spines["top"].set_visible(False)     
            ax.spines["bottom"].set_visible(True)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(True) 
            fig.set_facecolor('black')
            ax.set_facecolor('black')
            colors=['blue','red','green']
            for i in range(len(zones)):
                plot_options={"color":colors[i]}
                res.plot(zone=zones[i],wl_name=chan,prot=prot,plot_options=plot_options)
            co[0].pyplot(fig)
            fig, ax = plt.subplots()
            ax.spines["top"].set_visible(False)    
            ax.spines["bottom"].set_visible(True)    
            ax.spines["right"].set_visible(False)    
            ax.spines["left"].set_visible(True) 
            for i in range(len(zones)):
                try:
                    st.write(zones[i])
                    plot_options={"color":colors[i]}
                    res.plot_mean(zone=zones[i],wl_name=chan,prot=prot,plot_options=plot_options)
                except:
                    st.write('cannot plot the means: no datas? problem in the datas?')
                    st.write('theoretical number of data :'+ str(len(res)))
                    st.write("Unexpected error:", sys.exc_info()[0])
            co[1].pyplot(fig)
        
            plotly_fig=go.Figure()
            for i in range(len(zones)):
                toplot=res.xy2plot(zone=zones[i],wl_name=chan,prot=prot)
                for trace in toplot:
                    plotly_fig.add_trace(trace)
            plotly_fig.update_layout(plot_bgcolor='rgb(255,255,255)',legend_itemclick='toggle')
            plotly_fig.update_xaxes(showgrid=True,visible=True,color='rgb(0,0,0)')
            plotly_fig.update_yaxes(showgrid=False,visible=True,color='rgb(0,0,0)')
            st.plotly_chart(plotly_fig)
        

def make_canvas(i):
    img=np.array(st.session_state.exp.get_last_image(i,st.session_state.pos))
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
    st.session_state.exp.mask_act=np.mean(canvas_result.image_data,axis=2)>0
    plt.imshow(st.session_state.exp.mask_act,cmap='gray')
    fig1.savefig('mask.png',bbox_inches="tight",pad_inches = 0)
    

if __name__ == "__main__":
   main()
   
   
   
   



# =============================================================================
# 
# def make_all_movies():
#     exp=copy.deepcopy(st.session_state.exp)
#     if exp.nbpos==1:
#         posstring=''
#     else:
#         posstring='_s'+str(st.session_state.pos)
#     crop=st.session_state.crop
#     for i in range(exp.nbwl):
#         make_movie(exp.name+'_w'+str(i+1)+exp.wl[i].name+posstring+'_t','.tif',exp.nbtime,max([wl.step for wl in exp.wl]),crop)
# 
# def make_movie(file1,file2,nbimg,step,crop):
#     #initial step to find size
#     img=Image.open(file1+str(1+2*step)+file2)
#     size = img.size
#     if crop:
#         size=(1024,1024)
#         start=512
#         end=1537
#         imgarray=np.array(img)[start:end,start:end]
#     else:
#         imgarray=np.array(img)
#     sort=np.sort(imgarray.flatten())
#     #sortcut=sort[int(len(sort)/1000):int(999*len(sort)/1000)]
#     maxi,mini=np.max(sort),np.min(sort)
#     
#     out = cv2.VideoWriter(file1+'all.avi',0,7, size)
#     for i in range(1,nbimg+1,step):
#         if crop:
#             img=np.array(Image.open(file1+str(i)+file2))[start:end,start:end]
#         else:
#             img=np.array(Image.open(file1+str(i)+file2))
#         Image.fromarray(improve_contrast(img,mini,maxi)).save('./temp.png')
#         img=cv2.imread('./temp.png')
#         out.write(img)
#     
# 
# def improve_contrast(img,mini,maxi):
#     eq=255*(img.astype('float')-mini)/(maxi-mini)
#     eq[eq<0]=0
#     eq[eq>255]=255
#     return np.uint8(eq)
# =============================================================================
