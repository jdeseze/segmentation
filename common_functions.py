# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 12:36:33 2020

@author: Jean
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from skimage.transform import resize
from skimage import measure
from scipy import ndimage
from skimage import filters
import exifread
from scipy.interpolate import interp1d
import math
import plotly.express as px
import plotly.graph_objects as go

import napari
from glob import glob
from skimage.io import imread
from skimage.io.collection import alphanumeric_key
from dask import delayed
import dask.array as da


class WL:
    def __init__(self,name,step=1):
        self.name=name
        self.step=step

class Exp:

    def __init__(self,expname,wl=[],nbpos=1,nbtime=1,comments=[]):
        self.name=expname
        self.nbpos=nbpos
        self.nbtime=nbtime
        self.wl=wl
        self.nbwl=len(wl)
        self.commments=comments
        if self.nbtime==1:
            self.timestep=0
        else:
            maxwl_ind=min(list(range(self.nbwl)), key=lambda ind:self.wl[ind].step)
            try:
                open(self.get_image_name(maxwl_ind,timepoint=1), 'rb')
                self.stacks=False
            except:
                self.stacks=True
            if self.stacks:
                #print(self.get_stack_name(maxwl_ind))
                self.timestep=10
            else:
                with open(self.get_image_name(maxwl_ind,timepoint=1), 'rb') as opened:
                    tags = exifread.process_file(opened)
                    time_str=tags['Image DateTime'].values
                    h, m, s = time_str.split(' ')[1].split(':')
                    time1=int(h) * 3600 + int(m) * 60 + float(s)
                with open(self.get_image_name(maxwl_ind,timepoint=int((nbtime-1)/self.wl[maxwl_ind].step+1)), 'rb') as opened:
                    tags = exifread.process_file(opened)
                    time_str=tags['Image DateTime'].values
                    h, m, s = time_str.split(' ')[1].split(':')
                    time2=int(h) * 3600 + int(m) * 60 + float(s)
                    self.timestep=(time2-time1)/self.nbtime
    
    #use this if the stack was not build 
    def get_image_name(self,wl_ind,pos=1,timepoint=1,sub_folder=''):
        if self.nbtime==1:
            tpstring=''
        else:
            tpstring='_t'+str(timepoint)
        if self.nbpos==1:
            posstring=''
        else:
            posstring='_s'+str(pos)
        return '\\'.join(self.name.split('/')[0:-1]+[self.name.split('/')[-1]])+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+tpstring+'.tif'    
        #return self.name+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+tpstring+'.tif'
   
    #use this if there is only the stack, in the "Stacks" folder
    def get_stack_name(self,wl_ind,pos=1,sub_folder='Stacks'):
        if self.nbpos==1:
            return '\\'.join(self.name.split('\\')[0:-1]+[sub_folder]+[self.name.split('\\')[-1]])+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+'.tif'
        else:
            posstring='_s'+str(pos)
            return '\\'.join(self.name.split('\\')[0:-1]+[sub_folder]+[self.name.split('\\')[-1]])+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+'.tif'
    
    def get_first_image(self,wl_ind,pos=1,timepoint=''):
        timepoint=1
        if self.stacks:
            I=Image.open(self.get_stack_name(wl_ind,pos))
            I.seek(timepoint)
            return I
        else:
            return Image.open(self.get_image_name(wl_ind,pos,timepoint))
    
    def get_last_image(self,wl_ind,pos=1,timepoint=1):
        last_ind=int(self.nbtime/self.wl[wl_ind].step-1)*self.wl[wl_ind].step+1
        if self.stacks:
            I=Image.open(self.get_stack_name(wl_ind,pos))
            I.seek(timepoint)
            return I
        else:        
            return Image.open(self.get_image_name(wl_ind,pos,last_ind))
    
    def get_sizeimg(self):
        return self.get_first_image(0).size
    
    def disp_message(self):
        return self.get_stack_name(0)
    

class Result:
    def __init__(self, exp,prot,wl_ind,pos,startacq=0,act=[],notact=[],whole=[],background=0):
        self.exp=exp
        self.prot=prot
        self.wl_ind=wl_ind
        self.act=act
        self.notact=notact
        self.whole=whole
        self.channel=self.exp.wl[wl_ind]
        self.pos=pos
        self.startacq=startacq
        self.background=background
    
    def plot(self,zone='act',plot_options=None):
        toplot=self.get_zone(zone)#running_mean(self.get_zone(zone),4)
        toplot[toplot==0]=math.nan
        toplot=(np.array(toplot)-self.background)-(toplot[0]-self.background)
        if not plot_options:
            plot_options={}            
        x=(np.arange(toplot.size))*self.channel.step*self.exp.timestep/60
        plt.plot(x,toplot,**plot_options)
        return x,toplot#go.Scatter(x=x,y=toplot,mode='lines')
    
    def get_abs_val(self,zone='act'):
        toplot=np.array(self.get_zone(zone))
        toplot[toplot==0]=math.nan
        abs_value=np.mean(toplot[0])-self.background      
        return abs_value
    
    def xy2plot(self,zone='act',plot_options=None):
        toplot=self.get_zone(zone)#running_mean(self.get_zone(zone),4)
        toplot[toplot==0]=math.nan
        toplot=(toplot)#-self.background)-(np.mean(toplot[0])-self.background)
        if not plot_options:
            plot_options={}            
        x=(np.arange(toplot.size))*self.channel.step*self.exp.timestep/60
        return x,toplot#go.Scatter(x=x,y=toplot,mode='lines')   
    
    def get_zone(self,zone):
        if zone=='act':
            return np.array(self.act)
        if zone=='notact':
            return np.array(self.notact)
        if zone=='whole':
            return np.array(self.whole)
    
    def name(self):
        return self.exp.name.split('\\')[-1]+' : pos '+str(self.pos)
    
        
class Result_array(list):
    def __init__(self,data):
        list.__init__(self,data)
    
    def plot(self,zone='act',wl_name="TIRF 561",prot=True,plot_options={}):
        [result.plot(zone,plot_options) for result in self if result.channel.name==wl_name and result.prot==prot]    

    
    def xy2plot(self,zone='act',wl_name="TIRF 561",prot=True):
        toplot=[]
        zones=np.array(['act','notact','whole'])
        colors=np.array(['blue','red','green'])
        for res in self:
            if res.channel.name==wl_name and res.prot==prot:
                x,y=res.xy2plot(zone)            
                toplot.append(go.Scatter(x=x,y=y,mode='lines',line_color=colors[zones==zone][0],name=res.name()))
        return toplot
    
    def plot_mean(self,zone='act',wl_name="TIRF 561",prot=True,plot_options={}):
        #time step should be in minutes

        t_start=0
        t_end=min((len(result.get_zone(zone))-1)*result.exp.timestep for result in self if result.channel.name==wl_name)
        nbsteps=min(len(result.get_zone(zone)) for result in self)
        interp=[]
        for result in self: 
            if result.channel.name==wl_name and (not math.isnan(np.sum(result.get_zone(zone)))) and result.prot==prot:
                values=result.get_zone(zone)
                tstep=result.exp.timestep
                normvals=(np.array(result.get_zone(zone))-result.background)/(np.mean(np.array(result.get_zone(zone))[0])-result.background)
                lasttime=len(normvals)*tstep-0.001
                times=np.arange(0,lasttime,tstep)
                
                if sum(np.array(values)==0)>0:
                    f_endtemp=list(values).index(next(filter(lambda x: x==0, values)))
                    normvals=normvals[0:f_endtemp]
                    times=np.arange(0,(f_endtemp)*result.exp.timestep,tstep)
                    if f_endtemp*result.exp.timestep<t_end:
                        t_end=times[-1]

                interp.append(interp1d(times,normvals))
                
        x=np.arange(t_start,t_end,int((t_end-t_start)/nbsteps))
        y=np.vstack([f(x) for f in interp])
        
        ym=np.average(y, axis=0)
        sigma=np.std(y,axis=0)
        
        yh=ym+sigma/(y.shape[0]**0.5)
        yb=ym-sigma/(y.shape[0]**0.5)
        
        #clear_plot(size)
        
        plt.plot(x/60,ym,linewidth=2,**plot_options)

        plt.plot(x/60,yh,linewidth=0.05,**plot_options)
        plt.plot(x/60,yb,linewidth=0.05,**plot_options)
        plt.fill_between(x/60,yh,yb,alpha=0.2,**plot_options)

def image_with_seg(img1,contour):
    fig1,ax = plt.subplot()
    
    #original image
    a=plt.imshow(img1,cmap='gray')
    #find mask and contour
    if contour.any():        
        a.axes.plot(list(map(int,contour[:, 1])), list(map(int,contour[:, 0])), linewidth=2)
    a.axes.axis('off')
    return fig1,ax

def images_with_seg(img1,img2,contour1,contour2):
    fig1 = plt.figure()
    #original image
    #first image
    plt.subplot(1,2,1)
    a=plt.imshow(img1,cmap='gray')
    #find mask and contour
    if contour1.any():        
        a.axes.plot(list(map(int,contour1[:, 1])), list(map(int,contour1[:, 0])), linewidth=2)
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    #second image
    plt.subplot(1,2,2)
    b=plt.imshow(img2,cmap='gray')
    if contour2.any():        
        b.axes.plot(list(map(int,contour2[:, 1])), list(map(int,contour2[:, 0])), linewidth=2)
    b.axes.get_xaxis().set_visible(False)
    b.axes.get_yaxis().set_visible(False)
    return fig1

def get_exp(filename):
    nb_pos=1
    nb_wl=1
    with open(filename,'r') as file:
        i=0
        line=file.readline()
        comments=[]
        iscomments=False
        while not line.rstrip().split(', ')[0]=='"NTimePoints"' and i<50:
            if line.rstrip().split(', ')[0]=='"StartTime1"':
                iscomments=False
            if iscomments:
                comments.append(line.rstrip())
            if line.rstrip().split(', ')[0]=='"Description"':
                iscomments=True
                comments.append(str(line.rstrip().split(', ')[1]))
            line=file.readline()
            i+=1
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
        
        expname=filename.rstrip('d').rstrip('n').rstrip('.')
        
        return Exp(expname,wl,nb_pos,nb_tp,comments)

def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def segment_threshold(img,thresh):
    #img=(img/2^8).astype(np.uint8)
    binary = img > thresh
    dil=ndimage.binary_dilation(binary,iterations=2)
    filled=ndimage.binary_fill_holes(dil).astype(int)
    label_img, cc_num = ndimage.label(filled)
    CC = ndimage.find_objects(label_img)
    cc_areas = ndimage.sum(filled, label_img, range(cc_num+1))
    area_mask = (cc_areas < max(cc_areas))
    label_img[area_mask[label_img]] = 0
    contours = measure.find_contours(label_img, 0.8)
    if len(contours)>0:
        contour=contours[0]
    else:
        contour=np.array([None])
    return label_img>0, contour
    
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
    #for snake in contours:    
    ax.plot(contours[0][:, 0], contours[0][:, 1], '-b', lw=3);
    #return snake

def circle_points(resolution, center, radius):   

    s = np.linspace(0, 2*np.pi, resolution)
    r = center[0] + radius*np.sin(s)
    c = center[1] + radius*np.cos(s)

    return np.array([c, r]).T

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N


def view_in_napari(filename=r"F:\optorhoa\201208_RPE_optoRhoA_PAKiRFP\cell2s_50msact_1_w2TIRF 561_t*.tif"):
    filenames = sorted(glob(filename),key=alphanumeric_key)

    sample = imread(filenames[0])
    
    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    stack = da.stack(dask_arrays, axis=0)
    stack.shape  # (nfiles, nz, ny, nx)
    
    viewer=napari.view_image(stack, contrast_limits=[0,2000],name='561')  