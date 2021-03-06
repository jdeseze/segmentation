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
        if self.nbtime==1:
            self.timestep=0
        else:
            maxwl_ind=min(list(range(self.nbwl)), key=lambda ind:self.wl[ind].step)
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
    
    def get_image_name(self,wl_ind,pos=1,timepoint=1):
        if self.nbtime==1:
            tpstring=''
        else:
            tpstring='_t'+str(timepoint)
        if self.nbpos==1:
            posstring=''
        else:
            posstring='_s'+str(pos)
        return self.name+'_w'+str(wl_ind+1)+self.wl[wl_ind].name+posstring+tpstring+'.tif'
    
    def get_first_image(self,wl_ind,pos=1,timepoint=''):
        timepoint=1
        return Image.open(self.get_image_name(wl_ind,pos,timepoint))
    
    def get_last_image(self,wl_ind,pos=1,timepoint=1):
        last_ind=int(self.nbtime/self.wl[wl_ind].step-1)*self.wl[wl_ind].step+1
        return Image.open(self.get_image_name(wl_ind,pos,last_ind))
    
    def get_sizeimg(self):
        return self.get_first_image(0).size

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
        toplot=np.array(self.get_zone(zone))
        toplot[toplot==0]=math.nan
        toplot=(toplot-self.background)/(np.mean(toplot[0])-self.background)
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
        toplot=np.array(self.get_zone(zone))
        toplot[toplot==0]=math.nan
        toplot=(toplot-self.background)/(np.mean(toplot[0])-self.background)
        if not plot_options:
            plot_options={}            
        x=(np.arange(toplot.size))*self.channel.step*self.exp.timestep/60
        return x,toplot#go.Scatter(x=x,y=toplot,mode='lines')   
    
    def get_zone(self,zone):
        if zone=='act':
            return self.act
        if zone=='notact':
            return self.notact
        if zone=='whole':
            return self.whole
    
        
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
                toplot.append(go.Scatter(x=x,y=y,mode='lines',line_color=colors[zones==zone][0],name=str(res.exp.nbpos)))
        
        return toplot
    
    def plot_mean(self,zone='act',wl_name="TIRF 561",time_step=30,prot=True,plot_options={}):
        #time step should be in minutes

        t_start=0
        t_end=61#min(len(self[i].get_zone(zone)) for i in range(len(self)))
        for result in self:
            if result.channel.name==wl_name and (not math.isnan(np.sum(result.get_zone(zone)))) and result.prot==prot:
                values=result.get_zone(zone)
                if sum(np.array(values)==0)>0:
                    t_endtemp=values.index(next(filter(lambda x: x==0, values)))
                    if t_endtemp<t_end:
                        t_end=t_endtemp

        values=[(np.array(result.get_zone(zone))-result.background)/(np.mean(np.array(result.get_zone(zone))[0])-result.background) for result in self if result.channel.name==wl_name and (not math.isnan(np.sum(result.get_zone(zone)))) and result.prot==prot]
        x=np.arange(t_start,t_end)*time_step/60
        interp=[interp1d(x,yi[t_start:t_end]) for yi in values]
        y=np.vstack([f(x) for f in interp])
        
        ym=np.average(y, axis=0)
        sigma=np.std(y,axis=0)
        
        yh=ym+sigma/(y.shape[0]**0.5)
        yb=ym-sigma/(y.shape[0]**0.5)
        
        #clear_plot(size)
        
        plt.plot(x,ym,linewidth=2,**plot_options)

        plt.plot(x,yh,linewidth=0.05,**plot_options)
        plt.plot(x,yb,linewidth=0.05,**plot_options)
        plt.fill_between(x,yh,yb,alpha=0.2,**plot_options)

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
        
        expname=filename.rstrip('.nd')
        
        return Exp(expname,wl,nb_pos,nb_tp)

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

