# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 11:49:36 2017
Compared to v2, this version changes the algorithm of distance between neures.
@author: Zou_S.L
"""
import tensorflow as tf
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
from jdcal import jd2gcal,MJD_0
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.basemap import Basemap
import math,colorsys
import matplotlib.lines as mlines
from scipy.interpolate import interp1d



def light2dark(RGB,num):
      HLS=colorsys.rgb_to_hls(RGB[0],RGB[1],RGB[2])
      hlslist=np.zeros((num,3))
      hlslist[:,0]=HLS[0]
      hlslist[:,2]=HLS[2]
      hlslist[:,1]=np.linspace(0.1,0.9,num)
      rgblist=[]
      for i,val in enumerate(hlslist):
            rgblist.append(colorsys.hls_to_rgb(hlslist[i][0],hlslist[i][1],\
                                               hlslist[i][2]))
      return rgblist

      
# -----------load data 
nc_obj=nc.Dataset("D:\TCproject\Allstorms.ibtracs_wmo.v03r09.nc")
lat_wmo=nc_obj.variables['lat_wmo'][:]
lon_wmo=nc_obj.variables['lon_wmo'][:]
pres_wmo=nc_obj.variables['pres_wmo'][:]
time_wmo=nc_obj.variables['time_wmo'][:]
numObs=nc_obj.variables['numObs'][:]
wind_wmo=nc_obj.variables['wind_wmo'][:]
nature_wmo=nc_obj.variables['nature_wmo'][:]
basin=nc_obj.variables['basin'][:]
pres_wmo[(pres_wmo<=0)]=ma.masked
lon_wmo[lon_wmo<0]=lon_wmo[lon_wmo<0]+360

# -----------choosing data
pacific=1
normalization=0

weird_ind=[4552,5513,6238,6328,6330,6862] 
Tc_wind=wind_wmo[:]
Tc_wind[(Tc_wind<34)]=ma.masked
Tc_first=[]
for i in range(len(numObs)):
      temp1=Tc_wind[i].nonzero()
      first=temp1[0][0] if (len(temp1[0]) !=0) else np.nan
      Tc_first.append(first)
Tc_filled=Tc_wind.filled(np.nan)
Tc_numObs=np.nansum(np.divide(Tc_filled,Tc_filled),axis=1)
# -----------convert date to vector 
tt=time_wmo.filled()
time_wmo=np.reshape([jd2gcal(j,MJD_0) for i in tt for j in i],(len(numObs),-1))
# -----------chose prefined data
Tcdate=time_wmo[:,0]
#Tcdate_int=np.where((Tcdate>=1980)&(Tcdate<=2010))
Tcdate_int=np.where(np.logical_and(Tcdate>=1998,Tcdate<=2014))
Tcdate_int=Tcdate_int[0].tolist()
Tc_numObs=Tc_numObs.tolist()
temp=set([i for i in range(len(Tc_numObs)) if Tc_numObs[i]>9.0])& set(Tcdate_int)
if pacific:
      basin_ind=np.where(basin==2)
      basin_ind=set(basin_ind[0])

else:
      basin_ind=set(np.arange(7267))

            
temp=basin_ind&temp
ind=list(temp-set(weird_ind))
# ------------missing wind data
ind2=[]
for i in range(len(ind)):
      ii=ind[i]
      if sum(np.isnan(Tc_wind.filled(np.nan)[ii,Tc_first[ii]:int(Tc_first[ii]+\
                   Tc_numObs[ii])]))>0:
            ind2.append(i)
#ind=list(set(ind)-set(np.array(ind)[ind2]))
ind=[j for i,j in enumerate(ind) if i not in ind2]
datanum=len(ind)
# -----------interp
points=10
lat_interp=np.empty([datanum,points])
lon_interp=np.empty([datanum,points])
wind_interp=np.empty([datanum,points])
lat_wmo=lat_wmo.filled()
lon_wmo=lon_wmo.filled()
Tc_wind=Tc_wind.filled()
for i in range(datanum):
      ii=ind[i]
      time_index=np.arange(Tc_first[ii],Tc_first[ii]+Tc_numObs[ii],dtype='int64')
      time_interped=np.linspace(Tc_first[ii],Tc_first[ii]+Tc_numObs[ii]-1,points)
      f_lat=interp1d(time_index,lat_wmo[ii,time_index])
      f_lon=interp1d(time_index,lon_wmo[ii,time_index])
      f_wind=interp1d(time_index,Tc_wind[ii,time_index])
      lat_interp[i,:]=f_lat(time_interped)
      lon_interp[i,:]=f_lon(time_interped)
      wind_interp[i,:]=f_wind(time_interped)
      
   
if normalization:
      lat_interp0=lat_interp-(np.tile(lat_interp[:,0],(points,1))).T 
      lon_interp0=lon_interp-(np.tile(lon_interp[:,0],(points,1))).T  
      
 
else:
      lat_interp0=lat_interp
      lon_interp0=lon_interp
lat_interp1=(lat_interp0-np.nanmin(lat_interp0))/(np.nanmax(lat_interp0)-np.nanmin(lat_interp0))
lon_interp1=(lon_interp0-np.nanmin(lon_interp0))/(np.nanmax(lon_interp0)-np.nanmin(lon_interp0))

#lat_interp0[:,1]*=5
#lon_interp0[:,1]*=5
wind_interp0=(wind_interp-np.nanmin(wind_interp))/(np.nanmax(wind_interp)-np.nanmin(wind_interp))
Input=np.concatenate((lat_interp0,lon_interp0,wind_interp*0.5),axis=1)
#lat_interp0[:,1]/=5
#lon_interp0[:,1]/=5


# -----------SOM
class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    Apply to python 3.6 & new tensorflow grammar
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, method, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
 
        #Assign required variables first
        self._m = m
        self._n = n
        self.method= method
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
        #Method
        if method == "square":
              locations=np.array(list(self._neuron_locations_square(m, n)),dtype
                                 ="float32")
        elif method == "hex":
              locations=np.array(list(self._neuron_locations_hex(m, n)),dtype=
                                 "float32")
        ##Obtain the distances between different neures
        
        #children
        length=len(locations)
        locations_children=[[] for i in range(length)]
        for i ,val in enumerate(locations):
              for j in range(length):
                    dist=np.linalg.norm(val-locations[j])
                    if round(dist,1) ==1.0:
                          locations_children[i].append(j)
        
        #the Function MR_Right is to determine whether the children -"hope" concludes the 
        #Mr.Right (^_^)    
        distance=np.zeros((length,length))
        global indice
    
        global dead
        
        def MR_Right(hope,true):
               global indice  
               global dead
               store=set()
               for loc in hope:
                    if loc == true:
                          return indice
                          break
                    dead.add(loc)
                    for ll in locations_children[loc]:
                          if ll not in dead:
                                store.add(ll)
               else:
                    indice= indice+1
                    store=list(store)
                    return MR_Right(store,true)

        for i  in range(length):
              for j  in range(i,length):
                    indice=1
                    dead=set()
                    if i == j:
                          distance[i,j]=0
                    else:  
                          hope=locations_children[i]
                          distance[i,j]=MR_Right(hope,j)
        distance=np.matrix(distance)
        distance=distance+distance.T
        
        
             
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            tf.set_random_seed(1234)
            self._weightage_vects = tf.Variable(tf.random_normal(
                [m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            
            self._location_vects = tf.constant(locations)
                     
            self._distance=tf.constant(distance)
               
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
            
            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_percent = tf.subtract(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_percent)
            _sigma_op = tf.multiply(sigma, learning_rate_percent)
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_distance_squares = tf.pow(tf.reshape(tf.slice(self._distance, slice_input,
                                        tf.constant(np.array([1,-1],dtype=np.int64))),[m*n]),2)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                        bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                            self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations_square(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
                
    def _neuron_locations_hex(self,m,n):
        """
        Hexagonal grid for SOM.
        """
        h=math.sqrt(1**2-0.5**2)
        for i in range(m):
               if i%2 == 0: 
                     for j in range(n):
                          yield np.array([i*h,j])
               else:
                     for j in range(n):
                          yield np.array([i*h,0.5+j])    
        
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                  self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        #Store a centroid grid for easy retrieval later on
        self._locations = list(self._sess.run(self._location_vects))
        self._weightages = self._sess.run(self._weightage_vects)
        centroid_grid= self._weightages.reshape(self._m,self._n,-1)
      
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """ 
        if not self._trained:
            raise ValueError("SOM not trained yet") 
        to_return = []
        
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index]) 
        return to_return
  
    def classication(self,input_vects):
        """
        To calculate the number of vectors which have similar characters in each same 
        neuron. It can be used especially in the cases that m*n < the number of 
        'vect_input'
        """
        dic=[[] for i in range(self._m*self._n)]        
        for i,loc in enumerate(self._locations):
              for j,locinvc in enumerate(self.map_vects(input_vects)):
                    if np.array_equal(loc,locinvc):                         
                          dic[i].append(j)
        return dic
        
# -----------classify
# Attention: You would better normalize the Input vector.Otherwise, you need a 
# large sigma to modify the neure in case the emergence of "dead neure"
som=SOM(2,5,30,'square',100)
som.train(Input)
Tc_clas=som.classication(Input)

lat_ave=[]
lon_ave=[]
wind_ave=[]
for i,val in enumerate(Tc_clas):
      lat_ave.append(np.mean(lat_interp[val],axis=0))
      lon_ave.append(np.mean(lon_interp[val],axis=0))
      wind_ave.append(np.mean(wind_interp[val],axis=0))


lat_ave0=[]
lon_ave0=[]
for i,val in enumerate(Tc_clas):
      lat_ave0.append(np.mean(lat_interp0[val],axis=0))
      lon_ave0.append(np.mean(lon_interp0[val],axis=0))
# -----------plot1
"""
m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.bluemarble()
m.drawcountries()
lat=np.transpose(lat_interp)
lon=np.transpose(lon_interp)
m.plot(lon,lat,'r-',latlon=True,linewidth=0.5)
# draw parallels and meridians.
#m.drawparallels(np.arange(-90.,91.,60.),labels=[True,False,False,False])
#m.drawmeridians(np.arange(-180.,181.,120.),labels=[False,False,False,True])
plt.title('All typhoon tracks')

plt.savefig('All typhoon tracks.png', dpi=1000)
# -----------plot2
plt.figure()
plt.subplot(2,1,1)
m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.drawcountries()
number = som._m*som._n
cmap = plt.get_cmap('tab20')
clors = [cmap(i) for i in np.linspace(0, 1, number)]
for i,val in enumerate(Tc_clas):
      lat=np.transpose(lat_interp[val])
      lon=np.transpose(lon_interp[val])
      m.plot(lon,lat,'r-',latlon=True,color=clors[i])
plt.title('classified typhoon tracks')
plt.show()
#-----------------------------------------------------
plt.subplot(2,1,2)
m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.drawcountries()
number = som._m*som._n-1
cmap = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=number)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
for i,val in enumerate(Tc_clas):
      lat=np.transpose(lat_interp[val])
      lon=np.transpose(lon_interp[val])
      colorval=scalarMap.to_rgba(i)
      m.plot(lon,lat,'r-',latlon=True,color=colorval,linewidth=1)
plt.title('classified typhoon tracks')
plt.show()               
# -----------identified value-based color
# If you want to overlay these plots, just execute the plot mutiply. Please pay
# attention to the drawing order of axes, howerver, you can change it with "zorder"
# atrribute.
#the gradual color is based on the range of all averaged typhoon's wind value  
fig=plt.figure()
m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.drawcountries()
number = points*2
cmap = plt.get_cmap('copper')
clors = [cmap(i) for i in np.linspace(0, 1, number)]
for i in range(som._m*som._n):
      color=[]
      if not all(np.isnan(wind_ave[i])):
                  for j,val in enumerate(wind_ave[i]):
                        por=math.floor((val-np.nanmin(wind_ave))/(np.nanmax(wind_ave)\
                                        -np.nanmin(wind_ave))*(number-1))
                        color.append(clors[::-1][por])            
                  m.scatter(lon_ave[i],lat_ave[i],latlon=True,s=5,color=color)
plt.title('classified typhoon tracks')
plt.show()
fig.savefig('classified typhoon tracks',dpi=1000)
#the gradual color is based on the range of each averaged typhoon's wind value  
m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
            llcrnrlon=0,urcrnrlon=360,resolution='c')
m.drawcoastlines()
m.drawcountries()
number = som._m*som._n
cmap = plt.get_cmap('tab20')
clors = [cmap(i) for i in np.linspace(0, 1, number)]
for i in range(som._m*som._n):
      color=[]
      spec_clors = light2dark(clors[i],points*2)
      if not all(np.isnan(wind_ave[i])):
            for j,val in enumerate(wind_ave[i]):
                  por=math.floor((val-np.min(wind_ave[i]))/(np.max(wind_ave[i])\
                                  -np.min(wind_ave[i]))*(points*2-1))
                  color.append(spec_clors[::-1][por])            
            m.scatter(lon_ave[i],lat_ave[i],latlon=True,s=15,color=color)
plt.title('classified typhoon tracks')

plt.savefig('classified typhoon tracks.png', dpi=1000)
plt.show()



"""


# -----------combined world plot

#m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
#           llcrnrlon=0,urcrnrlon=360,resolution='c')
m=Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=60,\
            llcrnrlon=80,urcrnrlon=250,resolution='c')
m.drawcoastlines()
m.drawcountries()
number = som._m*som._n-1
cmap = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=number)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
legendd=[]
for i,val in enumerate(Tc_clas):
      lat=np.transpose(lat_interp[val])
      lon=np.transpose(lon_interp[val])
      colorval=scalarMap.to_rgba(i)
      m.plot(lon,lat,'r-',latlon=True,color=colorval,linewidth=0.5,zorder=1)
      legendd.append(mlines.Line2D([],[],color=colorval,label="c"+str(i+1)+" "+str(len(val))))
      
number = points*2
cmap = plt.get_cmap('copper')
clors = [cmap(i) for i in np.linspace(0, 1, number)]
for i in range(som._m*som._n):
      color=[]
      if not all(np.isnan(wind_ave[i])):
                  for j,val in enumerate(wind_ave[i]):
                        por=math.floor((val-np.nanmin(wind_ave))/(np.nanmax(wind_ave)\
                                        -np.nanmin(wind_ave))*(number-1))
                        color.append(clors[::-1][por])                       
                  m.scatter(lon_ave[i],lat_ave[i],latlon=True,s=3,color=color,edgecolor="k",linewidth=0.3,
                  zorder=2)

plt.title('classified typhoon tracks')
lgd=plt.legend(handles=legendd,fontsize=8,bbox_to_anchor=(1, 1),loc=2)
plt.savefig('classified typhoon tracks2.png', dpi=1000,bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
'''
#-----------divergent plot
if normalization:
      for i in range(som._m*som._n):
            plt.plot(lon_ave0[i],lat_ave0[i],label="c"+str(i+1))
      plt.legend(bbox_to_anchor=(1, 1),loc=2)
      plt.title('normalized distribution')
      
# -----------combined WP plot      
m=Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=60,\
            llcrnrlon=80,urcrnrlon=250,resolution='c')
m.drawcoastlines()
m.drawcountries()
number = som._m*som._n-1
cmap = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=number)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
legendd=[]
for i,val in enumerate(Tc_clas):
      lat=np.transpose(lat_interp[val])
      lon=np.transpose(lon_interp[val])
      colorval=scalarMap.to_rgba(i)
      m.plot(lon,lat,'r-',latlon=True,color=colorval,linewidth=0.5,zorder=1)
      m.scatter(lon_ave[i],lat_ave[i],latlon=True,s=3,color=colorval,edgecolor="k",\
                linewidth=0.5,zorder=2)
      legendd.append(mlines.Line2D([],[],color=colorval,label="c"+str(i+1)+" "+str(len(val))))
      

plt.title('classified west Pacific typhoon tracks')
lgd=plt.legend(handles=legendd,fontsize=8,bbox_to_anchor=(1, 1),loc=2)
plt.savefig('classified typhoon tracks2.png', dpi=1000,bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
'''