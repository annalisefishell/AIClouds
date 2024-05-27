import numpy as np
import xarray as xr
import random as rn

LR = 0.001
EPOCH = 100
rn.seed(123)

#read data
filepath_data = 'ERA5_tester_data.nc'
data = xr.open_dataset(filepath_data)
df = data.to_dataframe() # pandas dataframe (dep and ind as columns and scales as keys)
# df.time for keys and df['tcc] for variables

#important variables
dependent = 'tcc'
independent = ['p71.162', 'p72.162', 'p54.162']
scales = ['longitude', 'latitude', 'time'] # these are the keys, dont actually use
num_data = df.shape[0]
num_var = len(df)

#build something for the domain and the grid with pixels
# ds.sel(lon=slice(self.lon_b,self.lon_e),lat=slice(self.lat_b,self.lat_e))

#test for leap year 
# DATASET = DATASET_wleap.sel(time=~((DATASET_wleap.time.dt.month==2) & (DATASET_wleap.time.dt.day==29)))

#standardize - either with ref array or alone

#compute means and stds?

#include seasons
# if seas :
#     cosvect = np.tile([cos(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 
#     sinvect = np.tile([sin(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 

#     INPUT_1D.append(cosvect.reshape(INPUT_2D.shape[0],1,1,1))
#     INPUT_1D.append(sinvect.reshape(INPUT_2D.shape[0],1,1,1))

#split to training and test (80 20?)
idx_train = rn.sample(range(num_data), int(0.8*num_data))
df_train = [df[k][idx_train,:,:,:] for k in range(num_var)]
df_test = [np.delete(df[k],idx_train,axis=0) for k in range(num_var)]

#split dataset into dependent and independent variables
df_train_dep = df_train[dependent]
df_train_ind = df_train[independent]

df_test_dep = df_test[dependent]
df_test_ind = df_test[independent]

#train model

#test model