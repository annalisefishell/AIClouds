import numpy as np
import xarray as xr
import random as rn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import math


LR = 0.001
EPOCH = 100
BS = 32
DATA_CRS = ccrs.PlateCarree()
CMAP = [cmr.get_sub_cmap('binary_r', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1), 
        cmr.get_sub_cmap('PuBu', 0, 1), cmr.get_sub_cmap('coolwarm', 0, 1)]


def plot_var(ax, var, lon, lat, title, cmap="Blues", cbar_label="",
             cbar_label_rotation=0, cbar_location="right", regional=True):
  '''Plots a map on axis a with the projection defined as a variable in the
  code. The map then has a layer determined by the var variable. If the data
  given represents a specific region, the axis can be zoomed.'''

  # plot
  if regional:
       ax.set_extent([min(lon), max(lon), min(lat), max(lat)], crs=DATA_CRS)
  else:
     ax.set_global()

  ax.coastlines(resolution='50m')
  im=plt.pcolormesh(lon, lat, var, transform=DATA_CRS, cmap=cmap)
  plt.title(title)

  # adjust colorbar
  cbar = plt.colorbar(im, ax=ax, shrink=0.4, pad=0.02, label=cbar_label,
                      location=cbar_location)
  cbar.ax.tick_params(rotation=cbar_label_rotation)

def obtain_lon_lat(dataframe):
  '''Finds the unique values for the longitude and latitude of the desired area.
  Do not use if mapping globally.'''
  lon = np.zeros(len(dataframe.index))
  lat = np.zeros(len(dataframe.index))
  for i in range(len(dataframe.index)):
      lon[i] = dataframe.index[i][0]
      lat[i] = dataframe.index[i][1]
  lon = np.unique(lon)
  lat = np.unique(lat)
  return lon, lat

def plot_vars(dataframe, variables, date):
  '''Plots the values of a dataframe given a specific date and a variable list
  of unknown length'''
  num_var = len(variables)
  df_plot = dataframe.loc[:,:,date]

  lon, lat = obtain_lon_lat(df_plot)

  plt.figure(figsize=(8,5), layout='constrained')
  for i in range(num_var):
    t = []
    for l in lon:
      t.append(list(dataframe.loc[l,:,date][variables[i]].values))
    ax = plt.subplot(math.ceil(num_var/2), math.ceil(num_var/2), i+1, projection=DATA_CRS)
    plot_var(ax, t, lon, lat, variables[i], CMAP[i], " ", cbar_location="left")  
  plt.show()
# plt.savefig("tcc.png")

# %% read data
filepath_data = 'ERA5_tester_data.nc'
data = xr.open_dataset(filepath_data)
df = data.to_dataframe()

# %% important variables
check_date = '2010-01-01 00:00:00'
var_list = ['tcc', 'p71.162', 'p72.162', 'p54.162'] # interest is always 0
keys = ['longitude', 'latitude', 'time']
num_data = df.shape[0]

# %% Checking the data via plots
plot_vars(df, var_list, check_date)

# %% Data cleaning
#standardize
for var in var_list:
   df[var] = (df[var]-df[var].mean())/df.std()

#build something for the domain and the grid with pixels
# ds.sel(lon=slice(self.lon_b,self.lon_e),lat=slice(self.lat_b,self.lat_e))

#test for leap year 
# DATASET = DATASET_wleap.sel(time=~((DATASET_wleap.time.dt.month==2) & (DATASET_wleap.time.dt.day==29)))

#include seasons
# if seas :
#     cosvect = np.tile([cos(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 
#     sinvect = np.tile([sin(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 

#     INPUT_1D.append(cosvect.reshape(INPUT_2D.shape[0],1,1,1))
#     INPUT_1D.append(sinvect.reshape(INPUT_2D.shape[0],1,1,1))

# %% split to training and test (80 20?)
rn.seed(123)
# idx_train = rn.sample(range(num_data), int(0.8*num_data))
# df_train = [df[k][idx_train,:,:,:] for k in range(num_var)]
# df_test = [np.delete(df[k],idx_train,axis=0) for k in range(num_var)]

# #split dataset into dependent and independent variables
# df_train_dep = df_train[dependent]
# df_train_ind = df_train[independent]

# df_test_dep = df_test[dependent]
# df_test_ind = df_test[independent]

# %% train model

# %% test model