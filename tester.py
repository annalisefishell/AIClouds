import numpy as np
import xarray as xr
import random as rn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr

LR = 0.001
EPOCH = 100
BS = 32
DATA_CRS = ccrs.PlateCarree()
CMAP = cmr.get_sub_cmap('gist_earth_r', 0, 1)

def plot_var(ax, var, lon, lat, title, cmap="Blues", cbar_label="",
             cbar_label_rotation=0, cbar_location="right"):
  '''Plots a map on axis a with the projection defined as a variable in the
  code. The map then has a layer determined by the var variable. If the data
  given represents corals, the colorbar is made to be categorical.'''

  # plot
  ax.set_global()
  ax.coastlines(resolution='50m')
  im=plt.pcolormesh(lon, lat, var, transform=DATA_CRS, cmap=cmap)
  plt.title(title)

  # adjust colorbar
  cbar = plt.colorbar(im, ax=ax, shrink=0.4, pad=0.02, label=cbar_label,
                      location=cbar_location)
  cbar.ax.tick_params(rotation=cbar_label_rotation)


#read data
filepath_data = 'ERA5_tester_data.nc'
data = xr.open_dataset(filepath_data)
df = data.to_dataframe()

#important variables
dependent = 'tcc'
independent = ['p71.162', 'p72.162', 'p54.162']
keys = ['longitude', 'latitude', 'time']
num_data = df.shape[0]
num_var = len(df)

plt.figure(figsize=(12,5), layout='constrained')

df_plot = df.loc[:,:,'2010-01-01 00:00:00']['tcc']

lon = np.zeros(len(df_plot.index))
lat = np.zeros(len(df_plot.index))

for i in range(len(df_plot.index)):
    lon[i] = df_plot.index[i][0]
    lat[i] = df_plot.index[i][1]

temp=[]
for i in np.unique(lon):
   temp.append(list(df.loc[i,:,'2010-01-01 00:00:00']['tcc'].values))
data_plot = np.array(temp)

ax = plt.subplot(1,2,1, projection=DATA_CRS)

plot_var(ax, data_plot, np.unique(lon), np.unique(lat), "Total cloud cover",
         CMAP, "Percent coverage", cbar_location="left")
plt.savefig("tcc.png")

rn.seed(123)
#build something for the domain and the grid with pixels
# ds.sel(lon=slice(self.lon_b,self.lon_e),lat=slice(self.lat_b,self.lat_e))

#test for leap year 
# DATASET = DATASET_wleap.sel(time=~((DATASET_wleap.time.dt.month==2) & (DATASET_wleap.time.dt.day==29)))

#compute means and stds?

#standardize - either with ref array or alone

#include seasons
# if seas :
#     cosvect = np.tile([cos(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 
#     sinvect = np.tile([sin(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 

#     INPUT_1D.append(cosvect.reshape(INPUT_2D.shape[0],1,1,1))
#     INPUT_1D.append(sinvect.reshape(INPUT_2D.shape[0],1,1,1))

#split to training and test (80 20?)
# idx_train = rn.sample(range(num_data), int(0.8*num_data))
# df_train = [df[k][idx_train,:,:,:] for k in range(num_var)]
# df_test = [np.delete(df[k],idx_train,axis=0) for k in range(num_var)]

# #split dataset into dependent and independent variables
# df_train_dep = df_train[dependent]
# df_train_ind = df_train[independent]

# df_test_dep = df_test[dependent]
# df_test_ind = df_test[independent]

#train model

#test model