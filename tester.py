import numpy as np
import xarray as xr
import random as rn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, LeakyReLU, MaxPooling2D, Dense


LR = 0.001
EPOCH = 100
BS = 32
NUM_FILTERS = 32
NUM_LAYERS = 3
ACTIVATION = 'relu'
KERNEL_SIZE = (3,3)
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
  # plt.show()
  plt.savefig("tcc.png")


# %% read data
filepath_data = 'ERA5_tester_data.nc'
data = xr.open_dataset(filepath_data)
df = data.to_dataframe()


# %% important variables
check_date = '2010-01-01 00:00:00'
var_list = ['tcc', 'p71.162', 'p72.162', 'p54.162'] # interest is always 0
keys = ['longitude', 'latitude', 'time']
num_data = df.shape[0]

x_vars = []
for i in range(len(var_list)):
   x_vars.append(var_list[i])


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

# %% split to training, test and validation(80 20?)
df_train, df_test, df_train_label, df_test_label = train_test_split(df[x_vars],
                                                                      df[var_list[0]],
                                                                      test_size=0.2,
                                                                      random_state=13)
df_train, df_valid, df_train_label, df_valid_label = train_test_split(df_train, 
                                                                      df_train_label, 
                                                                      test_size=0.2, 
                                                                      random_state=13)


# %% train model -callbacks? optimizers or losses or activation
model = Sequential()
model.add(Conv2D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION,
                  input_shape=(9,9,1)))


for layer in NUM_LAYERS-1:
   model.add(Conv2D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION))
   model.add(LeakyReLU(alpha=0.1))
   model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation=ACTIVATION))

model.compile(loss='mean_squares_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(df_train, df_train_label, epochs=EPOCH, batch_size=BS, 
          validation_data=(df_valid, df_valid_label))


# %% test model
eval = model.evaluate(df_test, df_test_label)
print('loss: ', eval[0])
print('mae: ', eval[1])
