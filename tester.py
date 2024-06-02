import numpy as np
import xarray as xr
import random as rn
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
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


# %% read data
filepath_data = 'ERA5_tester_data.nc'
data = xr.open_dataset(filepath_data)

# %% important variables
check_date = '2010-01-26 00:00:00'
var_list = ['tcc', 'p71.162', 'p72.162', 'p54.162']
keys = ['longitude', 'latitude', 'time']

# %% Checking the data via plots
data_plot = data.sel(time=slice(check_date, check_date))

plt.figure(figsize=(8,5))
num_var = len(var_list)
for i in range(num_var):
  ax = plt.subplot(math.ceil(num_var/2), math.ceil(num_var/2), i+1, projection=DATA_CRS)
  ax.set_extent([min(data.longitude), max(data.longitude), min(data.latitude), max(data.latitude)], crs=DATA_CRS)
  ax.coastlines(resolution='50m')
  data_plot[var_list[i]].plot(cmap=CMAP[i])
  plt.title(var_list[i])
plt.show()
# plt.savefig("tcc.png")

# %% Data cleaning
# standardize
for var in var_list:
   data[var] = (data[var]-data[var].mean())/data[var].std()

# split and resize
x_vars = []
X = []
for i in range(len(var_list)-1):
   x_vars.append(var_list[i+1])
   X.append(data[var_list[i+1]])

X = np.moveaxis(np.asarray(X), 0, -1)
Y = np.expand_dims(data['tcc'].values, 3)

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

## given lat and lon make matrix per time
# %% split to training, test and validation(80 20?)
df_train, df_test, df_train_label, df_test_label = train_test_split(X, Y, test_size=0.2,
                                                                      random_state=13)

df_train, df_valid, df_train_label, df_valid_label = train_test_split(df_train, df_train_label, 
                                                                      test_size=0.2, 
                                                                      random_state=13)

# %% train model -callbacks? optimizers or losses or activation
model = Sequential()
model.add(Conv2D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION,
                  input_shape=(X.shape[1],X.shape[2],X.shape[3])))

for layer in range(NUM_LAYERS-1):
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
