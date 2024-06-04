# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import math
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, LeakyReLU, MaxPooling2D, Dense


# %% constants 
# reading
FILEPATH_DATA = ['ERA5-total_cloud_cover-1961-1980-WQBox.nc4', # always make target 1
                 'ERA5-total_column_water-1961-1980-WQBox.nc4', 
                 'ERA5-2m_temperature-1961-1980-WQBox.nc4'] #['ERA5_tester_data.nc']
FILEPATH_CLEANED_DATA = 'cleaned_data.nc4'
HAPPY_W_DATA = False
NUM_FILES = len(FILEPATH_DATA)

# plotting
DATA_CRS = ccrs.PlateCarree()
CMAP = [cmr.get_sub_cmap('GnBu', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1), 
        cmr.get_sub_cmap('coolwarm', 0, 1)] # cmr.get_sub_cmap('PuBu', 0, 1),
START_DATE = '1961-01-01T00:00:00.000000000' #'2010-01-26 00:00:00'
END_DATE = '1965-01-01T00:00:00.000000000'

# model
LR = 0.001
EPOCH = 100
BS = 32
NUM_FILTERS = 32
NUM_HIDDEN_LAYERS = 3
ACTIVATION = 'relu'
KERNEL_SIZE = (3,3)

# %% plotting functions
def plot_var(data_plot, i, var, num_var):
  '''Given a slice of an xarray data set plots the matrix of the region'''
  ax = plt.subplot(math.ceil(num_var/2), math.ceil(num_var/2), i+1, 
                   projection=DATA_CRS)
  ax.set_extent([min(data_plot.lon), max(data_plot.lon), 
                 min(data_plot.lat), max(data_plot.lat)], 
                 crs=DATA_CRS)
  ax.coastlines(resolution='50m')
  im = data_plot[var].plot(cmap=CMAP[i],
                           cbar_kwargs={"label": data_plot[var].attrs['units']})

  plt.title(data_plot[var].attrs['long_name']) #needs to be formatted

def plot_all_vars(ds, vars, save=False):
   '''Given an xarray dataset plots all the variables given they are 
   geographic'''
   data_plot = ds.sel(time=slice(START_DATE, START_DATE))
   num_var = len(vars)

   plt.figure(figsize=(6,10))
   for i in range(num_var):
      plot_var(data_plot, i, vars[i], num_var)
   if save:
      plt.savefig("trial.png")
   else:
      plt.show()


# %% pre-processing functions
def standardize(ds, vars):
   for var in vars:
      ds[var] = ((ds[var]-ds[var].values.mean())/ds[var].values.std())
   return ds

def reshape_vars(ds, vars):
   X = []
   for i in range(len(vars)-1):
      X.append(ds[vars[i+1]])

   X = np.moveaxis(np.asarray(X), 0, -1)
   Y = np.expand_dims(ds[vars[0]].values, 3)
   return X,Y

def read_data():
   if os.path.isfile(FILEPATH_CLEANED_DATA) and HAPPY_W_DATA:
      print ('Data already preprocessed')
      data_slice = xr.open_dataset(FILEPATH_CLEANED_DATA)
      var_list = list(data_slice.keys())
   else:
      data = xr.open_dataset(FILEPATH_DATA[0])
      for i in range(NUM_FILES-1):
         t = xr.open_dataset(FILEPATH_DATA[i+1])
         data = xr.merge([data, t])

      data_slice = data.sel(time=slice(START_DATE, END_DATE))
      var_list = list(data.keys())

      data_slice = standardize(data_slice, var_list)
      # need to add more preprocessing?
      #build something for the domain and the grid with pixels?
      # ds.sel(lon=slice(self.lon_b,self.lon_e),lat=slice(self.lat_b,self.lat_e))

      #test for leap year?
      # DATASET = DATASET_wleap.sel(time=~((DATASET_wleap.time.dt.month==2) & (DATASET_wleap.time.dt.day==29)))

      #include seasons?
      # if seas :
      #     cosvect = np.tile([cos(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 
      #     sinvect = np.tile([sin(2*i*pi/nbdays)  for i in range(nbdays)],int(INPUT_2D.shape[0]/nbdays)) 

      #     INPUT_1D.append(cosvect.reshape(INPUT_2D.shape[0],1,1,1))
      #     INPUT_1D.append(sinvect.reshape(INPUT_2D.shape[0],1,1,1))
      data_slice.to_netcdf(path=FILEPATH_CLEANED_DATA)

   # plot_all_vars(data_slice, var_list, True)

   return data_slice, var_list


# %% model funcions
def build_model(x): # need to adjust a lo of paameters - callbacks? optimizers or losses or activation
   model = Sequential()
   model.add(Conv2D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION,
                  input_shape=(x.shape[1],x.shape[2],x.shape[3])))

   for layer in range(NUM_HIDDEN_LAYERS-1):
      model.add(Conv2D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, activation=ACTIVATION))
      model.add(LeakyReLU(alpha=0.1))
      model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

   model.add(Flatten())
   model.add(Dropout(0.5))
   model.add(Dense((x.shape[1],x.shape[2]), activation=ACTIVATION)) #maybe need to change to make per matrix (size of len lon vs len lat)

   model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
   return model


# %% main
data, var_list = read_data()

x,y = reshape_vars(data, var_list)

df_train, df_valid, df_train_label, df_valid_label = train_test_split(x, y, test_size=0.2,
                                                                      random_state=13)

model = build_model(x)
model.fit(df_train, df_train_label, epochs=EPOCH, batch_size=BS, 
          validation_data=(df_valid, df_valid_label))


# %% test model
# eval = model.evaluate(df_test, df_test_label)
# print('loss: ', eval[0])
# print('mae: ', eval[1])