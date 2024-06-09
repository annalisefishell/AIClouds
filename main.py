# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import math
import os
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, LeakyReLU, Dense


# %% constants 
# reading
FILEPATH_DATA = ['ERA5-total_cloud_cover-1961-1980-WQBox.nc4', # always make target first
                  'ERA5-total_column_water-1961-1980-WQBox.nc4', 
                  'ERA5-2m_temperature-1961-1980-WQBox.nc4'] # ['ERA5_tester_data.nc']
FILEPATH_CLEANED_DATA = 'cleaned_data.pkl'
HAPPY_W_DATA = True
NUM_FILES = len(FILEPATH_DATA)

# plotting
DATA_CRS = ccrs.PlateCarree()
CMAP = [cmr.get_sub_cmap('GnBu', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1), 
        cmr.get_sub_cmap('coolwarm', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1)]

# dates
START_DATE = '1961-01-01T00:00:00.000000000' #'2010-01-01 00:00:00'
SPLIT_DATE = '1961-01-20T00:00:00.000000000' #'2010-07-31 00:00:00'
END_DATE = '1961-01-25T00:00:00.000000000'

# model
FILEPATH_MODEL = 'model.pkl'
EPOCH = 10
NUM_FILTERS = 30
NUM_HIDDEN_LAYERS = 3
ACTIVATION = 'relu'
KERNEL_SIZE = (3,3)


# %% plotting functions
def plot_var(data_plot, i, var, num_var):
  '''Given a slice of an xarray dataset and a specific variable, 
  plots the matrix of the region. Also needs specification on the
  index and number of variables for the allocation of the subplot'''

  ax = plt.subplot(math.ceil(num_var/2), math.ceil(num_var/2), i+1, 
                   projection=DATA_CRS)
  ax.set_extent([min(data_plot.lon), max(data_plot.lon), 
                 min(data_plot.lat), max(data_plot.lat)], 
                 crs=DATA_CRS)
  ax.coastlines(resolution='50m')

  im = data_plot[var].plot(cmap=CMAP[i],
                           cbar_kwargs={"label": 
                                        data_plot[var].attrs['units']})

  plt.title(data_plot[var].attrs['long_name']) #needs to be formatted!!

def plot_all_vars(ds, vars, date, save=False):
   '''Given an xarray dataset, plots all the variables in the 
   variable list given they are geographic in subplots. The 
   save variable chooses whether to show of save in a file.'''

   data_plot = ds.sel(time=slice(date, date))
   num_var = len(vars)

   plt.figure(figsize=(10,5))
   for i in range(num_var):
      plot_var(data_plot, i, vars[i], num_var)

   if save:
      plt.savefig("trial.png")
   else:
      plt.show()


# %% pre-processing functions
def standardize(ds, vars):
   '''Standardize all the variables in a dataset using the basic 
   method, and returns the standardized dataset.'''

   for var in vars:
      ds[var] = ((ds[var]-ds[var].values.mean())/ds[var].values.std())
   return ds

def reshape_vars(ds, vars):
   '''Given a dataset, seperates the independent and dependent 
   variables for the cnn and reshape the arrays to fit the model 
   format. Returns both the numpy arrays.'''
   
   X = []
   for i in range(len(vars)-1):
      X.append(ds[vars[i+1]])

   X = np.moveaxis(np.asarray(X), 0, -1)
   Y = np.expand_dims(ds[vars[0]].values, 3)

   return X,Y

def clean_data(data, var_list, start, end):
   temp_data = data.sel(time=slice(start, end))
   temp_data = standardize(temp_data, var_list)
   x, y = reshape_vars(temp_data, var_list)
   return x,y

def read_data(reduce=False):
   '''Given a file path, checks if the data was already read 
   an processed. If yes, obtains only the cleaned data. If not,
   opens the data, cleans it, and saves it in a new file. Returns
   both the data and a list of present variables.'''
   
   if os.path.isfile(FILEPATH_CLEANED_DATA) and HAPPY_W_DATA:
      print('Data already preprocessed')
      with open(FILEPATH_CLEANED_DATA, 'rb') as f:
         loaded_dict = pickle.load(f)
         var_list = loaded_dict['variables']
         x_train = loaded_dict['x_train']
         y_train = loaded_dict['y_train']
         x_test = loaded_dict['x_test']
         y_test = loaded_dict['y_test']
   else:
      print('Reading data.....')
      data = xr.open_dataset(FILEPATH_DATA[0])
      for i in range(NUM_FILES-1):
         t = xr.open_dataset(FILEPATH_DATA[i+1])
         data = xr.merge([data, t])

      if reduce: # only way to make big dataset work (maybe also consider coarsen)
         data = data.sel(lon=slice(-10,20), lat=slice(0,10))

      var_list = list(data.keys())
      x_train, y_train = clean_data(data, var_list, START_DATE, SPLIT_DATE)
      x_test, y_test = clean_data(data, var_list, SPLIT_DATE, END_DATE)

      plot_all_vars(data, var_list, SPLIT_DATE, True)
      data.close()
      
      with open(FILEPATH_CLEANED_DATA, 'wb') as f:
         pickle.dump({'variables': var_list,
                      'x_train': x_train, 'y_train': y_train,
                      'x_test': x_test, 'y_test': y_test}, f)

   return var_list, x_train, y_train, x_test, y_test


# %% model funcions
def build_model(x): # works, but need to adjust a lo of paameters - callbacks? optimizers or losses or activation
   '''Build the bones of a convolutional neural network with the same 
   input and output shape of x. Explain more once I figure out
   parameters. Then returns the model structure.'''
   
   model = Sequential()
   model.add(Conv2D(filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, 
                    activation=ACTIVATION, 
                    input_shape=(x.shape[1],x.shape[2],x.shape[3]),
                    padding='same'))

   for layer in range(NUM_HIDDEN_LAYERS-1):
      model.add(Conv2D(filters=NUM_FILTERS*(layer+1), 
                       kernel_size=KERNEL_SIZE, activation=ACTIVATION,
                       padding='same'))
      model.add(LeakyReLU(alpha=0.1))

   model.add(Dropout(0.5))
   model.add(Dense(x.shape[1]*x.shape[2]))

   model.compile(loss='mean_squared_error', optimizer='adam', 
                 metrics=['mean_absolute_error']) # which meterics do I want to use?
   
   return model

def load_model(x, FILEPATH_MODEL):
   '''Check if a CNN model is already built. If so, load the model from 
   the given pickle filepath. If not build it given the dimensions from 
   the dataset x, and save it to the filepath.'''

   if os.path.isfile(FILEPATH_MODEL) and HAPPY_W_DATA:
      print('Model already trained')
      model = pickle.load(open(FILEPATH_MODEL, 'rb'))
   else:
      print('Training model.....')
      model = build_model(x)
      pickle.dump(model, open(FILEPATH_MODEL, 'wb'))

   return model

def evaluate_model(model, test, test_labels):
   '''Given a trained model, use the test dataset and corect 
   labels to evaluate the performance.'''

   eval = model.evaluate(test, test_labels)
   print('loss: ', eval[0])
   print('mae: ', eval[1])


# %% main
tf.keras.backend.clear_session()

var_list, x_train, y_train, x_test, y_test = read_data(NUM_FILES!=0)

df_train, df_valid, df_train_label, df_valid_label = train_test_split(x_train, y_train, 
                                                                     test_size=0.2,
                                                                     random_state=13)

model = load_model(df_train, FILEPATH_MODEL)
model.fit(df_train, df_train_label, epochs=EPOCH, batch_size=len(df_train)//EPOCH, 
          validation_data=(df_valid, df_valid_label))
# print(model.summary())

evaluate_model(model, x_test, y_test)