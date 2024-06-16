'''
To do:
- adjust data based on size (generator/regional models/gpu/server?)
- perfect model parameters
- increase evaluation of the model (time)
'''
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
# convert to pytorch?

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
# CMAP = [cmr.get_sub_cmap('GnBu', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1), 
#         cmr.get_sub_cmap('coolwarm', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1)]

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
def determine_num_subplots(num_var):
   '''Determine how many subplots are needed and define the 
   subsequent square dimensions of those plots'''

   if num_var == 2:
      x, y = 1, 2
   else:
      x = y = math.ceil(num_var/2)

   return x, y

def plot_var(data_plot, i, var, num_var):
  '''Given a slice of an xarray dataset and a specific variable, 
  plots the matrix of the region. Also needs specification on the
  index and number of variables for the allocation of the subplot'''

  x,y = determine_num_subplots(num_var)
  ax = plt.subplot(x, y, i+1, projection=DATA_CRS)
  ax.set_extent([min(data_plot.lon), max(data_plot.lon), 
                 min(data_plot.lat), max(data_plot.lat)], 
                 crs=DATA_CRS)
  ax.coastlines(resolution='50m')

  im = data_plot[var].plot(cmap=cmr.get_sub_cmap('GnBu', 0, 1),
                           cbar_kwargs={"label": 
                                        data_plot[var].attrs['units']})

  plt.title(data_plot[var].attrs['long_name']) #needs to be formatted!!

def plot_all_vars(ds, vars, date, filename=''):
   '''Given an xarray dataset, plots all the variables in the 
   variable list given they are geographic in subplots. The 
   save variable chooses whether to show of save in a file.'''

   data_plot = ds.sel(time=slice(date, date))
   num_var = len(vars)

   plt.figure(figsize=(10,5))
   for i in range(num_var):
      plot_var(data_plot, i, vars[i], num_var)

   if filename != '':
      plt.savefig(filename)
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

def clean_data(ds, vars, start, end):
   '''Select desired data from a larger set, standardizing it 
   and adjusting it into the data type and shape needed for 
   further use.'''

   temp_data = ds.sel(time=slice(start, end))
   temp_data = standardize(temp_data, vars)
   x, y = reshape_vars(temp_data, vars)
   return x,y

def open_datafile():
   '''Read the data and combine subsequent files into one'''

   data = xr.open_dataset(FILEPATH_DATA[0])
   for i in range(NUM_FILES-1):
      t = xr.open_dataset(FILEPATH_DATA[i+1])
      data = xr.merge([data, t])
   
   return data

def get_key_info(ds):
   '''Get a list of numpy arrays containing the unique values of each 
   corrisponding index variable'''

   start_index = np.where(ds.time.values == np.datetime64(START_DATE))
   end_index = np.where(ds.time.values == np.datetime64(END_DATE))
   keys = [ds.time.values[int(start_index[0][0]):int(end_index[0][0])+1], 
            ds.lon.values, ds.lat.values]
   
   return keys

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
         keys = loaded_dict['keys']
         x_train = loaded_dict['x_train']
         y_train = loaded_dict['y_train']
         x_test = loaded_dict['x_test']
         y_test = loaded_dict['y_test']
   else:
      print('Preparing data.....')
      data = open_datafile()

      if reduce: # only way to make big dataset work (maybe also consider coarsen)
         data = data.sel(lon=slice(-10,20), lat=slice(0,10))

      var_list = list(data.keys())
      x_train, y_train = clean_data(data, var_list, START_DATE, SPLIT_DATE)
      x_test, y_test = clean_data(data, var_list, SPLIT_DATE, END_DATE)
      keys = get_key_info(data)

      plot_all_vars(data, var_list, SPLIT_DATE, 'original_data.png')
      data.close()
      
      with open(FILEPATH_CLEANED_DATA, 'wb') as f:
         pickle.dump({'variables': var_list, 'keys': keys,
                      'x_train': x_train, 'y_train': y_train,
                      'x_test': x_test, 'y_test': y_test}, f)

   return var_list, keys, x_train, y_train, x_test, y_test


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
   model.add(Dense(1)) #x.shape[1]*x.shape[2]

   model.compile(loss='mean_squared_error', optimizer='adam', 
                 metrics=['mean_absolute_error']) # which meterics do I want to use?
   
   return model

def load_model(x, FILEPATH_MODEL, train, train_label, valid, valid_label):
   '''Check if a CNN model is already built. If so, load the model from 
   the given pickle filepath. If not build it given the dimensions from 
   the dataset x, and save it to the filepath.'''

   if os.path.isfile(FILEPATH_MODEL) and HAPPY_W_DATA:
      print('Model already built')
      model = pickle.load(open(FILEPATH_MODEL, 'rb'))
   else:
      print('Building model.....')
      model = build_model(x)
      model.fit(train, train_label, epochs=EPOCH, batch_size=len(train)//EPOCH, 
          validation_data=(valid, valid_label))
      pickle.dump(model, open(FILEPATH_MODEL, 'wb'))

   return model


# %% evaluate functions
def build_compare_xarray(pred, real, keys):
   '''Constructs a xarray dataset that contains the predicted values, 
   real values and the difference between the two for each day in 
   the test dataset'''

   start_index = np.where(keys[0] == np.datetime64(SPLIT_DATE))
   end_index = np.where(keys[0] == np.datetime64(END_DATE))
   pred_time = keys[0][int(start_index[0][0]):int(end_index[0][0])+1], 

   target = np.array(pred).flatten().reshape(len(pred_time[0]),
                                               len(keys[2]),len(keys[1]))
   real = np.squeeze(real, axis=-1)
   diff = target-real

   compare = xr.Dataset(
      data_vars=dict(
         pred_tcc = (['time', 'lat', 'lon'], target, {'units': '%', 
                      'long_name': 'Predicted Total Cloud Cover'}),
         real_tcc = (['time', 'lat', 'lon'], real, {'units': '%', 
                      'long_name': 'Real Total Cloud Cover'}),
         diff_tcc = (['time', 'lat', 'lon'], diff, 
                     {'units': 'Difference of percentages',
                      'long_name': 'Difference'})),
      coords = dict(
         time = ('time', pred_time[0]),
         lat = ('lat', keys[2]),
         lon = ('lon', keys[1])),
      attrs = dict(description="Real total cloud cover compared to \
                 predicted from CNN")
   )
   return compare

def evaluate_model(model, test, test_labels, keys):
   '''Given a trained model, use the test dataset and corect 
   labels to evaluate the performance.'''

   eval = model.evaluate(test, test_labels)
   print('loss: ', eval[0])
   print('mae: ', eval[1])

   y_pred = model.predict(test)
   compare = build_compare_xarray(y_pred, test_labels, keys)
   
   plot_all_vars(compare, list(compare.keys()), END_DATE, 'comparison.png') # plot multiple days? see when the worst, why?


# %% main
tf.keras.backend.clear_session()

var_list, keys, x_train, y_train, x_test, y_test = read_data(NUM_FILES!=0)

df_train, df_valid, df_train_label, df_valid_label = train_test_split(x_train, y_train, 
                                                                     test_size=0.2,
                                                                     random_state=13)

model = load_model(df_train, FILEPATH_MODEL, df_train,
                  df_train_label, df_valid, df_valid_label)
# print(model.summary())

evaluate_model(model, x_test, y_test, keys)