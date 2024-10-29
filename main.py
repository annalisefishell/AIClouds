'''
To do:
- document code completely
- increase evaluation of the model (time) (methods)
'''
# %% imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import time
import math
import os
import pickle
import scipy.stats as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Conv2DTranspose, \
   Concatenate, Dropout, LeakyReLU, Dense, Activation, Resizing


# %% constants 
# reading data
FILEPATH_DATA = ['variables/ERA5-total_cloud_cover-1961-1980-WQBox.nc4', # always make target first
                  'variables/ERA5-total_column_water-1961-1980-WQBox.nc4', 
                  'variables/ERA5-2m_temperature-1961-1980-WQBox.nc4']
                  # 'variables/ERA5-total_precipitation-1961-1980-WQBox.nc4']
FILEPATH_CLEANED_DATA = 'cleaned_data.pkl'
HAPPY_W_DATA = True
NUM_VARS = len(FILEPATH_DATA)
FILEPATH_MASKS = [] #'variables/ERA5_elevation-WQBox.nc',
                  # 'variables/ERA5-land_sea_mask-WQBox.nc']
NUM_MASKS = len(FILEPATH_MASKS)

# dates
START_DATE = '1961-01-01T00:00:00.000000000'
SPLIT_DATE = '1975-01-01T00:00:00.000000000'
END_DATE = '1979-01-01T00:00:00.000000000'

# plotting
DATA_CRS = ccrs.PlateCarree()
CMAP = [cmr.get_sub_cmap('GnBu', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1), 
        cmr.get_sub_cmap('coolwarm', 0, 1), cmr.get_sub_cmap('PuBu', 0, 1)] # make a dict

# model
FILEPATH_MODEL = 'model.pkl'
HAPPY_W_MODEL = False
ARCHITECTURE = 'cyclone' #choose from 'river', 'cyclone', 'unet', or 'initial'

# evaluation
EVAL_METHODS = ['MeanSquaredError', 'MeanSquaredLogarithmicError', 'mean_absolute_error']
TEST = True
ENSEMBLE_SIZE = 120
START_TIME = time.time()

# %% plotting functions - double check
def determine_num_subplots(num_var):
   '''Based on how many subplots are needed, calculate the dimensions for 
   the plots'''

   if num_var <= 3:
      x, y = 1, num_var
   else:
      x = y = math.ceil(num_var/2)

   return x, y

def plot_var(data_plot, var, subplot, x_dim, y_dim):
  '''Given a slice of an xarray dataset and a specific variable, plots the 
  values of the region. Also needs specification on the index x dimension
  and y dimension for the allocation of the subplot'''

  ax = plt.subplot(x_dim, y_dim, subplot+1, projection=DATA_CRS)
  ax.set_extent([min(data_plot.lon), max(data_plot.lon), 
                 min(data_plot.lat), max(data_plot.lat)], 
                 crs=DATA_CRS)
  ax.coastlines(resolution='50m')

  if data_plot[var].attrs['units'] == '%':    # need to implement the use of colorbars
      im = data_plot[var].plot(cmap=cmr.get_sub_cmap('GnBu', 0, 1), add_colorbar=False, vmin=0,vmax=100)
  else:
      im = data_plot[var].plot(cmap=cmr.get_sub_cmap('GnBu', 0, 1), add_colorbar=False)

  cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.07) 
  cbar.set_label(data_plot[var].attrs['units'])
  plt.title(data_plot[var].attrs['long_name'], fontsize=12)

def plot_all_vars(ds, vars, date, filename='', title=''):
   '''Given an xarray dataset, plots all the variables in the variable 
   list in subplots. Only works on geogaphic data, on a single day. Can 
   also save the plot and give a title.'''

   data_plot = ds.sel(time=slice(date, date))
   num_var = len(vars)

   fig = plt.figure(figsize=(13,5))
   fig.suptitle(title, fontsize=18)

   x,y = determine_num_subplots(num_var)
   for i in range(num_var):
      plot_var(data_plot, vars[i], i, x, y)

   if filename != '':
      plt.savefig(filename)
   else:
      plt.show()



# %% pre-processing functions
def standardize(ds, vars):
   '''Standardize all the variables in a dataset using the min-max normalization 
   scaling, and returns the standardized dataset as an Xarray.DataArray.'''

   for var in vars:
      ds[var] = (ds[var] - ds[var].values.min())/(ds[var].values.max() - ds[var].values.min())
   return ds

def reshape_vars(ds, vars): # try to combine with reshape masks
   '''Given a dataset, seperates the independent and dependent variables and reshapes the 
   arrays to fit the model format (number of data points, number of latitude points, 
   number of longitude points, number of variables). Returns both as numpy arrays.'''
   
   X = []
   for i in range(len(vars)-1):
      X.append(ds[vars[i+1]])
   X = np.moveaxis(np.asarray(X), 0, -1)

   Y = np.expand_dims(ds[vars[0]].values, 3)
   return X,Y

def reshape_masks(masks, train_shape, test_shape): #
   x_masks = [] # would really like to combine this with the other reshape function
   for var in list(masks.keys()):
      if masks[var].max() > 2: # filter out binary values - im sure there is a better way
         # masks[var] = standardize(masks, var) - hope I can implement this
         masks[var] = (masks[var] - masks[var].values.min())/(masks[var].values.max() - masks[var].values.min())
      x_masks.append(masks[var])
   x_masks = np.moveaxis(np.array(x_masks), 0, -1)

   x_masks_train = np.tile(x_masks,(train_shape, 1, 1, 1))
   x_masks_test = np.tile(x_masks,(test_shape, 1, 1, 1))
   
   return x_masks_train, x_masks_test

def read_datafile(path, num_vars):
   '''Read multiple file paths and then combine subsequent files into one 
   Xarray.DataArray and return it.'''

   data=0
   if num_vars>0:

      data = xr.open_dataset(path[0])
      for i in range(num_vars-1):
         t = xr.open_dataset(path[i+1])
         data = xr.merge([data, t])

   return data

def get_key_info(ds):
   '''Given a dataset, returns a list of numpy arrays containing the unique values of each 
   corrisponding index variable where the unique attributes are time, lat and lon'''

   start_index = np.where(ds.time.values == np.datetime64(START_DATE))
   end_index = np.where(ds.time.values == np.datetime64(END_DATE))
   keys = [ds.time.values[int(start_index[0][0]):int(end_index[0][0])+1], 
            ds.lon.values, ds.lat.values]
   
   return keys

def process_data(reduce):
   '''Given data files, reads them, combindes them into one XArray Dataset. It is assumed 
   that there is always data, but not necessarily masks.If the region needs to be reduced, 
   it is, to the desired region. From this dataset, the variables and keys are extracted 
   and the data is standardized. Finally, the datasets are split along the test and train 
   based on the dates defined in constants'''

   data = read_datafile(FILEPATH_DATA, NUM_VARS)
   if NUM_MASKS > 0:
      masks = read_datafile(FILEPATH_MASKS, NUM_MASKS)

   if reduce: 
      data = data.sel(lat=slice(35, 55), lon=slice(0,15)) #lat -90 to 90 lon -30 to 60 for cyclone 70 by 70 worked (try 80 by 80 on a new open)
      if NUM_MASKS > 0:
         masks = masks.sel(latitude=slice(55, 35), longitude=slice(0,15))

   var_list = list(data.keys())
   stand_data = standardize(data.sel(time=slice(START_DATE, END_DATE)), var_list)
   keys = get_key_info(data)

   x_train, y_train = reshape_vars(stand_data.sel(time=slice(START_DATE, SPLIT_DATE)), var_list)
   x_test, y_test = reshape_vars(stand_data.sel(time=slice(SPLIT_DATE, END_DATE)), var_list) 

   if NUM_MASKS > 0:
      x_masks_train, x_masks_test = reshape_masks(masks, x_train.shape[0], x_test.shape[0])
      x_train = np.concatenate((x_train, x_masks_train), axis=-1)
      x_test = np.concatenate((x_test, x_masks_test), axis=-1)

   plot_all_vars(data, var_list, SPLIT_DATE, 'figures/original_data.png')
   data.close()

   return var_list, keys, x_train, y_train, x_test, y_test

def get_data(reduce=False):
   '''Using a list of filepaths for the data, and a file path for the location of the 
   cleaned data, checks if the data was already read and processed. If yes, 
   obtains only the cleaned data. If not, opens the data, cleans it, and saves it
   in a new file. Returns both the data and a list of present variables.'''
   
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
      var_list, keys, x_train, y_train, x_test, y_test = process_data(reduce)
      
      with open(FILEPATH_CLEANED_DATA, 'wb') as f:
         pickle.dump({'variables': var_list, 'keys': keys,
                      'x_train': x_train, 'y_train': y_train,
                      'x_test': x_test, 'y_test': y_test}, f)

   return var_list, keys, x_train, y_train, x_test, y_test



# %% model funcions
def unet_encoder_block(inputs, num_filters, bottleneck=False): # work on documentation here down
   #geeksforgeeks
  
   # Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(inputs) 
   x = Activation('relu')(x) 
      
   # Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(x) 
   x = Activation('relu')(x) 
  
   if not bottleneck:
      # Max Pooling with 2x2 filter 
      x = MaxPool2D(pool_size = (2, 2), strides = 2)(x) 
      
   return x

def unet_decoder_block(inputs, skip_features, num_filters): 

	# Upsampling with 2x2 filter
   x = Conv2DTranspose(num_filters, (2, 2), strides = 2, padding = 'same')(inputs) 
	
	# Copy and crop the skip features 
	# to match the shape of the upsampled input 
   skip_features = Resizing(x.shape[1], x.shape[2])(skip_features)
   x = Concatenate()([x, skip_features]) 
	
	# Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(x) 
   x = Activation('relu')(x) 

	# Convolution with 3x3 filter followed by ReLU activation 
   x = Conv2D(num_filters, 3, padding = 'same')(x) 
   x = Activation('relu')(x) 
	
   return x

def unet_maker(x):
      '''Builds a model with U-Net architecture 
      '''
      inputs = Input(shape=(x.shape[1],x.shape[2],x.shape[3]))

      # Contracting Path (Encoder)
      c1 = unet_encoder_block(inputs, 64)
      c2 = unet_encoder_block(c1, 128)
      c3 = unet_encoder_block(c2, 256)
      c4 = unet_encoder_block(c3, 512)

      # Bottleneck
      b1 = unet_encoder_block(c4, 1024, bottleneck=True)

      # Expanding Path (Decoder)
      c5 = unet_decoder_block(b1, c4, 512) 
      c6 = unet_decoder_block(c5, c3, 256) 
      c7 = unet_decoder_block(c6, c2, 128) 
      c8 = unet_decoder_block(c7, c1, 64) 

      # Output layer
      c8 = Resizing(x.shape[1], x.shape[2])(c8)
      outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(c8)

      # Define the model
      model = Model(inputs, outputs, name='U-Net')
      return model

def weather_model_maker(x, filters_n_kernels):
      inputs = Input(shape=(x.shape[1],x.shape[2],x.shape[3]))

      c1 = Conv2D(filters=filters_n_kernels[0], kernel_size=filters_n_kernels[1], 
                    activation='relu', 
                    input_shape=(x.shape[1],x.shape[2],x.shape[3]),
                    padding='same')(inputs)
      p1 = MaxPool2D(pool_size = filters_n_kernels[2], strides = 2)(c1)
      c2 = Conv2D(filters=filters_n_kernels[3], kernel_size=filters_n_kernels[4], 
                    activation='relu', 
                    input_shape=(x.shape[1],x.shape[2],x.shape[3]),
                    padding='same')(p1)
      p2 = MaxPool2D(pool_size = filters_n_kernels[5], strides = 2)(c2)
      c3 = Dense(filters_n_kernels[6], activation='relu')(p2)
      c4 = Dense(1, activation='sigmoid')(c3)
      
      outputs = Resizing(x.shape[1], x.shape[2])(c4)

      model = Model(inputs, outputs)
      return model

def initial_model_maker(x):
   model = Sequential()
   model.add(Conv2D(filters=30, kernel_size=(3,3), 
                    activation='relu', 
                    input_shape=(x.shape[1],x.shape[2],x.shape[3]),
                    padding='same'))

   for layer in range(2):
      model.add(Conv2D(filters=32*(layer+1), 
                       kernel_size=(3,3), activation='relu',
                       padding='same'))
      model.add(LeakyReLU(alpha=0.1))

   model.add(Dropout(0.5))
   model.add(Dense(1, activation='sigmoid')) # mention sigmoid needed for percentage range

   return model

def build_model(x, architecture= ''):
   '''Gathers the parameters from the desired architecture and compiles 
   and returns the model.'''

   if architecture.lower() == 'unet':
      # Paper: Convolutional Networks for Biomedical Image Segmentation
      model = unet_maker(x)
   elif architecture.lower() == 'cyclone':
      # Paper: Application of Deep Convolutional Neural Networks for Detecting 
      # Extreme Weather in Climate Datasets
      model = weather_model_maker(x, [8, (5,5), (2,2), 16, (5,5), (2,2), 50])
   elif architecture == 'river':
      # Paper: Application of Deep Convolutional Neural Networks for Detecting 
      # Extreme Weather in Climate Datasets
      model = weather_model_maker(x, [8, (12,12), (3,3), 16, (12,12), (2,2), 200])
   else:
      print('Original architecture being used (possible name was not known)')
      model = initial_model_maker(x)

   model.compile(optimizer='adam', loss= EVAL_METHODS[0], metrics=EVAL_METHODS[1:])
   
   return model

def load_model(FILEPATH_MODEL, train, train_label, valid, valid_label):
   '''Check if a CNN model is already built. If so, load the model from 
   the given pickle filepath. If not build it given the dimensions from 
   the dataset x, and save it to the filepath.'''

   if os.path.isfile(FILEPATH_MODEL) and HAPPY_W_MODEL:
      print('Model already built')
      model = pickle.load(open(FILEPATH_MODEL, 'rb'))
   else:
      print('Building model.....')
      model = build_model(df_train, ARCHITECTURE)
      # print(model.summary())
      model.fit(train, train_label, epochs=10, batch_size=len(train)//10, 
          validation_data=(valid, valid_label), verbose=0)
      pickle.dump(model, open(FILEPATH_MODEL, 'wb'))

   return model



# %% evaluate functions - add some evaluation
def build_compare_xarray(pred, real, keys):
   '''Constructs a xarray dataset that contains the predicted values, 
   real values and the difference between the two for each day in 
   the test dataset. Returns that as a Xarray.DataArray'''

   start_index = np.where(keys[0] == np.datetime64(SPLIT_DATE))
   end_index = np.where(keys[0] == np.datetime64(END_DATE))
   pred_time = keys[0][int(start_index[0][0]):int(end_index[0][0])+1], 

   target = np.array(pred).flatten().reshape(len(pred_time[0]),
                                               len(keys[2]),len(keys[1]))
   real = np.squeeze(real, axis=-1)

   # Convert to percentages
   target = target*100
   real = real*100
   diff = target-real

   compare = xr.Dataset(
      data_vars=dict(
         pred_tcc = (['time', 'lat', 'lon'], target, {'units': '%', 
                      'long_name': 'Predicted Total Cloud Cover'}),
         real_tcc = (['time', 'lat', 'lon'], real, {'units': '%', 
                      'long_name': 'Real Total Cloud Cover'}),
         diff_tcc = (['time', 'lat', 'lon'], diff, 
                     {'units': 'Diff in %',
                      'long_name': 'Difference'})),
      coords = dict(
         time = ('time', pred_time[0]),
         lat = ('lat', keys[2]),
         lon = ('lon', keys[1])),
      attrs = dict(description="Real total cloud cover compared to \
                 predicted from CNN")
   )
   return compare

def calc_ensemble_metrics(num_simulations, train, train_label, valid, valid_label):
   # need to make this depedent on metrics list
   mse = np.zeros(num_simulations)
   msle = np.zeros(num_simulations)
   mae = np.zeros(num_simulations)
   times = np.zeros(num_simulations)

   for i in range(num_simulations):
      model = load_model(FILEPATH_MODEL, train,
                  train_label, valid, valid_label)
   
      eval = model.evaluate(x_test, y_test)
      mse[i] = eval[0]
      msle[i] = eval[1]
      mae[i] = eval[2]
      times[i] = time.time() - sum(times) - START_TIME

   with open('metrics/'+ARCHITECTURE+'.txt', 'a') as f:
      f.write('MSE: ')
      for i in range(num_simulations):
         f.write(str(mse[i])+' ')
      f.write('\n')

      f.write('MSLE: ')
      for i in range(num_simulations):
         f.write(str(msle[i])+' ')
      f.write('\n')

      f.write('MAE: ')
      for i in range(num_simulations):
         f.write(str(mae[i])+' ')
      f.write('\n')

      f.write('Times: ')
      for i in range(num_simulations):
         f.write(str(times[i])+' ')
      f.write('\n')

   return mse, msle, mae 

def calc_diff_metrics(compare):
   total_off = np.zeros(len(compare['diff_tcc'].values))
   avg = np.zeros(len(compare['diff_tcc'].values))
   i=0

   for day in compare['diff_tcc'].values:
      day = np.absolute(day)
      total_off[i] = day.sum()
      avg[i] = total_off[i]/(day.shape[0]*day.shape[1])
      i+=1

   return total_off, avg

def evaluate_model(model, test, test_labels, keys):
   '''Given a trained model, use the test dataset and corect 
   labels to evaluate the performance.'''

   eval = model.evaluate(test, test_labels)
   for i in range(len(eval)):
      print(EVAL_METHODS[i], end=': ')
      print(round(eval[i],5))
      
   y_pred = model.predict(test)
   print(y_pred.max())
   compare = build_compare_xarray(y_pred, test_labels, keys)

   total_off, avg = calc_diff_metrics(compare)

   title = 'Comparison for model using '
   for var in FILEPATH_DATA[1:]:
      var_name = var.split('-')[1]
      title = title + var_name + ' '

   worst_day_avg = np.where(avg == avg.max())[0][0]
   plot_all_vars(compare, list(compare.keys()), compare['time'].values[worst_day_avg], 
                 'figures/worst_day.png', title+'- worst day')
   print("The worst day predicted average distance per cell:", avg.max())
   
   best_day_avg = np.where(avg == avg.min())[0][0]
   plot_all_vars(compare, list(compare.keys()), compare['time'].values[best_day_avg], 
                 'figures/best_day.png', title+'- best day')
   print("The best day predicted average distance per cell:", avg.min())

   plot_all_vars(compare, list(compare.keys()), END_DATE, 'figures/comparison.png',
                  title) 



# %% main
clear_session()

var_list, keys, x_train, y_train, x_test, y_test = get_data(NUM_VARS!=0)

df_train, df_valid, df_train_label, df_valid_label = train_test_split(x_train, y_train, 
                                                                     test_size=0.2,
                                                                     random_state=13)

# here make that the mask variables work
if TEST:
   # Multiple - create ensembles
   calc_ensemble_metrics(ENSEMBLE_SIZE, df_train, df_train_label, df_valid, 
                              df_valid_label)
else:
   # One model
   model = load_model(FILEPATH_MODEL, df_train,
                     df_train_label, df_valid, df_valid_label)
   evaluate_model(model, x_test, y_test, keys)