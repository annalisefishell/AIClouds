# %% imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import main

# %% constants 
COLORS = ['#606867', '#f5544c', '#2770af', '#BFBC28']
LIMITS = [(0.04, 0.15), (0.02, 0.07), (0.15, 0.3)]
METRICS = main.EVAL_METHODS

# %% Process data
def read_file(architecture):
  '''Reads a text file containing the information of the outcomes 
  of experiments. Each line in the file should start with the name of
  a metric in the METRICS constant. '''
  temp_dict = {}
  file = open("metrics/"+architecture+".txt", "r").readlines()

  for line in file:
    line = line.split(":")
    values = [float(x) for x in line[1].split()]

    if line[0] in temp_dict.keys():
      temp_dict[line[0]] += values
    else:
      temp_dict[line[0]] = values

  return temp_dict

def find_outliers(data):
  '''Calculates the outlier(s) of a list and returns them.'''
  Q1 = np.percentile(data, 25)
  Q3 = np.percentile(data, 75)
  IQR = Q3 - Q1
    
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
    
  outliers = [x for x in data if x < lower_bound or x > upper_bound]
    
  return outliers

def remove_outliers(list):
  '''Takes a list, finds the outliers, prints them and then removes 
  them from the list and returns the new one'''
  for arch in list.keys():
    for metric in METRICS:
      outliers = find_outliers(list[arch][metric])

      print(arch, end="")
      print(outliers)

      for value in outliers:
        list[arch][metric].remove(value)

  return list

def read_files():
  '''Reads all metric files, save the different test runs
  in different dictionaries and returns them.'''
  architechts = {
    'cyclone' : read_file('cyclone'),
    'river' : read_file('river'),
    'initial' : read_file('initial'),
    'unet' : read_file('unet')}

  vars = {
    'water' : read_file('water'),
    'temp' : read_file('temp'),
    'precip' : read_file('precip')}

  vars_masked = {
    'water_m' : read_file('water_masked'),
    'temp_m' : read_file('temp_masked'),
    'precip_m' : read_file('precip_masked')}
  
  return architechts, vars, vars_masked

# %% printing output
def confidence_interval(ensemble, confidence=0.95):
  '''Calculates the confidence interval around the mean for a list
  of data points. Default confidence percentage is 95%.'''
  n = len(ensemble)
  sem = np.std(ensemble, ddof=1) / np.sqrt(n)

  z_score= st.norm.ppf(confidence)
  margin_of_error = z_score * sem

  confidence_interval = (np.mean(ensemble) - margin_of_error, np.mean(ensemble) + margin_of_error)
  return confidence_interval

def print_metrics(mse, msle, mae, times, num_sims):
  '''Prints the means and confidence intervals for all metrics 
  for one experiment.'''
  print('MSE mean:', round(sum(mse)/num_sims,5))
  print('\tMSE conf:', confidence_interval(mse))
  
  print('MSLE mean:', round(sum(msle)/num_sims,5))
  print('\tMSLE conf:', confidence_interval(msle))

  print('MAE mean:', round(sum(mae)/num_sims,5))
  print('\tMAE conf:', confidence_interval(mae))

  print('Time mean', round(sum(times)/num_sims,5))
  print('\tTime mean', confidence_interval(times))
  print()

# %% Plotting
def plot_one_metric(ax, arch_list, metric, i=0, want_axis_labels=True, 
                    is_arch=False):
  '''Given one list representing a the values of a metric from experiment runs,
  plots that list. Also need to specify the axis to plot on.'''
  for key,value in arch_list.items():
    ax.hist(value[metric], bins=15, density=True,
           color=COLORS[i], alpha=0.6, rwidth=0.9, label=key)
    
    kde = st.gaussian_kde(value[metric])
    x = np.linspace(min(value[metric]), max(value[metric]), 1000)
    y = kde(x)
    ax.plot(x, y, color=COLORS[i], linewidth=2.5)
    i+=1
    # x_label = key.title()
    # ax.set_title(x_label, fontsize=26) # For time uncomment

  if want_axis_labels:
    x_label = metric.replace("_", " ").title()
    ax.set_title(x_label, fontsize=24)
  ax.tick_params(axis='x', labelsize=20)

  if is_arch:
    ax.set_xlim(LIMITS[METRICS.index(metric)])
  ax.set_ylabel("Density", fontsize=24)
  ax.tick_params(axis='y', labelsize=20)

  ax.grid(axis='y', alpha=0.7)
  ax.legend(fontsize=22) # For time comment out

def plot_mean_metrics(arch_list, name, is_arch=False):
  '''Given a dictionary that contains key information and metric values, 
  plots a subplot. Need to give the name of the tests currently plotting.'''
  fig = plt.figure(figsize=(20,20))
  fig.suptitle('Distribution of metrics for '+name+' testing', fontsize=30)

  for i in range(len(METRICS)):
    ax = plt.subplot(3,1,i+1)
    plot_one_metric(ax, arch_list, METRICS[i], is_arch=is_arch)

  plt.tight_layout()
  plt.subplots_adjust(top=0.93)

  plt.savefig('figures/metrics_'+name+'.png')

def plot_time(arch_list):
  '''Given a dictionary that contains key information and time values, 
  plots the times for all dictionary items in seperate plots.'''
  fig = plt.figure(figsize=(25,20))
  fig.suptitle('Distribution of Running Time (seconds)', fontsize=30)

  i=0
  for key in arch_list.keys():
    temp_dict = {key: arch_list[key]}
    ax = plt.subplot(2,2,i+1)
    plot_one_metric(ax, temp_dict, 'Times', i, want_axis_labels=False)
    i += 1

  plt.tight_layout()
  plt.subplots_adjust(top=0.93)

  plt.savefig('figures/time_dist.png')

# %% Main code
architechts, vars, vars_masked = read_files()
architechts = remove_outliers(architechts)

all_tests = {
  'architechts' : architechts,
  'variables' : vars,
  'variables with masks' : vars_masked
}

for test_name,outcome in all_tests.items():
  plot_mean_metrics(outcome, test_name, False) # Use true for the arch plot

  for key,value in outcome.items():
    print('For ' + key + ', the mean metrics with 95'+'%'+' confidence intervals are:')
    print_metrics(value[METRICS[0]], value[METRICS[1]], value[METRICS[2]], value['Times'], len(value['Times']))

plot_time(architechts)