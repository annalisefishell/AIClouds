# %% imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# %% constants 
COLORS = ['#06af6b', '#606867', '#2770af', '#a0b202']
METRICS = ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error']

# %% Process data
def read_file(architecture):
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

# %% printing output
def confidence_interval(ensemble, confidence=0.95):
  n = len(ensemble)
  sem = np.std(ensemble, ddof=1) / np.sqrt(n)

  z_score= st.norm.ppf(confidence)
  margin_of_error = z_score * sem

  confidence_interval = (np.mean(ensemble) - margin_of_error, np.mean(ensemble) + margin_of_error)
  return confidence_interval

def print_metrics(mse, msle, mae, times, num_sims):
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
def plot_one_metric(ax, arch_list, metric, i=0, want_axis_labels=True):
  for key,value in arch_list.items():
    counts, bins = np.histogram(value[metric])
    ax.hist(bins[:-1], bins, weights=counts, density=True,
           color=COLORS[i], alpha=0.6, rwidth=0.9, label=key)
    
    kde = st.gaussian_kde(value[metric])
    x = np.linspace(min(value[metric]), max(value[metric]), 1000)
    y = kde(x)
    ax.plot(x, y, color=COLORS[i])
    i+=1

  if want_axis_labels:
    x_label = metric.replace("_", " ").title()
    ax.set_title(x_label, fontsize=24)
  ax.tick_params(axis='x', labelsize=20)

  ax.set_ylabel("Density", fontsize=24)
  ax.tick_params(axis='y', labelsize=20)

  ax.grid(axis='y', alpha=0.7)
  ax.legend(fontsize=22)

def plot_mean_metrics(arch_list): # maybe deal with unet outliers
  fig = plt.figure(figsize=(20,20))
  fig.suptitle('Distribution of metrics', fontsize=30) #center
  for i in range(len(METRICS)):
    ax = plt.subplot(3,1,i+1)
    plot_one_metric(ax, arch_list, METRICS[i])
  plt.tight_layout()
  plt.subplots_adjust(top=0.93)
  plt.savefig('figures/metrics_dist.png')

def plot_time(arch_list): # why is density wrong?
  fig = plt.figure(figsize=(25,20))
  fig.suptitle('Distribution of Running Time (seconds)', fontsize=28)
  i=0
  for key in arch_list.keys():
    temp_dict = {key: arch_list[key]}
    ax = plt.subplot(2,2,i+1)
    plot_one_metric(ax, temp_dict, 'Times', i, False)
    i += 1
  plt.tight_layout()
  plt.subplots_adjust(top=0.93)
  plt.savefig('figures/time_dist.png')

# %% Main code
architechts = {
    'initial' : read_file('initial'),
    'cyclone' : read_file('cyclone'),
    'river' : read_file('river'),
    'unet' : read_file('unet')}

plot_mean_metrics(architechts)
plot_time(architechts)

for key,value in architechts.items():
  print('For achitecture ' + key + ', the mean metrics with confidence intervals are:')
  print_metrics(value[METRICS[0]], value[METRICS[1]], value[METRICS[2]], value['Times'], len(value['Times']))