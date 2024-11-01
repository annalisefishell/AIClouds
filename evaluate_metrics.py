import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def confidence_interval(ensemble, confidence=0.95):
  n = len(ensemble)
  sem = np.std(ensemble, ddof=1) / np.sqrt(n)

  z_score= st.norm.ppf(confidence)
  margin_of_error = z_score * sem

  confidence_interval = (np.mean(ensemble) - margin_of_error, np.mean(ensemble) + margin_of_error)
  return confidence_interval

def print_metrics(mse, msle, mae, times, num_sims):
  print('MSE mean:', round(sum(mse)/num_sims,5))
  print('MSE conf:', confidence_interval(mse))
  
  print('MSLE mean:', round(sum(msle)/num_sims,5))
  print('MSLE conf:', confidence_interval(msle))

  print('MAE mean:', round(sum(mae)/num_sims,5))
  print('MAE conf:', confidence_interval(mae))

  print('Time mean', round(sum(times)/num_sims,5))
  print('Time mean', confidence_interval(times))

def read_file(architecture):
  temp_dict = {}
  file = open("metrics"+architecture+".txt", "r").readlines()
  for line in file:
    line = line.split(":")
    temp_dict[line[0]] = line[1].split()
  return temp_dict

def plot_one_metric(ax, arch_list, metric, color):
  for key,value in arch_list:
    counts, bins = np.histogram(value[metric])
    ax.hist(bins[:-1], bins, weights=counts, density=True,
           color=color, alpha=0.6, rwidth=0.9, label=key)
    kde = st.gaussian_kde(value[metric])
    x = np.linspace(min(value[metrics]), max(value[metrics]), 1000)
    y = kde(x)
    ax.plot(x, y, color=color)
  ax.grid(axis='y', alpha=0.7)
  ax.set_xlabel(metric, fontsize=12)
  ax.set_ylabel("Density", fontsize=12)
  ax.legend()

architechts = {
    'initial' : read_file('initial'),
    'cyclone' : read_file('cyclone'),
    'river' : read_file('river'),
    'unet' : read_file('unet')
}
metrics = ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error']
colors = ['r', 'g', 'b', 'k']

fig = plt.figure(figsize=(15,5))
fig.suptitle('Distribution of metrics', fontsize=18)
for i in range(len(metrics)):
  ax = plt.subplot(1,3,i)
  plot_one_metric(ax, architechts, metrics[i], colors[i])
plt.savefig('figures/metrics_dist.png')

metrics = ['Times']
fig = plt.figure(figsize=(15,5))
fig.suptitle('Distribution of metrics', fontsize=18)
for i in range(len(metrics)):
  ax = plt.subplot(1,1,i)
  plot_one_metric(ax, architechts, metrics[i], colors[i])
plt.savefig('figures/time_dist.png')

for key,value in architechts:
  print('For achitecture ' + key + ', the mean metrics with confidence intervals are:')
  print_metrics(value['MSE'], value['MSLE'], value['MAE'], value['Times'], len(value['Times']))