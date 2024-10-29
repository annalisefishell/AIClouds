import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmasher as cmr
import math
import os
import scipy.stats as st

def confidence_interval(ensemble, confidence=0.95):

   n = len(ensemble)
   sem = np.std(ensemble, ddof=1) / np.sqrt(n)

   z_score= st.norm.ppf(confidence)
   # z_score = np.abs(np.percentile(np.random.normal(0, 1, 10**6), [100 * (1-confidence) / 2, 100 * (1 - (1-confidence) / 2)]))
   margin_of_error = z_score * sem

   confidence_interval = (np.mean(ensemble) - margin_of_error, np.mean(ensemble) + margin_of_error)
   return confidence_interval

##### Read the data from metrics.txt

print('MSE mean:', round(sum(mse)/num_simulations,5))
print('MSLE mean:', round(sum(msle)/num_simulations,5))
print('MAE mean:', round(sum(mae)/num_simulations,5))
print('Time mean', round(sum(times)/num_simulations,5))

# not completely correct yet, gives two arrays
print('MSE conf:', confidence_interval(mse))
print('MSLE conf:', confidence_interval(msle))
print('MAE conf:', confidence_interval(mae))
print('Time mean', confidence_interval(times))

# histogram of the errors to see distributions
fig = plt.figure(figsize=(10,10))
title = 'Distribution of metrics for architecht: ' + ARCHITECTURE + '\nwith variables '
for var in FILEPATH_DATA[1:]:
  var_name = var.split('-')[1]
  title = title + var_name + ', '
fig.suptitle(title, fontsize=18)

ax = plt.subplot(2, 2, 1)
counts, bins = np.histogram(mse)
ax.hist(bins[:-1], bins, weights=counts, density=True,
           color='#607c8e', alpha=0.8,  rwidth=0.9)
ax.grid(axis='y', alpha=0.6)
ax.set_xlabel('Mean squared error', fontsize=12)
ax.set_ylabel("Density", fontsize=12)
kde = st.gaussian_kde(mse)
x = np.linspace(min(mse), max(mse), 1000)
y = kde(x)
ax.plot(x, y)

ax = plt.subplot(2, 2, 2)
counts, bins = np.histogram(msle)
ax.hist(bins[:-1], bins, weights=counts, density=True,
           color='#607c8e', alpha=0.8,  rwidth=0.9)
ax.grid(axis='y', alpha=0.6)
ax.set_xlabel('Mean squared logarithmic error', fontsize=12)
kde = st.gaussian_kde(msle)
x = np.linspace(min(msle), max(msle), 1000)
y = kde(x)
ax.plot(x, y)

ax = plt.subplot(2, 2, 3)
counts, bins = np.histogram(mae)
ax.hist(bins[:-1], bins, weights=counts, density=True,
           color='#607c8e', alpha=0.8,  rwidth=0.9)
ax.grid(axis='y', alpha=0.6)
ax.set_xlabel('Mean absolute error', fontsize=12)
kde = st.gaussian_kde(mae)
x = np.linspace(min(mae), max(mae), 1000)
y = kde(x)
ax.plot(x, y)

ax = plt.subplot(2, 2, 4)
counts, bins = np.histogram(times)
ax.hist(bins[:-1], bins, weights=counts, density=True,
           color='#607c8e', alpha=0.8,  rwidth=0.9)
ax.grid(axis='y', alpha=0.6)
ax.set_xlabel('Time (s)', fontsize=12)
kde = st.gaussian_kde(times)
x = np.linspace(min(times), max(times), 1000)
y = kde(x)
ax.plot(x, y)

plt.savefig('figures/distributions')