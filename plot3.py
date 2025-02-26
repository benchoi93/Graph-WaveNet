import pandas as pd
import os 
os.chdir("/app")
import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
data = pd.read_csv('newfile.csv')
metrics = ['rmse', 'mape', 'mae']
colors = ['red', 'green', 'blue', 'gold']
cols = ['15min', '30min', '45min', '60min']

# Plot setup
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

for ax, metric in zip(axs, metrics):
    subset = data[data['Unnamed: 5'] == metric]
    for idx, col in enumerate(['2', '5', '8', '11']):
        ax.plot(subset['Name'].values, subset[col].values, marker='o', color=colors[idx], label=f'Col {col}')
    ax.set_title(metric.upper())
    ax.set_xlabel('rho')
    ax.set_ylabel('Prediction Error')

ax.legend(title='Columns')
plt.tight_layout()
plt.show()