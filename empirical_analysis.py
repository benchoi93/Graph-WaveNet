import numpy as np
import pandas as pd
import torch
import seaborn as sns
from matplotlib import pyplot as plt


df = pd.read_hdf("/app/data/pems-bay.h5")
df = torch.FloatTensor(df.values)

B = df.shape[0]
N = df.shape[1]
T = 12

# contruct BxNxT tensor with delay embedding
data = torch.concat([df[i:(B-T+i), :].unsqueeze(-1) for i in range(T)], -1)
data_flatten = data.reshape(data.shape[0], N * T)

sns.heatmap(torch.cov(df.T))
plt.show()


# compute empirical covariance matrix
cov_flat = torch.cov(data_flatten.T).cpu().numpy()
i = 3
j = 4
sns.heatmap(cov_flat[325*i:325*(i+1), 325*j:325*(j+1)])
plt.show()


sns.heatmap(cov_flat)
plt.show()
