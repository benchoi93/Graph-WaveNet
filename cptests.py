import scipy
import pickle
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
import pandas as pd


DATA_X = np.load("/app/data/PEMS-BAY/train.npz")['x'][:, :, :, 0]

ranklist = [1, 3, 5, 7, 10, 20, 30, 50, 100, 200, 300]
result = []
# for rank in ranklist:
rank = 300
factors = parafac(DATA_X, rank=rank, n_iter_max=100, init='random', verbose=True, tol=1e-8)
recon = tl.cp_tensor.cp_to_tensor(factors)
mask = DATA_X != 0
# rmse = (((recon - DATA_X)**2).mean())**0.5
# rmse = (((reconstruction - TEST_X)**2) * mask ).sum() / mask.sum()
rmse = (((recon - DATA_X)**2) * mask).sum() / mask.sum()
# result.append((rank, rmse))
# pd.DataFrame(result, columns=['rank', 'rmse']).to_csv('/app/result.csv', index=False)
print(rmse)
with open(f'/app/factor_{rank}.pkl', 'wb') as f:
    pickle.dump(factors, f)


with open('/app/factor_300.pkl', 'rb') as f:
    factors = pickle.load(f)

TEST_X = np.load("/app/data/PEMS-BAY/test.npz")['y'][:, :, :, 0]

B_TEST = np.zeros((TEST_X.shape[0], 300))
factors[1][0] = B_TEST


input_a = scipy.linalg.khatri_rao(factors[1][1], factors[1][2])
target_a = tl.unfold(TEST_X, mode=0)

B_TEST = np.linalg.solve(input_a.T.dot(input_a), input_a.T.dot(target_a.T))
factors[1][0] = B_TEST.T
reconstruction = tl.cp_tensor.cp_to_tensor((factors[0], factors[1]))

mask = TEST_X != 0

dev = reconstruction - TEST_X

rmse = ((((reconstruction - TEST_X)**2) * mask).sum((0, 2)) / mask.sum((0, 2)))**0.5

mape_out = np.abs(dev)/TEST_X
mape_out[~mask] = 0
mape = mape_out.sum((0, 2)) / mask.sum((0, 2)) * 100

mae = np.abs((reconstruction - TEST_X) * mask).sum((0, 2)) / mask.sum((0, 2))

# for i in range(12):
for i in [2, 5, 8, 11]:
    print(f"{(i+1)*5} min estimation accuracy == rmse: {rmse[i]} , mape: {mape[i]} , mae: {mae[i]}")
