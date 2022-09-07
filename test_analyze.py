import ast
import pandas as pd


def list_eval(str_in):
    return [float(x) for x in str_in[1:-1].split(" ") if x != ""]


name = "out_kronecker_0904"
df = pd.read_csv(f"/app/{name}.csv")

result = []
for i in range(df.shape[0]):
    model_path = df.iloc[i][1]
    params = model_path.split("_")[3:]

    n_components = int(params[0].split("N")[1])
    num_rank = int(params[1].split("R")[1])
    reg_coef = float(params[2].split("reg")[1])
    nhid = int(params[3].split("nhid")[1])
    pred_len = [2, 5, 8, 11]
    rho = float(params[5].split("rho")[1])
    diag = True if params[6].split("diag")[1] == "True" else False
    mse_coef = float(params[7].split("coef")[1])

    rmse = list_eval(df.iloc[i][2])
    mape = list_eval(df.iloc[i][3])
    mae = list_eval(df.iloc[i][4])
    crps = list_eval(df.iloc[i][5])
    ES = list_eval(df.iloc[i][6])

    for t in range(4):
        result.append(
            {
                "n_components": n_components,
                "num_rank": num_rank,
                "reg_coef": reg_coef,
                "nhid": nhid,
                "pred_len": (pred_len[t]+1) * 5,
                "rho": rho,
                "diag": diag,
                "mse_coef": mse_coef,
                "rmse": rmse[t],
                "mape": mape[t],
                "mae": mae[t],
                "crps": crps[t],
                "ES": ES[t]
            }
        )


result_pd = pd.DataFrame(result)
result_pd.to_csv(f"/app/{name}_analyzed.csv")


# df_train = pd.read_csv(f"/app/out_train.csv")
# df_val = pd.read_csv(f"/app/out_val.csv")
# df_test = pd.read_csv(f"/app/out_test.csv")

# df_train["type"] = "train"
# df_val["type"] = "val"
# df_test["type"] = "test"

# df = pd.concat([df_train, df_val, df_test])

# df[df["rho"].isin([0.0, 0.01])].groupby(["n_components","type","rho"]).mean()[["rmse","mape","mae","crps"]]
