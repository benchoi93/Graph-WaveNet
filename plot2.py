import pandas as pd
import os
os.chdir("/app")
df = pd.read_csv("newfile.csv")

# 	Name	2	5	8	11	Unnamed: 5
# 0	0	2.635	3.460	3.877	4.162	rmse
# 1	1	2.581	3.396	3.771	4.002	rmse
# 2	2	2.580	3.406	3.811	4.056	rmse
# 3	3	2.573	3.385	3.776	4.007	rmse
# 4	4	2.573	3.397	3.810	4.061	rmse

# draw bar plot depending on the last colume (rmse/mae/mape)
# x axis is "Name"
# y axis is value


import seaborn as sns
import matplotlib.pyplot as plt

df.columns = ['n', '15min', '30min', '45min', '60min', 'measure']
df = df.dropna()

# if df['n'] = 0 then change it to "baseline"
df['n'] = df['n'].apply(lambda x: 'baseline' if x == 0 else x)

df = pd.melt(df, id_vars=['n', 'measure'], value_vars=['15min', '30min', '45min', '60min'])

# color: first element = red
# following 10 elements randomly selected
cmap = sns.color_palette("muted", 11)
cmap = [(1,0,0)] + cmap[1:]

fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(15, 5))
sns.barplot(x='variable', y='value', hue='n', data=df[df['measure'] == 'rmse'], ax=ax[0], palette=cmap)
ax[0].set_ylabel('RMSE')
ax[0].set_ylim(2.3, 4.3)

sns.barplot(x='variable', y='value', hue='n', data=df[df['measure'] == 'mape'], ax=ax[1], palette=cmap)
ax[1].set_ylabel('MAPE')
ax[1].set_ylim(2.3, 5.2)

sns.barplot(x='variable', y='value', hue='n', data=df[df['measure'] == 'mae'], ax=ax[2], palette=cmap)
ax[2].set_ylabel('MAE')
ax[2].set_ylim(1, 2.3)

plt.tight_layout()
# delete legend
ax[0].legend_.remove()
ax[1].legend_.remove()
# ax[2].legend_.remove()

# font 20 for everything
for i in range(3):
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    # ax[i].set_xlabel('')
    # y ticks 1 decimal
    ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))
    ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=20)
    ax[i].set_xlabel('Prediction Horizon', fontsize=20)   

# space between subplots
plt.subplots_adjust(wspace=0.3)

# ax[2] legend center right
ax[2].legend(loc='center right', bbox_to_anchor=(1.6, 0.5), title='K', title_fontsize='15', fontsize='15')

plt.tight_layout()

plt.savefig('plot2.pdf')