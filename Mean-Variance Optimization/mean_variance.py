import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

# Read data from files
os.chdir("D:/Navid/UW/MS CFRM/Courses/CFRM 501/HomeWorks/HW4/Week 4 Data Files")
name = []
info = []
for f in os.listdir():
    file_path = os.path.join(os.getcwd(), f)
    filename, file_ext = os.path.splitext(f)
    if file_ext == '.csv':
        d = pd.read_csv(f)
        info.append(d['Adj Close'].values[1:] / d['Adj Close'].values[:-1] - 1)
        name.append(filename)
# convert data to matrix
Table = dict(zip(name, info))
data = pd.DataFrame(Table)
data = pd.DataFrame.to_numpy(data)
data = np.asmatrix(data)
# calculate daily and yearly return and covariance matrix
one_vector = np.ones((10, 1))
daily_return = data.mean(axis=0).T
yearly_return = np.power(daily_return + np.ones((10, 1)), one_vector * 252) - one_vector
daily_cov = np.cov(data, rowvar=False, bias=True)
yearly_cov = np.power(daily_cov + (1 + daily_return) * (1 + daily_return).T, np.ones((10, 10)) * 252) - \
             np.power((1 + daily_return) * (1 + daily_return).T, np.ones((10, 10)) * 252)
# calculate correlation from convarince matrix
v = np.sqrt(np.diag(yearly_cov))
outer_v = np.outer(v, v)
correlation = yearly_cov / outer_v
correlation[yearly_cov == 0] = 0
# new correlation which has 0.1 difference in cross correlation
change_factor = -0.1
change_corr = correlation + np.ones((10, 10)) * change_factor
np.fill_diagonal(change_corr, val=1)
new_yearly_cov = np.multiply(change_corr, outer_v)


# finding global minimum variance
def global_min(asset_cov, asset_return):
    A = float(one_vector.T * np.linalg.inv(asset_cov) * one_vector)
    B = float(one_vector.T * np.linalg.inv(asset_cov) * asset_return)
    C = float(asset_return.T * np.linalg.inv(asset_cov) * asset_return)
    global_portfolio_phi = A * C - (B ** 2)
    global_portfolio_var = 1 / A
    global_portfolio_weight = np.linalg.inv(asset_cov) * one_vector * (1 / A)
    portfolio_mean = float(global_portfolio_weight.T * asset_return)
    # print(global_portfolio_phi)

    return global_portfolio_var, portfolio_mean, global_portfolio_weight, global_portfolio_phi


# sketch the graph for mean-variance boundary
def mean_var_graph(g_var, g_return, g_phi, graph_text, graph_color):
    portfolio_mean = np.arange(-1, 1, 0.001)
    diff = portfolio_mean - np.ones(len(portfolio_mean)) * g_return
    portfolio_std = np.sqrt(g_var + (np.multiply(diff, diff) * (1 / (g_phi * g_var))))
    plt.plot(portfolio_std, portfolio_mean, color=graph_color,label=f"Mean Variance Boundary{graph_text}")
    plt.plot(math.sqrt(g_var), g_return, 'o', label=f'global minimum variance{graph_text}')
    plt.xlabel('std')
    plt.ylabel('mean')
    plt.grid(True)



# sketch the assets points
def plot_points(data_table, asset_cov, asset_return):
    for i, asset in enumerate(data_table.keys()):
        plt.plot(math.sqrt(asset_cov[i, i]), asset_return[i, 0], 'o', label=asset)


global_var, global_return, global_weight, global_phi = global_min(yearly_cov, yearly_return)
new_global_var, new_global_return, new_global_weight, new_phi = global_min(new_yearly_cov, yearly_return)

mean_var_graph(global_var, global_return, global_phi,"","k" )
mean_var_graph(new_global_var, new_global_return, new_phi, " Corr-0.1","r")
plot_points(Table, yearly_cov, yearly_return)
plt.title('Mean Variance Boundary')
plt.legend(loc="upper left", ncol=2, prop={'size': 6})
plt.tight_layout()
plt.show()
# printing global minimum variance  weight, return and std
for i, asset in enumerate(Table.keys()):
    print("{}:Global wight: {}, Yearly std: {}, Yearly return: {} ".format(asset,
                                                                           global_weight[i, 0],
                                                                           math.sqrt(yearly_cov[i, i]),
                                                                           yearly_return[i, 0]))

# printing new global minimum variance  weight, return and std after changing the correlation
for i, asset in enumerate(Table.keys()):
    print("{}:New Global wight: {}, New Yearly std: {}, New Yearly return: {}".format(asset,
                                                                                      new_global_weight[i, 0],
                                                                                      math.sqrt(new_yearly_cov[i, i]),
                                                                                      yearly_return[i, 0]))


# printing global min variance return and std with and without change in correlation
print("Global return: ", global_return)
print("Global Std: ", math.sqrt(global_var))
print("Global return after change in Corr: ", new_global_return)
print("Global Std after change in Corr: ", math.sqrt(new_global_var))
